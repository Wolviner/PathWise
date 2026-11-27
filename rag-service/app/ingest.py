"""Ingestion: documents -> chunks -> embeddings -> rag_chunks (pgvector).

Two chunking variants are stored side by side so the evaluation can compare them:
  - "llm":   week-5 day-5 LLM chunking (headline + summary + original text). Used in production.
  - "basic": plain overlapping text splitting (the day-2/3 baseline).
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field

from . import llm
from .config import CHUNK_MODEL, CHUNK_WORKERS, GUIDES_DIR, SHARED_USER
from .db import connect, to_vector
from .knowledge_base import Document, build_documents, fetch_user_data, guide_documents

log = logging.getLogger("rag.ingest")

AVERAGE_CHUNK_SIZE = 500  # characters
BASIC_CHUNK_SIZE = 1000
BASIC_OVERLAP = 200


@dataclass
class ChunkRow:
    doc: Document
    headline: str
    summary: str
    content: str  # what gets embedded and shown to the LLM


class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk, a few words, likely to match a user's question")
    summary: str = Field(description="One or two sentences summarising this chunk, including the key amounts and dates")
    original_text: str = Field(description="The original text of this chunk from the document, exactly as is")


class Chunks(BaseModel):
    chunks: list[Chunk]


def _content(doc: Document, headline: str, summary: str, text: str) -> str:
    parts = [f"Source: {doc.title}", headline, summary, text]
    return "\n\n".join(p for p in parts if p)


def basic_chunks(doc: Document) -> list[ChunkRow]:
    """Overlapping splits on line boundaries, like RecursiveCharacterTextSplitter."""
    lines = doc.text.splitlines()
    chunks, current = [], []
    for line in lines:
        if current and len("\n".join(current + [line])) > BASIC_CHUNK_SIZE:
            chunks.append("\n".join(current))
            overlap, size = [], 0
            for prev in reversed(current):  # carry ~BASIC_OVERLAP chars into the next chunk
                if size + len(prev) > BASIC_OVERLAP:
                    break
                overlap.insert(0, prev)
                size += len(prev)
            current = overlap
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return [ChunkRow(doc, doc.title, "", _content(doc, "", "", text)) for text in chunks]


def _chunk_prompt(doc: Document) -> str:
    how_many = len(doc.text) // AVERAGE_CHUNK_SIZE + 1
    return f"""
You split a document into overlapping chunks for a knowledge base.

The document is part of one PathWise user's personal finance records.
Document type: {doc.doc_type}
Document title: {doc.title}

A chatbot will use these chunks to answer the user's questions about their own money.
Divide the document as you see fit, making sure the entire document is covered across the chunks; leave nothing out.
This document should probably be split into about {how_many} chunks (fewer for short documents), with individual chunks able to answer specific questions.
Overlap neighbouring chunks by about 25% so the same text appears in more than one chunk.
Keep every amount, date and name exactly as written. Repeat the month and account name in each chunk's headline.

For each chunk give a headline, a summary and the original text.

Here is the document:

{doc.text}
"""


def llm_chunks(doc: Document) -> list[ChunkRow]:
    try:
        result = llm.generate_json(_chunk_prompt(doc), Chunks, model=CHUNK_MODEL)
        rows = [
            ChunkRow(doc, c.headline, c.summary, _content(doc, c.headline, c.summary, c.original_text))
            for c in result.chunks
            if c.original_text.strip()
        ]
        if rows:
            return rows
        log.warning("LLM returned no chunks for %s; using basic chunking", doc.source)
    except llm.QuotaExceeded:
        raise
    except Exception as exc:  # keep ingesting even if one document fails
        log.warning("LLM chunking failed for %s (%s); using basic chunking", doc.source, exc)
    return basic_chunks(doc)


def make_chunks(docs: list[Document], variant: str) -> list[ChunkRow]:
    if variant == "basic":
        return [row for doc in docs for row in basic_chunks(doc)]
    with ThreadPoolExecutor(max_workers=max(1, CHUNK_WORKERS)) as pool:
        return [row for rows in pool.map(llm_chunks, docs) for row in rows]


def store(owner_id: str, variant: str, rows: list[ChunkRow]) -> int:
    vectors = llm.embed([r.content for r in rows], task="RETRIEVAL_DOCUMENT") if rows else []
    with connect() as conn, conn.cursor() as cur:
        # Replace this owner's chunks in one transaction, so questions never see a half-built index.
        cur.execute("DELETE FROM rag_chunks WHERE user_id = %s AND variant = %s", (owner_id, variant))
        cur.executemany(
            """
            INSERT INTO rag_chunks (id, user_id, variant, doc_type, source, title, month, headline, summary, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            """,
            [
                (str(uuid.uuid4()), owner_id, variant, r.doc.doc_type, r.doc.source, r.doc.title,
                 r.doc.month, r.headline, r.summary, r.content, to_vector(v))
                for r, v in zip(rows, vectors)
            ],
        )
    return len(rows)


def reindex_user(user_id: str, variant: str = "llm", today: date | None = None) -> dict:
    with connect() as conn:
        data = fetch_user_data(conn, user_id)
    if data is None:
        raise LookupError(f"user {user_id} not found")
    docs = build_documents(data, today)
    rows = make_chunks(docs, variant)
    count = store(user_id, variant, rows)
    log.info("reindexed user %s (%s): %d documents, %d chunks", user_id, variant, len(docs), count)
    return {"documents": len(docs), "chunks": count}


def user_indexed(user_id: str, variant: str = "llm") -> bool:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM rag_chunks WHERE user_id = %s AND variant = %s LIMIT 1", (user_id, variant))
        return cur.fetchone() is not None


def shared_exists(variant: str = "llm") -> bool:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM rag_chunks WHERE user_id = %s AND variant = %s LIMIT 1", (SHARED_USER, variant))
        return cur.fetchone() is not None


def reindex_shared(variant: str = "llm") -> dict:
    docs = guide_documents(GUIDES_DIR)
    rows = make_chunks(docs, variant)
    count = store(SHARED_USER, variant, rows)
    return {"documents": len(docs), "chunks": count}


def purge_orphans() -> int:
    """Delete chunks of users who no longer exist in PathWise."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM rag_chunks c WHERE c.user_id <> %s AND NOT EXISTS (SELECT 1 FROM users u WHERE u.id = c.user_id)",
            (SHARED_USER,),
        )
        return cur.rowcount
