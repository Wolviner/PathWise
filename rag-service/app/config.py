"""Settings for the PathWise RAG service. Everything comes from environment variables
(a local .env file in development, Space secrets/variables on Hugging Face)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


def _secret(name: str) -> str:
    # Values copied from a Next.js .env are often quoted; docker --env-file keeps the quotes.
    return os.environ.get(name, "").strip().strip('"').strip("'")


# Same Supabase Postgres database PathWise uses (the pooler URL is fine).
DATABASE_URL = _secret("DATABASE_URL")

# Google AI Studio key (free tier works).
GEMINI_API_KEY = _secret("GEMINI_API_KEY")

# Shared secret: PathWise sends it in the X-RAG-Secret header on every request.
RAG_SHARED_SECRET = _secret("RAG_SHARED_SECRET")

# Models. Free-tier quotas are per model and per day. gemini-3-flash-preview allows only
# 20 requests/day on the free tier (and PathWise's receipt scanner already uses it), so the
# Copilot defaults to the lite model for everything. gemini-flash-lite-latest pointed to
# gemini-3.5-flash-lite when this was written (~1s per call). Override per environment.
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-flash-lite-latest")  # final answer
FAST_MODEL = os.environ.get("FAST_MODEL", "gemini-flash-lite-latest")  # rewrite + rerank
CHUNK_MODEL = os.environ.get("CHUNK_MODEL", FAST_MODEL)  # ingestion
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", LLM_MODEL)  # evaluation only
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768  # must match vector(768) in the rag_chunks table

# Gemini 3 thinking level. Measured on a statement chunking call: "minimal" ~4s,
# "low" ~65s, default (high) 90-200s. "default" leaves the model's own setting.
THINKING_LEVEL = os.environ.get("THINKING_LEVEL", "minimal")


def _parse_rpm(value: str) -> dict[str, int]:
    limits = {}
    for item in value.split(","):
        if "=" in item:
            model, rpm = item.split("=", 1)
            limits[model.strip()] = int(rpm)
    return limits


# Requests per minute we allow ourselves per model (stay under the free-tier limits;
# raise them if you enable billing). 0 = no limit. Embeddings are not limited by default.
RPM_LIMITS = _parse_rpm(os.environ.get(
    "RPM_LIMITS",
    "gemini-3-flash-preview=5,gemini-flash-lite-latest=10,gemini-2.5-flash-lite=10,gemini-embedding-001=0",
))
DEFAULT_RPM = int(os.environ.get("DEFAULT_RPM", "5"))

# Seconds a chat request may spend waiting on rate limits per LLM call (PathWise times out at ~55s).
INTERACTIVE_BUDGET = float(os.environ.get("INTERACTIVE_BUDGET", "15"))

RETRIEVAL_K = 20  # chunks fetched per query (original and rewritten)
FINAL_K = 10  # chunks kept after reranking and sent to the answer prompt
CHUNK_WORKERS = int(os.environ.get("CHUNK_WORKERS", "2"))

SHARED_USER = "shared"  # user_id for the shared guides
GUIDES_DIR = Path(__file__).resolve().parent.parent / "knowledge-base" / "guides"
