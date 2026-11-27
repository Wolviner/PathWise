"""PathWise RAG service (FastAPI). Deployed as a Docker Space on Hugging Face.

Every endpoint except the health check requires the X-RAG-Secret header. Only the
PathWise server calls this service, and it always sends the user_id it got from Clerk.
"""

from __future__ import annotations

import hmac
import logging
import re
import threading
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import ingest, llm
from .answer import answer_question
from .config import RAG_SHARED_SECRET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag.api")

app = FastAPI(title="PathWise RAG", version="1.0.0")

USER_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def require_secret(x_rag_secret: str = Header(default="")) -> None:
    if not RAG_SHARED_SECRET or not hmac.compare_digest(x_rag_secret.encode(), RAG_SHARED_SECRET.encode()):
        raise HTTPException(status_code=401, detail="unauthorized")


def check_user_id(user_id: str) -> str:
    if not USER_ID.match(user_id) or user_id == "shared":
        raise HTTPException(status_code=400, detail="invalid user_id")
    return user_id


# ---------- health ----------

@app.get("/")
def health():
    return {"status": "ok", "service": "pathwise-rag"}


# ---------- ask ----------

class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    user_id: str
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


INDEXING_REPLY = (
    "I'm reading your PathWise records for the first time. This takes a couple of minutes. "
    "Please ask again shortly."
)


@app.post("/ask", dependencies=[Depends(require_secret)])
def ask(req: AskRequest, background: BackgroundTasks):
    check_user_id(req.user_id)
    # First question from a user who has never been indexed: start indexing instead of
    # answering "I don't have that data" from the shared guides alone.
    with _lock:
        indexed_empty = _jobs.get(req.user_id, {}).get("state") == "done"  # indexed, but no records
    if not indexed_empty and not ingest.user_indexed(req.user_id):
        _queue_reindex(req.user_id, background, rerun_if_running=False)
        return {"answer": INDEXING_REPLY, "sources": [], "rewritten_query": None, "indexing": True}
    try:
        result = answer_question(req.question.strip(), [t.model_dump() for t in req.history], req.user_id)
    except llm.QuotaExceeded:
        raise HTTPException(status_code=429, detail="The AI quota is used up for now. Please try again in a minute.")
    return {"answer": result["answer"], "sources": result["sources"], "rewritten_query": result["rewritten_query"]}


# ---------- reindex (runs in the background; PathWise doesn't wait) ----------

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _run_reindex(user_id: str) -> None:
    while True:
        with _lock:
            _jobs[user_id].update(state="running", rerun=False, started_at=datetime.now(timezone.utc).isoformat())
        try:
            if not ingest.shared_exists():
                ingest.reindex_shared()
            result = ingest.reindex_user(user_id)
            update = {"state": "done", "error": None, **result}
        except Exception as exc:  # recorded in the status endpoint
            log.exception("reindex failed for %s", user_id)
            update = {"state": "failed", "error": str(exc)}
        with _lock:
            job = _jobs[user_id]
            job.update(update, finished_at=datetime.now(timezone.utc).isoformat())
            if not job.get("rerun"):
                return
            # data changed while we were indexing: run once more


def _queue_reindex(user_id: str, background: BackgroundTasks, rerun_if_running: bool = True) -> str:
    with _lock:
        job = _jobs.get(user_id)
        if job and job["state"] in ("queued", "running"):
            if rerun_if_running:
                job["rerun"] = True
            return "already running"
        _jobs[user_id] = {"state": "queued", "rerun": False}
    background.add_task(_run_reindex, user_id)
    return "queued"


@app.post("/reindex/{user_id}", status_code=202, dependencies=[Depends(require_secret)])
def reindex(user_id: str, background: BackgroundTasks):
    check_user_id(user_id)
    status = _queue_reindex(user_id, background)
    return {"status": "queued" if status == "queued" else "already running; will run again when finished"}


@app.get("/reindex/{user_id}", dependencies=[Depends(require_secret)])
def reindex_status(user_id: str):
    check_user_id(user_id)
    with _lock:
        return _jobs.get(user_id, {"state": "unknown"})


@app.post("/reindex-shared", dependencies=[Depends(require_secret)])
def reindex_shared(only_if_missing: bool = False):
    if only_if_missing and ingest.shared_exists():
        return {"status": "skipped"}
    return {"status": "done", **ingest.reindex_shared()}


@app.post("/maintenance/purge-orphans", dependencies=[Depends(require_secret)])
def purge_orphans():
    return {"deleted": ingest.purge_orphans()}
