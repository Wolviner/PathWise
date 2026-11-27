"""Thin wrappers around the Gemini API (google-genai), built for the free tier:

- a per-model requests-per-minute limiter (free tier: e.g. 5 RPM for gemini-3-flash),
- retries on 429/5xx that wait as long as Google's RetryInfo says,
- a time budget per call, so interactive requests fail fast instead of hanging.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception

from .config import (
    DEFAULT_RPM, EMBED_DIM, EMBED_MODEL, GEMINI_API_KEY, INTERACTIVE_BUDGET, LLM_MODEL, RPM_LIMITS,
    THINKING_LEVEL,
)

T = TypeVar("T", bound=BaseModel)
EMBED_BATCH = 50
BATCH_BUDGET = 600  # seconds a background (ingestion/eval) call may wait in total


class QuotaExceeded(RuntimeError):
    """Gemini's free-tier limits were hit and the time budget ran out."""


# ---------- client ----------

_client: genai.Client | None = None
_client_lock = threading.Lock()


def client() -> genai.Client:
    """One shared client. Created under a lock: a second, unreferenced Client would be
    garbage-collected mid-request and close the shared HTTP connection."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not GEMINI_API_KEY:
                    raise RuntimeError("GEMINI_API_KEY is not set")
                _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ---------- rate limiting ----------

class RateLimiter:
    """Sliding one-minute window shared by all threads."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self, deadline: float) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= 60:
                    self.calls.popleft()
                if len(self.calls) < self.rpm:
                    self.calls.append(now)
                    return
                wait = 60 - (now - self.calls[0]) + 0.1
            if time.monotonic() + wait > deadline:
                raise QuotaExceeded("Gemini free-tier rate limit: too many requests this minute")
            time.sleep(wait)


_limiters: dict[str, RateLimiter] = {}
_limiters_lock = threading.Lock()

# Share of each model's RPM kept free for chat: background work (ingestion, evaluation)
# may only use the rest, so a reindex never starves the Copilot. The eval CLI sets it to 0.
RESERVE_SHARE = 0.4


def _limiters_for(model: str, background: bool) -> list[RateLimiter]:
    rpm = RPM_LIMITS.get(model, DEFAULT_RPM)
    if rpm <= 0:
        return []
    with _limiters_lock:
        limiters = [_limiters.setdefault(model, RateLimiter(rpm))]
        if background and RESERVE_SHARE > 0:
            batch_rpm = max(1, rpm - math.ceil(rpm * RESERVE_SHARE))
            limiters.insert(0, _limiters.setdefault(f"{model}#background", RateLimiter(batch_rpm)))
    return limiters


# ---------- retries ----------

def _retry_delay(exc: BaseException) -> float | None:
    """Seconds from Google's RetryInfo ("retryDelay": "6s"), if present."""
    try:
        for detail in exc.details["error"]["details"]:
            if detail.get("@type", "").endswith("RetryInfo"):
                return float(str(detail["retryDelay"]).rstrip("s"))
    except Exception:
        pass
    return None


def _desired_wait(state) -> float:
    exc = state.outcome.exception()
    delay = _retry_delay(exc) if exc else None
    return delay + 1 if delay is not None else min(60.0, 4.0 * 2 ** (state.attempt_number - 1))


def is_daily_quota(exc: BaseException) -> bool:
    """True for 'requests per day' 429s: retrying within minutes can't help."""
    try:
        return any(
            "PerDay" in v.get("quotaId", "")
            for d in exc.details["error"]["details"]
            if "QuotaFailure" in d.get("@type", "")
            for v in d.get("violations", [])
        )
    except Exception:
        return False


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and exc.code in (429, 500, 503) and not is_daily_quota(exc)


def _call(model: str, fn, budget: float, **kwargs):
    deadline = time.monotonic() + budget
    # Long budgets mean background work (ingestion, evaluation); chat calls use INTERACTIVE_BUDGET.
    limiters = _limiters_for(model, background=budget > INTERACTIVE_BUDGET)

    def attempt():
        for limiter in limiters:
            limiter.acquire(deadline)
        return fn(model=model, **kwargs)

    def out_of_time(state) -> bool:
        # Give up instead of sleeping past the budget (Google may ask for a very long wait).
        return state.attempt_number >= 8 or time.monotonic() + _desired_wait(state) > deadline

    try:
        return retry(
            retry=retry_if_exception(_retryable),
            wait=_desired_wait,
            stop=out_of_time,
            reraise=True,
        )(attempt)()
    except errors.APIError as exc:
        if exc.code == 429:
            kind = "daily quota" if is_daily_quota(exc) else "rate limit"
            raise QuotaExceeded(f"Gemini {kind} reached for {model}") from exc
        raise


# ---------- public helpers ----------

def _thinking(model: str) -> types.ThinkingConfig | None:
    """Chunking, rewriting, reranking and short answers don't need deep reasoning.
    Measured on gemini-3-flash-preview: minimal ~4s per call, default 90-200s."""
    if THINKING_LEVEL == "default":
        return None
    if "gemini-3" in model:
        return types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    if "gemini-2.5-flash" in model:
        return types.ThinkingConfig(thinking_budget=0)
    return None


def _config(model: str, **kwargs) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        thinking_config=_thinking(model),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        **kwargs,
    )


def to_contents(history: list[dict], question: str) -> list[types.Content]:
    """PathWise chat history ({role: user|assistant, content}) -> Gemini contents."""
    contents = []
    for turn in history:
        role = "model" if turn.get("role") == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn.get("content", ""))]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    return contents


def generate_text(contents, system: str | None = None, model: str = LLM_MODEL, budget: float = BATCH_BUDGET) -> str:
    config = _config(model, system_instruction=system, temperature=0.2)
    response = _call(model, client().models.generate_content, budget, contents=contents, config=config)
    return (response.text or "").strip()


def generate_json(prompt: str, schema: type[T], system: str | None = None, model: str = LLM_MODEL,
                  budget: float = BATCH_BUDGET) -> T:
    config = _config(model, system_instruction=system, temperature=0,
                     response_mime_type="application/json", response_schema=schema)
    response = _call(model, client().models.generate_content, budget, contents=prompt, config=config)
    if isinstance(response.parsed, schema):
        return response.parsed
    return schema.model_validate_json(response.text or "{}")


def embed(texts: list[str], task: str, budget: float = BATCH_BUDGET) -> list[list[float]]:
    """task: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        config = types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM)
        response = _call(EMBED_MODEL, client().models.embed_content, budget, contents=batch, config=config)
        vectors.extend(e.values for e in response.embeddings)
    return vectors
