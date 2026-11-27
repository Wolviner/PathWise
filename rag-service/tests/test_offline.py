"""Offline unit tests (no database, no Gemini calls).

    docker run --rm -v "${PWD}:/app" -w /app python:3.12-slim sh -c "pip install -q -r requirements.txt pytest && pytest -q"
"""

from datetime import date, datetime

from app.answer import Retrieved, apply_order, merge_chunks, sources_for
from app.ingest import basic_chunks
from app.knowledge_base import build_documents, money, monthly_stats
from evaluation.make_tests import build_tests
from evaluation.metrics import ndcg, reciprocal_rank, retrieval_scores


def tx(id, type_, amount, day, category, account="a1", desc=None, recurring=False):
    return {
        "id": id, "type": type_, "amount": amount, "description": desc or f"Paid for {category}",
        "date": day, "category": category, "is_recurring": recurring,
        "recurring_interval": "MONTHLY" if recurring else None, "next_recurring_date": None,
        "last_processed": None, "status": "COMPLETED", "account_id": account,
    }


DATA = {
    "user": {"id": "u1", "name": "Demo"},
    "accounts": [
        {"id": "a1", "name": "Everyday", "type": "CURRENT", "balance": 1200.5, "is_default": True},
        {"id": "a2", "name": "Savings", "type": "SAVINGS", "balance": 5000.0, "is_default": False},
    ],
    "transactions": [
        tx("t1", "INCOME", 5000, datetime(2026, 9, 1), "salary"),
        tx("t2", "EXPENSE", 100, datetime(2026, 9, 5), "food"),
        tx("t3", "EXPENSE", 1500, datetime(2026, 9, 3), "housing", desc="Rent", recurring=True),
        tx("t4", "INCOME", 5000, datetime(2026, 10, 1), "salary"),
        tx("t5", "EXPENSE", 142, datetime(2026, 10, 7), "food"),
        tx("t6", "EXPENSE", 1500, datetime(2026, 10, 3), "housing", desc="Rent"),
        tx("t7", "EXPENSE", 2000, datetime(2026, 10, 12), "travel", desc="Goa flights"),
        tx("t8", "EXPENSE", 300, datetime(2026, 10, 15), "shopping", account="a2"),
    ],
    "budget": {"amount": 3000.0, "last_alert_sent": None},
}


def test_money():
    assert money(1234.5) == "$1,234.50"
    assert money(-3) == "-$3.00"


def test_monthly_stats_totals():
    s = monthly_stats(DATA)["2026-10"]
    assert s["income"] == 5000
    assert s["expense"] == 142 + 1500 + 2000 + 300
    assert s["default_account_expense"] == 142 + 1500 + 2000  # shopping was on Savings
    assert s["expense_by_category"]["food"] == [142, 1]


def test_summary_document_has_precomputed_answers():
    docs = build_documents(DATA, today=date(2026, 11, 20))
    oct_summary = next(d for d in docs if d.source == "summary:2026-10")
    text = oct_summary.text
    assert "Total expenses $3,942.00" in text
    assert "Food: $142.00 (+42% vs September 2026), 1 transactions" in text
    assert "Travel: $2,000.00 (new this month vs September 2026)" in text
    assert "spent $3,642.00 on that account in October 2026, 121% of the budget, exceeded by $642.00" in text
    assert "1. $2,000.00 — Goa flights — 12 Oct 2026 — Travel — Everyday" in text


def test_statements_per_account_and_month():
    docs = build_documents(DATA, today=date(2026, 11, 20))
    sources = {d.source for d in docs}
    assert {"statement:a1:2026-10", "statement:a2:2026-10", "statement:a1:2026-09"} <= sources
    assert {"accounts", "budget", "recurring"} <= sources
    savings = next(d for d in docs if d.source == "statement:a2:2026-10")
    assert "| 15 Oct 2026 | Paid for shopping | Shopping | Expense | -$300.00 |" in savings.text


def test_basic_chunks_cover_whole_document():
    docs = build_documents(DATA, today=date(2026, 11, 20))
    doc = next(d for d in docs if d.source == "summary:2026-10")
    doc.text = doc.text + "\n" + "\n".join(f"- filler line {i} " + "x" * 80 for i in range(40))
    chunks = basic_chunks(doc)
    assert len(chunks) > 1
    assert all(c.content.startswith("Source: October 2026 summary") for c in chunks)
    assert "filler line 39" in chunks[-1].content


def r(i):
    return Retrieved(id=str(i), doc_type="t", source=f"s{i % 3}", title=f"T{i % 3}", month=None, content=f"c{i}", distance=0.1)


def test_merge_dedupes_by_id():
    merged = merge_chunks([r(1), r(2)], [r(2), r(3)])
    assert [c.id for c in merged] == ["1", "2", "3"]


def test_apply_order_handles_bad_llm_output():
    chunks = [r(1), r(2), r(3), r(4)]
    ranked = apply_order(chunks, [3, 3, 99, 1, 0])
    assert [c.id for c in ranked] == ["3", "1", "2", "4"]


def test_sources_unique_and_limited():
    assert [s["title"] for s in sources_for([r(i) for i in range(10)], limit=2)] == ["T0", "T1"]


def test_metrics():
    docs = ["nothing", "has Food $142.00", "Food again"]
    assert reciprocal_rank("food", docs) == 0.5
    assert 0 < ndcg("food", docs) < 1
    assert retrieval_scores(["food", "missing"], docs)["coverage"] == 50
    assert retrieval_scores([], docs)["mrr"] is None


def test_retry_delay_parsed_from_google_error():
    from app.llm import _retry_delay

    class Fake(Exception):
        details = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "6s"}]}}

    assert _retry_delay(Fake()) == 6.0
    assert _retry_delay(ValueError()) is None


def test_rate_limiter_gives_up_when_budget_is_too_short():
    import time

    import pytest

    from app.llm import QuotaExceeded, RateLimiter

    limiter = RateLimiter(rpm=2)
    deadline = time.monotonic() + 1
    limiter.acquire(deadline)
    limiter.acquire(deadline)
    with pytest.raises(QuotaExceeded):
        limiter.acquire(deadline)  # third call in the same minute would need a ~60s wait


def _api_error(quota_id: str, retry_delay: str):
    from google.genai import errors

    body = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id, "quotaValue": "20"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
    ]}}
    return errors.ClientError(429, body)


def test_daily_quota_fails_fast_without_retrying():
    import pytest

    from app import llm

    calls = []

    def fake(**_):
        calls.append(1)
        raise _api_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "40000s")

    with pytest.raises(llm.QuotaExceeded, match="daily quota"):
        llm._call("test-model-daily", fake, budget=600)
    assert len(calls) == 1


def test_long_retry_delay_never_exceeds_budget():
    import time

    import pytest

    from app import llm

    def fake(**_):
        raise _api_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "45s")

    start = time.monotonic()
    with pytest.raises(llm.QuotaExceeded, match="rate limit"):
        llm._call("test-model-rpm", fake, budget=5)  # Google asks for 45s; budget is 5s
    assert time.monotonic() - start < 2


def test_background_work_leaves_headroom_for_chat():
    from app import llm

    chat = llm._limiters_for("gemini-flash-lite-latest", background=False)
    background = llm._limiters_for("gemini-flash-lite-latest", background=True)
    assert [lim.rpm for lim in chat] == [10]
    assert [lim.rpm for lim in background] == [6, 10]  # 40% of the minute stays free for chat


def test_generated_tests_use_document_numbers():
    tests = build_tests(DATA, today=date(2026, 11, 20))
    total = next(t for t in tests if t["question"] == "How much did I spend in total in October 2026?")
    assert total["keywords"] == ["$3,942.00"]
    docs = build_documents(DATA, today=date(2026, 11, 20))
    all_text = "\n".join(d.text for d in docs)
    for t in tests:
        if t["category"] not in ("guide", "unanswerable"):
            for kw in t["keywords"]:
                assert kw.lower() in all_text.lower(), (t["question"], kw)
