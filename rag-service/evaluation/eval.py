"""Evaluate the Copilot for one user, in one of three modes (run all three for the portfolio):

  basic : basic text-splitter chunks, single vector search          (week 5 day 3 baseline)
  llm   : LLM chunks, single vector search                           (day 5 chunking only)
  full  : LLM chunks + query rewrite + dual retrieval + merge + rerank (production)

    python -m evaluation.make_tests --user-id <id>
    python -m evaluation.eval --user-id <id> --mode basic --reindex
    python -m evaluation.eval --user-id <id> --mode llm --reindex
    python -m evaluation.eval --user-id <id> --mode full

Retrieval metrics (MRR, nDCG, keyword coverage) as in week 5, plus LLM-as-judge scores
(accuracy, completeness, relevance, 1-5). Results go to evaluation/results/<mode>.json.
--reindex builds that mode's index first ("basic" and "llm" are stored separately).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from app import ingest, llm
from app.answer import answer_question, retrieve
from app.config import JUDGE_MODEL

from .metrics import retrieval_scores

HERE = Path(__file__).resolve().parent


class AnswerEval(BaseModel):
    feedback: str = Field(description="Concise feedback comparing the answer with the reference answer")
    accuracy: float = Field(description="1 (wrong; any wrong answer must score 1) to 5 (perfectly accurate)")
    completeness: float = Field(description="1 (missing key information) to 5 (all information from the reference answer)")
    relevance: float = Field(description="1 (off-topic) to 5 (directly answers, nothing extra)")


def judge(question: str, answer: str, reference: str) -> AnswerEval:
    prompt = f"""Question:
{question}

Generated answer:
{answer}

Reference answer:
{reference}

Score the generated answer from 1 to 5 for accuracy, completeness and relevance against the reference answer.
Amounts must match exactly to count as accurate. If the reference says the data is not available, an answer that
says so is accurate, and an answer that invents figures scores 1. Only give 5 for perfect answers."""
    return llm.generate_json(
        prompt, AnswerEval, model=JUDGE_MODEL,
        system="You are an expert evaluator of a personal-finance assistant's answers.",
    )


def avg(values):
    values = [v for v in values if v is not None]
    return round(mean(values), 4) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--mode", choices=["basic", "llm", "full"], default="full")
    parser.add_argument("--reindex", action="store_true", help="rebuild this mode's index before evaluating")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N tests")
    parser.add_argument("--skip-judge", action="store_true", help="retrieval metrics only (fewer LLM calls)")
    args = parser.parse_args()
    llm.RESERVE_SHARE = 0  # this process does no chat, so use the full rate limits

    tests_file = HERE / f"tests.{args.user_id}.jsonl"
    if not tests_file.exists():
        raise SystemExit(f"{tests_file.name} not found. Run: python -m evaluation.make_tests --user-id {args.user_id}")
    tests = [json.loads(line) for line in tests_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        tests = tests[: args.limit]

    variant = "basic" if args.mode == "basic" else "llm"
    if args.reindex:
        print(f"Building '{variant}' index…")
        if not ingest.shared_exists(variant):
            ingest.reindex_shared(variant)
        print(ingest.reindex_user(args.user_id, variant))

    rows = []
    for i, test in enumerate(tests, start=1):
        if args.skip_judge:
            chunks, _ = retrieve(test["question"], [], args.user_id, args.mode, budget=llm.BATCH_BUDGET)
            answer, verdict = None, None
        else:
            result = answer_question(test["question"], [], args.user_id, args.mode, budget=llm.BATCH_BUDGET)
            chunks, answer = result["chunks"], result["answer"]
            verdict = judge(test["question"], answer, test["reference_answer"])
        scores = retrieval_scores(test["keywords"], [c.content for c in chunks])
        rows.append({
            **test, **scores, "answer": answer,
            **({"accuracy": verdict.accuracy, "completeness": verdict.completeness,
                "relevance": verdict.relevance, "feedback": verdict.feedback} if verdict else {}),
        })
        mrr = "  -  " if scores["mrr"] is None else f"{scores['mrr']:.2f}"
        acc = f" acc {verdict.accuracy:.0f}/5" if verdict else ""
        print(f"[{i}/{len(tests)}] {test['category']:<12} MRR {mrr}{acc}  {test['question']}")

    summary = {
        "mode": args.mode,
        "tests": len(rows),
        "mrr": avg(r["mrr"] for r in rows),
        "ndcg": avg(r["ndcg"] for r in rows),
        "keyword_coverage": avg(r["coverage"] for r in rows),
        "accuracy": avg(r.get("accuracy") for r in rows),
        "completeness": avg(r.get("completeness") for r in rows),
        "relevance": avg(r.get("relevance") for r in rows),
    }
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.mode}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"{k:>17}: {v}")
    print("=" * 60)
    if summary["mrr"] is not None:
        print(f"Portfolio numbers ({args.mode}): MRR {summary['mrr'] * 100:.0f}%, nDCG {summary['ndcg'] * 100:.0f}%, "
              f"keyword coverage {summary['keyword_coverage']:.0f}%"
              + (f", LLM judge accuracy {summary['accuracy']:.1f}/5" if summary["accuracy"] else ""))


if __name__ == "__main__":
    main()
