"""Generate evaluation questions for one PathWise user from their real data.

Week 5 used a hand-written tests.jsonl. Here the expected keywords and reference answers
are computed from the database, so the test set always matches the user's records
(use a seeded demo user). Output: evaluation/tests.<user_id>.jsonl (git-ignored).

    python -m evaluation.make_tests --user-id <PathWise users.id>
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from app.db import connect
from app.knowledge_base import (
    fetch_user_data, money, month_key, month_name, monthly_stats, nice_category, previous_month,
)

HERE = Path(__file__).resolve().parent


def build_tests(data: dict, today: date | None = None) -> list[dict]:
    today = today or date.today()
    stats = monthly_stats(data)
    # complete months only (the current month is still changing)
    months = [m for m in sorted(stats) if m < month_key(today)][-3:]
    tests: list[dict] = []

    def add(question, keywords, reference, category):
        tests.append({"question": question, "keywords": keywords, "reference_answer": reference, "category": category})

    for key in months:
        s, name = stats[key], month_name(key)
        add(f"How much did I spend in total in {name}?", [money(s["expense"])],
            f"Your total expenses in {name} were {money(s['expense'])}.", "totals")
        add(f"What was my total income in {name}?", [money(s["income"])],
            f"Your total income in {name} was {money(s['income'])}.", "totals")

        cats = sorted(s["expense_by_category"].items(), key=lambda kv: -kv[1][0])
        if cats:
            top, (amount, count) = cats[0]
            add(f"Which category did I spend the most on in {name}?", [nice_category(top), money(amount)],
                f"{nice_category(top)} was your biggest category in {name}: {money(amount)} across {count} transactions.",
                "category")
        for cat, (amount, _) in cats[1:3]:
            add(f"How much did I spend on {nice_category(cat).lower()} in {name}?", [money(amount)],
                f"You spent {money(amount)} on {nice_category(cat).lower()} in {name}.", "category")

        if s["expenses"]:
            big = max(s["expenses"], key=lambda t: t["amount"])
            add(f"What was my largest expense in {name}?", [money(big["amount"])],
                f"Your largest expense in {name} was {money(big['amount'])} ({big['description']}, "
                f"{big['date'].strftime('%d %b %Y')}).", "lookup")

        prev = stats.get(previous_month(key))
        shared = [c for c, _ in cats if prev and c in prev["expense_by_category"]]
        if shared:
            cat = shared[0]  # biggest category that also appears in the previous month
            now_amt = s["expense_by_category"][cat][0]
            prev_amt = prev["expense_by_category"][cat][0]
            more = "more" if now_amt > prev_amt else "less"
            add(f"Did I spend more on {nice_category(cat).lower()} in {name} than in {month_name(previous_month(key))}?",
                [money(now_amt), money(prev_amt)],
                f"You spent {money(now_amt)} on {nice_category(cat).lower()} in {name}, {more} than the "
                f"{money(prev_amt)} in {month_name(previous_month(key))}.", "comparison")

    if data["budget"]:
        add("What is my monthly budget?", [money(data["budget"]["amount"])],
            f"Your monthly budget is {money(data['budget']['amount'])}.", "budget")

    for a in data["accounts"][:2]:
        add(f"What is the balance of my {a['name']} account?", [money(a["balance"])],
            f"Your {a['name']} account balance is {money(a['balance'])}.", "accounts")

    recurring = [t for t in data["transactions"] if t["is_recurring"]][:3]
    if recurring:
        add("What recurring transactions do I have?", [t["description"] for t in recurring],
            "Your recurring transactions are: " + "; ".join(
                f"{t['description']} ({money(t['amount'])}, {(t['recurring_interval'] or '').lower()})" for t in recurring
            ) + ".", "recurring")

    add("What is the 50/30/20 budgeting rule?", ["50%", "30%", "20%"],
        "Split take-home income into 50% for needs, 30% for wants and 20% for savings and debt.", "guide")
    add("How do budget alerts work in PathWise?", ["80%"],
        "PathWise emails a budget alert when 80% of the monthly budget is used on the default account, at most once a month.",
        "guide")
    # Must not be answerable: checks the model says it doesn't have the data instead of guessing.
    add("How much did I spend at Zorblax Electronics?", [],
        "I don't have that data: there are no transactions from Zorblax Electronics in your records.", "unanswerable")
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    args = parser.parse_args()
    with connect() as conn:
        data = fetch_user_data(conn, args.user_id)
    if not data:
        raise SystemExit(f"User {args.user_id} not found")
    tests = build_tests(data)
    out = HERE / f"tests.{args.user_id}.jsonl"
    out.write_text("\n".join(json.dumps(t) for t in tests) + "\n", encoding="utf-8")
    print(f"Wrote {len(tests)} tests to {out}")


if __name__ == "__main__":
    main()
