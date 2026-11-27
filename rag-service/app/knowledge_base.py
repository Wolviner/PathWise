"""Turns a user's PathWise rows into markdown "documents" for the knowledge base.

Embeddings can't add numbers, so totals, category breakdowns and month-over-month
changes are computed here, when the documents are written. A question like
"how much did I spend on food in October?" then retrieves a chunk that already
contains the answer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime

MAX_MONTHS = 12  # only the most recent months go into the knowledge base


@dataclass
class Document:
    doc_type: str  # statement | monthly-summary | budget | recurring | accounts | guide
    source: str  # stable id, e.g. "summary:2026-10"
    title: str  # human-readable, shown as a source chip in PathWise
    text: str
    month: str | None = None  # "YYYY-MM"


# ---------- formatting helpers ----------

def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def month_key(d: date | datetime) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_name(key: str) -> str:
    year, month = key.split("-")
    return date(int(year), int(month), 1).strftime("%B %Y")


def short_date(d: date | datetime) -> str:
    return d.strftime("%d %b %Y")


def nice_category(category: str) -> str:
    return category.replace("-", " ").capitalize()


def pct_change(current: float, previous: float | None) -> str:
    if previous is None:
        return ""
    if previous == 0:
        return "new this month" if current else "no change"
    change = (current - previous) / previous * 100
    return f"{change:+.0f}%"


# ---------- loading from Postgres ----------

def fetch_user_data(conn, user_id: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute('SELECT id, name FROM users WHERE id = %s', (user_id,))
        row = cur.fetchone()
        if not row:
            return None
        user = {"id": row[0], "name": row[1]}

        cur.execute(
            'SELECT id, name, type::text, balance, "isDefault" FROM accounts WHERE "userId" = %s ORDER BY name',
            (user_id,),
        )
        accounts = [
            {"id": r[0], "name": r[1], "type": r[2], "balance": float(r[3]), "is_default": r[4]}
            for r in cur.fetchall()
        ]

        cur.execute(
            """
            SELECT id, type::text, amount, description, date, category, "isRecurring",
                   "recurringInterval"::text, "nextRecurringDate", "lastProcessed",
                   status::text, "accountId"
            FROM transactions
            WHERE "userId" = %s AND status <> 'FAILED'
            ORDER BY date
            """,
            (user_id,),
        )
        transactions = [
            {
                "id": r[0], "type": r[1], "amount": float(r[2]), "description": r[3] or "",
                "date": r[4], "category": r[5], "is_recurring": r[6], "recurring_interval": r[7],
                "next_recurring_date": r[8], "last_processed": r[9], "status": r[10], "account_id": r[11],
            }
            for r in cur.fetchall()
        ]

        cur.execute('SELECT amount, "lastAlertSent" FROM budgets WHERE "userId" = %s', (user_id,))
        b = cur.fetchone()
        budget = {"amount": float(b[0]), "last_alert_sent": b[1]} if b else None

    return {"user": user, "accounts": accounts, "transactions": transactions, "budget": budget}


# ---------- statistics ----------

def monthly_stats(data: dict) -> dict[str, dict]:
    """Per month: income, expenses, per-category totals and counts, per-account totals."""
    default_account = next((a for a in data["accounts"] if a["is_default"]), None)
    stats: dict[str, dict] = {}
    for t in data["transactions"]:
        key = month_key(t["date"])
        s = stats.setdefault(key, {
            "income": 0.0, "expense": 0.0, "count": 0,
            "expense_by_category": defaultdict(lambda: [0.0, 0]),
            "income_by_category": defaultdict(lambda: [0.0, 0]),
            "by_account": defaultdict(lambda: [0.0, 0.0]),
            "default_account_expense": 0.0, "expenses": [],
        })
        s["count"] += 1
        if t["type"] == "EXPENSE":
            s["expense"] += t["amount"]
            s["expense_by_category"][t["category"]][0] += t["amount"]
            s["expense_by_category"][t["category"]][1] += 1
            s["by_account"][t["account_id"]][1] += t["amount"]
            s["expenses"].append(t)
            if default_account and t["account_id"] == default_account["id"]:
                s["default_account_expense"] += t["amount"]
        else:
            s["income"] += t["amount"]
            s["income_by_category"][t["category"]][0] += t["amount"]
            s["income_by_category"][t["category"]][1] += 1
            s["by_account"][t["account_id"]][0] += t["amount"]
    return stats


def recent_months(stats: dict) -> list[str]:
    return sorted(stats)[-MAX_MONTHS:]


def previous_month(key: str) -> str:
    year, month = map(int, key.split("-"))
    return f"{year - 1:04d}-12" if month == 1 else f"{year:04d}-{month - 1:02d}"


# ---------- documents ----------

def build_documents(data: dict, today: date | None = None) -> list[Document]:
    today = today or date.today()
    accounts = {a["id"]: a for a in data["accounts"]}
    default_account = next((a for a in data["accounts"] if a["is_default"]), None)
    budget = data["budget"]
    stats = monthly_stats(data)
    months = recent_months(stats)
    docs: list[Document] = []

    # 1. Monthly summaries (totals already calculated)
    for key in months:
        s = stats[key]
        prev_key = previous_month(key)
        prev = stats.get(prev_key)
        name = month_name(key)
        net = s["income"] - s["expense"]
        rate = f"{net / s['income'] * 100:.0f}%" if s["income"] else "n/a"
        lines = [
            f"# {name} summary",
            f"Total income {money(s['income'])} · Total expenses {money(s['expense'])} · "
            f"Net savings {money(net)} · Savings rate {rate} · {s['count']} transactions",
        ]
        if prev:
            lines.append(
                f"Compared with {month_name(prev_key)}: expenses {pct_change(s['expense'], prev['expense'])}, "
                f"income {pct_change(s['income'], prev['income'])}."
            )
        if budget and default_account:
            spent = s["default_account_expense"]
            used = spent / budget["amount"] * 100 if budget["amount"] else 0
            status = (
                f"exceeded by {money(spent - budget['amount'])}"
                if spent > budget["amount"]
                else f"{money(budget['amount'] - spent)} left"
            )
            lines.append(
                f"Budget {money(budget['amount'])} per month (tracked on the default account "
                f"\"{default_account['name']}\"): spent {money(spent)} on that account in {name}, "
                f"{used:.0f}% of the budget, {status}."
            )

        lines.append(f"\n## Spending by category in {name}")
        for cat, (amount, count) in sorted(s["expense_by_category"].items(), key=lambda kv: -kv[1][0]):
            prev_amount = prev["expense_by_category"][cat][0] if prev and cat in prev["expense_by_category"] else (0.0 if prev else None)
            change = pct_change(amount, prev_amount)
            change_text = f" ({change} vs {month_name(prev_key)})" if change else ""
            lines.append(f"- {nice_category(cat)}: {money(amount)}{change_text}, {count} transactions")

        if s["income_by_category"]:
            lines.append(f"\n## Income by source in {name}")
            for cat, (amount, count) in sorted(s["income_by_category"].items(), key=lambda kv: -kv[1][0]):
                lines.append(f"- {nice_category(cat)}: {money(amount)}, {count} transactions")

        lines.append(f"\n## Largest expenses in {name}")
        for i, t in enumerate(sorted(s["expenses"], key=lambda t: -t["amount"])[:5], start=1):
            acct = accounts.get(t["account_id"], {}).get("name", "Unknown account")
            lines.append(f"{i}. {money(t['amount'])} — {t['description']} — {short_date(t['date'])} — {nice_category(t['category'])} — {acct}")

        if len(s["by_account"]) > 1:
            lines.append(f"\n## By account in {name}")
            for acct_id, (inc, exp) in s["by_account"].items():
                acct = accounts.get(acct_id, {}).get("name", "Unknown account")
                lines.append(f"- {acct}: income {money(inc)}, expenses {money(exp)}")

        docs.append(Document("monthly-summary", f"summary:{key}", f"{name} summary", "\n".join(lines), key))

    # 2. Statements: one per account per month, every transaction as a row
    by_account_month: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in data["transactions"]:
        key = month_key(t["date"])
        if key in months:
            by_account_month[(t["account_id"], key)].append(t)
    for (acct_id, key), rows in sorted(by_account_month.items(), key=lambda kv: kv[0][1]):
        acct = accounts.get(acct_id, {"name": "Unknown account", "type": ""})
        money_in = sum(t["amount"] for t in rows if t["type"] == "INCOME")
        money_out = sum(t["amount"] for t in rows if t["type"] == "EXPENSE")
        name = month_name(key)
        lines = [
            f"# {acct['name']} ({acct['type'].lower()} account) statement for {name}",
            f"Money in {money(money_in)} · Money out {money(money_out)} · Net {money(money_in - money_out)} · {len(rows)} transactions",
            "",
            "| Date | Description | Category | Type | Amount |",
            "|---|---|---|---|---|",
        ]
        for t in rows:
            signed = t["amount"] if t["type"] == "INCOME" else -t["amount"]
            lines.append(
                f"| {short_date(t['date'])} | {t['description']} | {nice_category(t['category'])} | "
                f"{t['type'].capitalize()} | {money(signed)} |"
            )
        docs.append(Document(
            "statement", f"statement:{acct_id}:{key}", f"{acct['name']} statement · {name}", "\n".join(lines), key
        ))

    # 3. Accounts overview
    if data["accounts"]:
        lines = ["# Accounts overview", f"Balances as of {short_date(today)}:"]
        for a in data["accounts"]:
            default = " (default account)" if a["is_default"] else ""
            lines.append(f"- {a['name']}{default}: {a['type'].lower()} account, balance {money(a['balance'])}")
        lines.append(f"Total balance across all accounts: {money(sum(a['balance'] for a in data['accounts']))}")
        docs.append(Document("accounts", "accounts", "Accounts overview", "\n".join(lines)))

    # 4. Budget
    if budget:
        lines = ["# Monthly budget", f"Monthly budget: {money(budget['amount'])}."]
        if default_account:
            lines.append(
                f"The budget is tracked against expenses on the default account \"{default_account['name']}\". "
                "PathWise sends an email alert when 80% of the budget is used."
            )
            this_month = month_key(today)
            spent = stats.get(this_month, {}).get("default_account_expense", 0.0)
            lines.append(
                f"This month ({month_name(this_month)}) so far: {money(spent)} spent, "
                f"{(spent / budget['amount'] * 100 if budget['amount'] else 0):.0f}% of the budget."
            )
            over = [k for k in months if stats[k]["default_account_expense"] > budget["amount"]]
            lines.append(
                "Months over budget: " + (", ".join(month_name(k) for k in over) if over else "none") + "."
            )
        if budget["last_alert_sent"]:
            lines.append(f"Last budget alert sent on {short_date(budget['last_alert_sent'])}.")
        docs.append(Document("budget", "budget", "Budget", "\n".join(lines)))

    # 5. Recurring transactions
    recurring = [t for t in data["transactions"] if t["is_recurring"]]
    if recurring:
        lines = ["# Recurring transactions", "These repeat automatically in PathWise:"]
        for t in recurring:
            acct = accounts.get(t["account_id"], {}).get("name", "Unknown account")
            nxt = f", next on {short_date(t['next_recurring_date'])}" if t["next_recurring_date"] else ""
            last = f", last processed {short_date(t['last_processed'])}" if t["last_processed"] else ""
            interval = (t["recurring_interval"] or "").lower()
            lines.append(
                f"- {t['description']}: {money(t['amount'])} {t['type'].lower()}, {interval}, "
                f"{nice_category(t['category'])}, {acct}, started {short_date(t['date'])}{nxt}{last}"
            )
        docs.append(Document("recurring", "recurring", "Recurring transactions", "\n".join(lines)))

    return docs


def guide_documents(guides_dir) -> list[Document]:
    docs = []
    for path in sorted(guides_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        docs.append(Document("guide", f"guide:{path.stem}", title, text))
    return docs
