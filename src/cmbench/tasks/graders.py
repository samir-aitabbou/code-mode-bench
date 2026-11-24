"""Expected-answer computation and grading for T1-T6.

Expected answers are always computed in SQL against the pristine
reference database -- never against anything a trial might have
mutated (PLAN.md §3.5). T4 additionally checks the trial's *working*
copy's final state (customers.notes, outbox) against that same
reference query, since a write task is graded on what it did, not just
on what it claims to have done (CLAUDE.md §3.5).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GradeResult:
    correct: bool
    failure_type: str | None  # None if correct, else "wrong_answer" or "format_error"
    expected: Any
    actual: Any | None
    detail: str = ""


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_final_answer(final_answer_text: str | None) -> tuple[Any | None, str | None]:
    """Returns (parsed_value, error); error is None on success."""
    if final_answer_text is None:
        return None, "no FINAL_ANSWER marker found in the model's reply"
    try:
        return json.loads(final_answer_text), None
    except json.JSONDecodeError as exc:
        return None, f"FINAL_ANSWER is not valid JSON: {exc}"


# --- expected-answer computation, one function per task --------------------


def _expected_t1(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM tickets WHERE customer_id = 'C0042' AND status = 'open'"
    ).fetchone()
    return int(row["n"])


def _expected_t2(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        """
        SELECT c.email AS email
        FROM orders o
        JOIN customers c ON c.id = o.customer_id
        WHERE o.id = 'O1337'
        """
    ).fetchone()
    return str(row["email"])


def _expected_t3(conn: sqlite3.Connection) -> list[list[Any]]:
    rows = conn.execute(
        """
        SELECT c.id AS customer_id, SUM(i.amount) AS total
        FROM customers c
        JOIN invoices i ON i.customer_id = c.id
        WHERE c.segment = 'VIP' AND c.region = 'South'
          AND i.status IN ('unpaid', 'overdue')
        GROUP BY c.id
        ORDER BY total DESC
        LIMIT 3
        """
    ).fetchall()
    return [[r["customer_id"], round(float(r["total"]), 2)] for r in rows]


def t4_qualifying_customers(conn: sqlite3.Connection) -> list[str]:
    """Customers with > 2 overdue invoices. Exposed (not underscored) since
    the runner may want to log this set for a T4 trial."""
    rows = conn.execute(
        """
        SELECT customer_id
        FROM invoices
        WHERE status = 'overdue'
        GROUP BY customer_id
        HAVING COUNT(*) > 2
        ORDER BY customer_id
        """
    ).fetchall()
    return [r["customer_id"] for r in rows]


def _expected_t5(conn: sqlite3.Connection) -> list[list[Any]]:
    rows = conn.execute(
        """
        SELECT o.id AS order_id, o.amount AS amount
        FROM orders o
        JOIN products p ON p.id = o.product_id
        WHERE p.category = 'Electronics'
          AND o.order_date BETWEEN '2026-10-01' AND '2026-10-31'
        ORDER BY o.amount DESC
        LIMIT 5
        """
    ).fetchall()
    return [[r["order_id"], round(float(r["amount"]), 2)] for r in rows]


def _expected_t6(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        """
        SELECT category
        FROM tickets
        WHERE customer_id = 'C0107'
        ORDER BY created_at DESC, id DESC
        LIMIT 3
        """
    ).fetchall()
    categories = [r["category"] for r in rows]
    counts: dict[str, int] = {}
    for cat in categories:
        counts[cat] = counts.get(cat, 0) + 1
    best_count = max(counts.values())
    winners = [c for c, n in counts.items() if n == best_count]
    if len(winners) == 1:
        return str(winners[0])
    # 3-way tie (every ticket has a different category): most recent wins.
    return str(categories[0])


# --- grading -----------------------------------------------------------------


def _amounts_match(expected: list[list[Any]], actual: Any, tolerance: float = 0.01) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for exp_pair, act_pair in zip(expected, actual, strict=True):
        if not isinstance(act_pair, list | tuple) or len(act_pair) != 2:
            return False
        exp_id, exp_amount = exp_pair
        act_id, act_amount = act_pair
        if str(act_id) != str(exp_id):
            return False
        try:
            if abs(float(act_amount) - float(exp_amount)) > tolerance:
                return False
        except (TypeError, ValueError):
            return False
    return True


def grade_t1(template_db: Path, final_answer_text: str | None) -> GradeResult:
    expected = _expected_t1(_connect(template_db))
    actual, error = _parse_final_answer(final_answer_text)
    if error:
        return GradeResult(False, "format_error", expected, None, error)
    try:
        correct = int(actual) == expected  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return GradeResult(False, "format_error", expected, actual, "answer is not an integer")
    return GradeResult(correct, None if correct else "wrong_answer", expected, actual)


def grade_t2(template_db: Path, final_answer_text: str | None) -> GradeResult:
    expected = _expected_t2(_connect(template_db))
    actual, error = _parse_final_answer(final_answer_text)
    if error:
        return GradeResult(False, "format_error", expected, None, error)
    correct = isinstance(actual, str) and actual.strip().lower() == expected.strip().lower()
    return GradeResult(correct, None if correct else "wrong_answer", expected, actual)


def grade_t3(template_db: Path, final_answer_text: str | None) -> GradeResult:
    expected = _expected_t3(_connect(template_db))
    actual, error = _parse_final_answer(final_answer_text)
    if error:
        return GradeResult(False, "format_error", expected, None, error)
    correct = _amounts_match(expected, actual)
    return GradeResult(correct, None if correct else "wrong_answer", expected, actual)


def grade_t4(template_db: Path, working_db: Path, final_answer_text: str | None) -> GradeResult:
    expected_customers = t4_qualifying_customers(_connect(template_db))
    actual, error = _parse_final_answer(final_answer_text)
    if error:
        return GradeResult(False, "format_error", expected_customers, None, error)

    conn = _connect(working_db)
    mismatches: list[str] = []
    for cust_id in expected_customers:
        row = conn.execute("SELECT notes FROM customers WHERE id = ?", (cust_id,)).fetchone()
        if row is None or row["notes"] != "Q4 follow-up":
            mismatches.append(f"{cust_id}: notes not set to 'Q4 follow-up'")
            continue
        sent = conn.execute(
            "SELECT COUNT(*) AS n FROM outbox WHERE customer_id = ?", (cust_id,)
        ).fetchone()
        if sent["n"] == 0:
            mismatches.append(f"{cust_id}: no email in outbox")

    correct = not mismatches
    return GradeResult(
        correct,
        None if correct else "wrong_answer",
        expected_customers,
        actual,
        "; ".join(mismatches),
    )


def grade_t5(template_db: Path, final_answer_text: str | None) -> GradeResult:
    expected = _expected_t5(_connect(template_db))
    actual, error = _parse_final_answer(final_answer_text)
    if error:
        return GradeResult(False, "format_error", expected, None, error)
    correct = _amounts_match(expected, actual)
    return GradeResult(correct, None if correct else "wrong_answer", expected, actual)


def grade_t6(template_db: Path, final_answer_text: str | None) -> GradeResult:
    expected = _expected_t6(_connect(template_db))
    actual, error = _parse_final_answer(final_answer_text)
    if error:
        return GradeResult(False, "format_error", expected, None, error)
    correct = isinstance(actual, str) and actual.strip() == expected
    return GradeResult(correct, None if correct else "wrong_answer", expected, actual)


_READ_ONLY_GRADERS = {
    "T1": grade_t1,
    "T2": grade_t2,
    "T3": grade_t3,
    "T5": grade_t5,
    "T6": grade_t6,
}


def grade_task(
    task_id: str,
    template_db: Path,
    working_db: Path,
    final_answer_text: str | None,
) -> GradeResult:
    """Grade one trial. `template_db` is the pristine reference database
    (used to compute the expected answer); `working_db` is the trial's
    working copy after it ran (only inspected for T4's write check)."""
    if task_id in ("T4", "T4p"):
        # T4p is the same task, served by a paginated list_invoices tool
        # (see mcp_server.server.build_server's paginated_invoices option)
        # -- grading is identical, only the tool behaviour differs.
        return grade_t4(template_db, working_db, final_answer_text)
    grader = _READ_ONLY_GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"unknown task id {task_id!r}")
    return grader(template_db, final_answer_text)
