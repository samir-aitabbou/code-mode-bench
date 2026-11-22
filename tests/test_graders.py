import json
import shutil
import sqlite3
from pathlib import Path

from cmbench.mcp_server.tools import CrmTools
from cmbench.tasks.graders import _expected_t6, grade_task, t4_qualifying_customers


def _row_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def test_t1_correct_and_wrong(seeded_db: Path) -> None:
    conn = sqlite3.connect(seeded_db)
    expected = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE customer_id='C0042' AND status='open'"
    ).fetchone()[0]

    ok = grade_task("T1", seeded_db, seeded_db, str(expected))
    assert ok.correct

    bad = grade_task("T1", seeded_db, seeded_db, str(expected + 1))
    assert not bad.correct
    assert bad.failure_type == "wrong_answer"


def test_t1_format_error(seeded_db: Path) -> None:
    result = grade_task("T1", seeded_db, seeded_db, "not json")
    assert not result.correct
    assert result.failure_type == "format_error"

    result2 = grade_task("T1", seeded_db, seeded_db, None)
    assert not result2.correct
    assert result2.failure_type == "format_error"


def test_t2_case_insensitive(seeded_db: Path) -> None:
    conn = sqlite3.connect(seeded_db)
    email = conn.execute(
        "SELECT c.email FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.id = 'O1337'"
    ).fetchone()[0]

    ok = grade_task("T2", seeded_db, seeded_db, json.dumps(email.upper()))
    assert ok.correct

    bad = grade_task("T2", seeded_db, seeded_db, json.dumps("nobody@example.com"))
    assert not bad.correct


def test_t3_correct_order_matters(seeded_db: Path) -> None:
    conn = sqlite3.connect(seeded_db)
    rows = conn.execute(
        """
        SELECT c.id, SUM(i.amount) AS total FROM customers c
        JOIN invoices i ON i.customer_id = c.id
        WHERE c.segment='VIP' AND c.region='South' AND i.status IN ('unpaid','overdue')
        GROUP BY c.id ORDER BY total DESC LIMIT 3
        """
    ).fetchall()
    expected = [[r[0], round(r[1], 2)] for r in rows]

    ok = grade_task("T3", seeded_db, seeded_db, json.dumps(expected))
    assert ok.correct

    reversed_answer = json.dumps(list(reversed(expected)))
    bad = grade_task("T3", seeded_db, seeded_db, reversed_answer)
    assert not bad.correct  # order matters: reversed is wrong unless amounts happen to tie


def test_t3_tolerates_small_rounding(seeded_db: Path) -> None:
    conn = sqlite3.connect(seeded_db)
    rows = conn.execute(
        """
        SELECT c.id, SUM(i.amount) AS total FROM customers c
        JOIN invoices i ON i.customer_id = c.id
        WHERE c.segment='VIP' AND c.region='South' AND i.status IN ('unpaid','overdue')
        GROUP BY c.id ORDER BY total DESC LIMIT 3
        """
    ).fetchall()
    expected = [[r[0], round(r[1] + 0.005, 2)] for r in rows]  # within tolerance
    ok = grade_task("T3", seeded_db, seeded_db, json.dumps(expected))
    assert ok.correct


def test_t4_correct_when_all_writes_applied(tmp_path: Path, seeded_db: Path) -> None:
    working = tmp_path / "working.db"
    shutil.copyfile(seeded_db, working)

    conn = sqlite3.connect(working)
    crm = CrmTools(conn)
    qualifying = t4_qualifying_customers(_row_conn(seeded_db))
    for cust_id in qualifying:
        crm.update_customer_note(cust_id, "Q4 follow-up")
        crm.send_email(cust_id, "Overdue invoices", "Please settle your overdue balance.")

    result = grade_task(
        "T4", seeded_db, working, json.dumps({"customers_updated": qualifying})
    )
    assert result.correct


def test_t4_incorrect_when_one_customer_missed(tmp_path: Path, seeded_db: Path) -> None:
    working = tmp_path / "working.db"
    shutil.copyfile(seeded_db, working)

    conn = sqlite3.connect(working)
    crm = CrmTools(conn)
    qualifying = t4_qualifying_customers(_row_conn(seeded_db))
    assert len(qualifying) >= 2
    for cust_id in qualifying[1:]:  # skip the first one on purpose
        crm.update_customer_note(cust_id, "Q4 follow-up")
        crm.send_email(cust_id, "Overdue invoices", "Please settle your overdue balance.")

    result = grade_task(
        "T4", seeded_db, working, json.dumps({"customers_updated": qualifying})
    )
    assert not result.correct
    assert result.failure_type == "wrong_answer"
    assert qualifying[0] in result.detail


def test_t5_correct_and_wrong(seeded_db: Path) -> None:
    conn = sqlite3.connect(seeded_db)
    rows = conn.execute(
        """
        SELECT o.id, o.amount FROM orders o JOIN products p ON p.id = o.product_id
        WHERE p.category='Electronics' AND o.order_date BETWEEN '2026-10-01' AND '2026-10-31'
        ORDER BY o.amount DESC LIMIT 5
        """
    ).fetchall()
    expected = [[r[0], round(r[1], 2)] for r in rows]

    ok = grade_task("T5", seeded_db, seeded_db, json.dumps(expected))
    assert ok.correct

    wrong = [[oid, amt + 100] for oid, amt in expected]
    bad = grade_task("T5", seeded_db, seeded_db, json.dumps(wrong))
    assert not bad.correct


def test_t6_matches_expected_helper(seeded_db: Path) -> None:
    expected = _expected_t6(_row_conn(seeded_db))
    ok = grade_task("T6", seeded_db, seeded_db, json.dumps(expected))
    assert ok.correct

    bad = grade_task("T6", seeded_db, seeded_db, json.dumps("not_a_real_category_xyz"))
    assert not bad.correct


def test_t6_tie_break_uses_most_recent(tmp_path: Path) -> None:
    db_path = tmp_path / "mini.db"
    conn = _row_conn(db_path)
    conn.executescript(
        """
        CREATE TABLE tickets (
            id TEXT PRIMARY KEY,
            customer_id TEXT,
            created_at TEXT,
            category TEXT
        );
        """
    )
    # three different categories -> 3-way tie -> most recent (K0003) wins
    conn.executemany(
        "INSERT INTO tickets (id, customer_id, created_at, category) VALUES (?, ?, ?, ?)",
        [
            ("K0001", "C0107", "2026-01-01", "billing"),
            ("K0002", "C0107", "2026-02-01", "shipping"),
            ("K0003", "C0107", "2026-03-01", "support"),
        ],
    )
    conn.commit()

    assert _expected_t6(conn) == "support"


def test_t6_majority_wins_even_if_not_most_recent(tmp_path: Path) -> None:
    db_path = tmp_path / "mini2.db"
    conn = _row_conn(db_path)
    conn.executescript(
        """
        CREATE TABLE tickets (
            id TEXT PRIMARY KEY,
            customer_id TEXT,
            created_at TEXT,
            category TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO tickets (id, customer_id, created_at, category) VALUES (?, ?, ?, ?)",
        [
            ("K0001", "C0107", "2026-01-01", "billing"),
            ("K0002", "C0107", "2026-02-01", "billing"),
            ("K0003", "C0107", "2026-03-01", "support"),  # most recent, but not majority
        ],
    )
    conn.commit()

    assert _expected_t6(conn) == "billing"
