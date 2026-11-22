import hashlib
import sqlite3
from pathlib import Path

from cmbench.data.seed import (
    N_CUSTOMERS,
    N_INVOICES,
    N_ORDERS,
    N_PRODUCTS,
    N_TICKETS,
    generate_database,
    reset_database,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_is_deterministic(tmp_path: Path) -> None:
    db1 = tmp_path / "a.db"
    db2 = tmp_path / "b.db"
    generate_database(db1, seed=42)
    generate_database(db2, seed=42)
    assert _sha256(db1) == _sha256(db2)


def test_different_seed_changes_data(tmp_path: Path) -> None:
    db1 = tmp_path / "a.db"
    db2 = tmp_path / "b.db"
    generate_database(db1, seed=42)
    generate_database(db2, seed=43)
    assert _sha256(db1) != _sha256(db2)


def test_row_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "crm.db"
    generate_database(db_path, seed=42)
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == N_CUSTOMERS
        assert conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == N_PRODUCTS
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == N_ORDERS
        assert conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0] == N_INVOICES
        assert conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == N_TICKETS
        assert conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
    finally:
        conn.close()


def test_task_anchor_ids_exist(tmp_path: Path) -> None:
    db_path = tmp_path / "crm.db"
    generate_database(db_path, seed=42)
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT 1 FROM customers WHERE id = 'C0042'").fetchone()
        assert conn.execute("SELECT 1 FROM customers WHERE id = 'C0107'").fetchone()
        assert conn.execute("SELECT 1 FROM orders WHERE id = 'O1337'").fetchone()
    finally:
        conn.close()


def test_t3_top3_vip_south_is_unambiguous(tmp_path: Path) -> None:
    db_path = tmp_path / "crm.db"
    generate_database(db_path, seed=42)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT c.id, SUM(i.amount) AS total
            FROM customers c
            JOIN invoices i ON i.customer_id = c.id
            WHERE c.segment = 'VIP' AND c.region = 'South'
              AND i.status IN ('unpaid', 'overdue')
            GROUP BY c.id
            ORDER BY total DESC
            """
        ).fetchall()
        assert len(rows) >= 4
        top4 = [r[1] for r in rows[:4]]
        for a, b in zip(top4, top4[1:], strict=False):
            assert abs(a - b) > 0.01
    finally:
        conn.close()


def test_t4_has_at_least_5_qualifying_customers(tmp_path: Path) -> None:
    db_path = tmp_path / "crm.db"
    generate_database(db_path, seed=42)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT customer_id, COUNT(*) AS n
            FROM invoices
            WHERE status = 'overdue'
            GROUP BY customer_id
            HAVING n > 2
            """
        ).fetchall()
        assert len(rows) >= 5
    finally:
        conn.close()


def test_t5_top5_october_electronics_is_unambiguous(tmp_path: Path) -> None:
    db_path = tmp_path / "crm.db"
    generate_database(db_path, seed=42)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT o.id, o.amount
            FROM orders o
            JOIN products p ON p.id = o.product_id
            WHERE p.category = 'Electronics'
              AND o.order_date BETWEEN '2026-10-01' AND '2026-10-31'
            ORDER BY o.amount DESC
            """
        ).fetchall()
        assert len(rows) >= 6
        top6 = [r[1] for r in rows[:6]]
        for a, b in zip(top6, top6[1:], strict=False):
            assert abs(a - b) > 0.01
    finally:
        conn.close()


def test_reset_database_copies_template(tmp_path: Path) -> None:
    template = tmp_path / "template.db"
    working = tmp_path / "working.db"
    generate_database(template, seed=42)
    reset_database(template, working)
    assert _sha256(template) == _sha256(working)
