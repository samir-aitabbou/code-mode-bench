"""Deterministic generator for the synthetic mini-CRM SQLite database.

Running this module twice with the same seed produces byte-for-byte
identical data (ids, amounts, dates, texts). It also enforces the
"non-ambiguous answer" requirements from PLAN.md §3: no tie in the T3 and
T5 top-N rankings, and at least 5 customers qualify for T4.
"""

from __future__ import annotations

import argparse
import logging
import random
import sqlite3
from pathlib import Path

from faker import Faker

from cmbench.data.constants import (
    INVOICE_STATUSES,
    PRODUCT_CATEGORIES,
    REGIONS,
    SEGMENTS,
    TICKET_CATEGORIES,
    TICKET_STATUSES,
)

logger = logging.getLogger(__name__)

N_CUSTOMERS = 500
N_PRODUCTS = 200
N_ORDERS = 5000
N_INVOICES = 8000
N_TICKETS = 1500

# Weighted so that roughly 1 in 5 customers is VIP, and VIP customers are
# spread across regions -- "South" is not special-cased, T3 just needs the
# VIP+South slice to be non-trivial (checked by _verify_unambiguous).
SEGMENT_WEIGHTS = (0.2, 0.8)  # VIP, standard
INVOICE_STATUS_WEIGHTS = (0.70, 0.15, 0.15)  # paid, unpaid, overdue
TICKET_STATUS_WEIGHTS = (0.4, 0.6)  # open, closed

ORDER_DATE_START = "2025-01-01"
ORDER_DATE_END = "2026-11-22"
# T5 target window: at least this many Electronics orders in Oct 2026,
# so the top-5-by-amount is drawn from a comfortably large pool.
MIN_OCTOBER_ELECTRONICS_ORDERS = 40

SCHEMA = """
CREATE TABLE customers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    region TEXT NOT NULL,
    segment TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);

CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    product_id TEXT NOT NULL REFERENCES products(id),
    order_date TEXT NOT NULL,
    amount REAL NOT NULL
);

CREATE TABLE invoices (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    amount REAL NOT NULL,
    status TEXT NOT NULL,
    due_date TEXT NOT NULL
);

CREATE TABLE tickets (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL,
    text TEXT NOT NULL
);

CREATE TABLE outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    sent_at TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_product ON orders(product_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_invoices_customer ON invoices(customer_id);
CREATE INDEX idx_tickets_customer ON tickets(customer_id);
"""


def _random_date(rng: random.Random, start: str, end: str) -> str:
    from datetime import date, timedelta

    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    delta_days = (end_d - start_d).days
    return (start_d + timedelta(days=rng.randint(0, delta_days))).isoformat()


def _money(rng: random.Random, low: float, high: float) -> float:
    """Round to cents, with enough entropy that ties across thousands of
    rows are vanishingly unlikely (verified by _verify_unambiguous)."""
    return round(rng.uniform(low, high), 2)


def _gen_customers(conn: sqlite3.Connection, rng: random.Random, fake: Faker) -> None:
    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        cid = f"C{i:04d}"
        segment = rng.choices(SEGMENTS, weights=SEGMENT_WEIGHTS, k=1)[0]
        region = rng.choice(REGIONS)
        name = fake.name()
        email = fake.unique.email()
        notes = fake.sentence(nb_words=8) if rng.random() < 0.3 else ""
        rows.append((cid, name, email, region, segment, notes))
    conn.executemany(
        "INSERT INTO customers (id, name, email, region, segment, notes) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def _gen_products(conn: sqlite3.Connection, rng: random.Random, fake: Faker) -> None:
    rows = []
    for i in range(1, N_PRODUCTS + 1):
        pid = f"P{i:03d}"
        category = rng.choice(PRODUCT_CATEGORIES)
        name = f"{fake.word().capitalize()} {category[:-1] if category.endswith('s') else category}"
        price = _money(rng, 5.0, 2000.0)
        rows.append((pid, name, category, price))
    conn.executemany(
        "INSERT INTO products (id, name, category, price) VALUES (?, ?, ?, ?)",
        rows,
    )


def _gen_orders(conn: sqlite3.Connection, rng: random.Random) -> None:
    product_rows = conn.execute("SELECT id, category, price FROM products").fetchall()
    customer_ids = [r[0] for r in conn.execute("SELECT id FROM customers").fetchall()]
    electronics = [r for r in product_rows if r[1] == "Electronics"]

    rows = []
    for i in range(1, N_ORDERS + 1):
        oid = f"O{i:04d}"
        customer_id = rng.choice(customer_ids)
        # Oversample Electronics-in-October a bit so the T5 pool is comfortably large.
        if rng.random() < 0.05 and electronics:
            product_id, _category, price = rng.choice(electronics)
            order_date = _random_date(rng, "2026-10-01", "2026-10-31")
        else:
            product_id, _category, price = rng.choice(product_rows)
            order_date = _random_date(rng, ORDER_DATE_START, ORDER_DATE_END)
        quantity = rng.randint(1, 5)
        amount = round(price * quantity + rng.uniform(-0.5, 0.5), 2)
        rows.append((oid, customer_id, product_id, order_date, amount))
    conn.executemany(
        "INSERT INTO orders (id, customer_id, product_id, order_date, amount) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def _gen_invoices(conn: sqlite3.Connection, rng: random.Random) -> None:
    customer_ids = [r[0] for r in conn.execute("SELECT id FROM customers").fetchall()]
    rows = []
    for i in range(1, N_INVOICES + 1):
        iid = f"I{i:05d}"
        customer_id = rng.choice(customer_ids)
        amount = _money(rng, 20.0, 5000.0)
        status = rng.choices(INVOICE_STATUSES, weights=INVOICE_STATUS_WEIGHTS, k=1)[0]
        due_date = _random_date(rng, "2025-06-01", "2026-12-31")
        rows.append((iid, customer_id, amount, status, due_date))
    conn.executemany(
        "INSERT INTO invoices (id, customer_id, amount, status, due_date) VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def _gen_tickets(conn: sqlite3.Connection, rng: random.Random, fake: Faker) -> None:
    customer_ids = [r[0] for r in conn.execute("SELECT id FROM customers").fetchall()]
    rows = []
    for i in range(1, N_TICKETS + 1):
        tid = f"K{i:04d}"
        customer_id = rng.choice(customer_ids)
        created_at = _random_date(rng, "2025-06-01", "2026-11-22")
        category = rng.choice(TICKET_CATEGORIES)
        status = rng.choices(TICKET_STATUSES, weights=TICKET_STATUS_WEIGHTS, k=1)[0]
        n_sentences = rng.randint(2, 6)
        text = " ".join(fake.sentence(nb_words=rng.randint(6, 14)) for _ in range(n_sentences))
        rows.append((tid, customer_id, created_at, category, status, text))
    conn.executemany(
        "INSERT INTO tickets (id, customer_id, created_at, category, status, text) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def _nudge_ties(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    amount_col: str,
    ids_in_order: list[str],
) -> None:
    """Add a tiny deterministic increment to break a near-tie between
    consecutive ranked rows. ids_in_order is ranked best-first; row 0 gets
    the largest bump so strict ordering is preserved."""
    for rank, row_id in enumerate(ids_in_order):
        bump = round(0.01 * (len(ids_in_order) - rank), 2)
        conn.execute(
            f"UPDATE {table} SET {amount_col} = {amount_col} + ? WHERE {id_col} = ?",
            (bump, row_id),
        )


def _verify_unambiguous(conn: sqlite3.Connection) -> None:
    """Enforce PLAN.md §3's non-ambiguity requirements, nudging amounts by
    a cent when needed. Raises AssertionError if a requirement cannot be
    satisfied (should not happen with the default volumes/seed)."""

    # --- T3: VIP + South customers ranked by total unpaid+overdue invoices ---
    while True:
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
        assert len(rows) >= 4, (
            f"Only {len(rows)} VIP+South customers with unpaid invoices; need >= 4 "
            "for an unambiguous top 3 (3rd vs 4th boundary). Increase N_CUSTOMERS or "
            "SEGMENT_WEIGHTS."
        )
        top4 = [r[1] for r in rows[:4]]
        gaps_ok = all(abs(top4[k] - top4[k + 1]) > 0.01 for k in range(3))
        if gaps_ok:
            break
        logger.info("T3 ranking has a near-tie in the top 4, nudging amounts")
        _nudge_ties(conn, "invoices", "customer_id", "amount", [r[0] for r in rows[:4]])

    # --- T4: customers with > 2 overdue invoices ---
    rows = conn.execute(
        """
        SELECT customer_id, COUNT(*) AS n
        FROM invoices
        WHERE status = 'overdue'
        GROUP BY customer_id
        HAVING n > 2
        """
    ).fetchall()
    assert len(rows) >= 5, f"Only {len(rows)} customers qualify for T4, need >= 5"

    # --- T5: October 2026 Electronics orders ranked by amount ---
    while True:
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
        assert len(rows) >= MIN_OCTOBER_ELECTRONICS_ORDERS, (
            f"Only {len(rows)} October-2026 Electronics orders, need >= "
            f"{MIN_OCTOBER_ELECTRONICS_ORDERS}"
        )
        top6 = [r[1] for r in rows[:6]]
        gaps_ok = all(abs(top6[k] - top6[k + 1]) > 0.01 for k in range(5))
        if gaps_ok:
            break
        logger.info("T5 ranking has a near-tie in the top 6, nudging amounts")
        _nudge_ties(conn, "orders", "id", "amount", [r[0] for r in rows[:6]])

    conn.commit()


def generate_database(db_path: Path, seed: int = 42) -> None:
    """Create `db_path` from scratch with deterministic synthetic data.

    Any existing file at `db_path` is removed first, so this is safe to
    call repeatedly (e.g. once per benchmark trial, to reset state).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _gen_customers(conn, rng, fake)
        _gen_products(conn, rng, fake)
        _gen_orders(conn, rng)
        _gen_invoices(conn, rng)
        _gen_tickets(conn, rng, fake)
        conn.commit()
        _verify_unambiguous(conn)
    finally:
        conn.close()

    logger.info("Generated database at %s (seed=%d)", db_path, seed)


def reset_database(template_path: Path, working_path: Path) -> None:
    """Copy the reference template over the working DB, used before every
    benchmark trial so each trial starts from identical state."""
    import shutil

    working_path = Path(working_path)
    working_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_path, working_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("./data/crm.db"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_database(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
