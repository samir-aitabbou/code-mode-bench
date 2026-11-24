"""Pure Python implementations of the 12 mini-CRM tools.

Kept separate from the MCP transport (`server.py`) so the exact same
logic can be:
- exposed as MCP tools to the model in `classic` mode, and
- called through the sandbox bridge in `codemode` mode,
without the two modes ever drifting apart.

Every `list_*` tool takes only simple equality/range filters -- never a
free-form SQL or expression string -- so that `codemode`'s advantage on
T5 (a large-volume task) comes from writing a short loop/aggregation in
Python, not from being handed a backdoor query language that `classic`
mode cannot use (see CLAUDE.md §3.1).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


class NotFoundError(Exception):
    """Raised when a get_* lookup finds no matching row."""


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


@dataclass
class CrmTools:
    """Bound to one SQLite connection (the per-trial working copy)."""

    conn: sqlite3.Connection

    def __post_init__(self) -> None:
        self.conn.row_factory = sqlite3.Row

    # --- customers -----------------------------------------------------

    def list_customers(
        self,
        region: str | None = None,
        segment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List customers, optionally filtered by exact region and/or segment."""
        clauses: list[str] = []
        params: list[Any] = []
        if region is not None:
            clauses.append("region = ?")
            params.append(region)
        if segment is not None:
            clauses.append("segment = ?")
            params.append(segment)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = self.conn.execute(
            f"SELECT * FROM customers {where} ORDER BY id LIMIT ? OFFSET ?",  # noqa: S608
            params,
        ).fetchall()
        return _rows_to_dicts(rows)

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Fetch one customer by id."""
        row = self.conn.execute(
            "SELECT * FROM customers WHERE id = ?", (customer_id,)
        ).fetchone()
        result = _row_to_dict(row)
        if result is None:
            raise NotFoundError(f"customer {customer_id!r} not found")
        return result

    # --- products --------------------------------------------------------

    def list_products(
        self,
        category: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List products, optionally filtered by exact category."""
        clauses = []
        params: list[Any] = []
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = self.conn.execute(
            f"SELECT * FROM products {where} ORDER BY id LIMIT ? OFFSET ?",  # noqa: S608
            params,
        ).fetchall()
        return _rows_to_dicts(rows)

    def get_product(self, product_id: str) -> dict[str, Any]:
        """Fetch one product by id."""
        row = self.conn.execute(
            "SELECT * FROM products WHERE id = ?", (product_id,)
        ).fetchone()
        result = _row_to_dict(row)
        if result is None:
            raise NotFoundError(f"product {product_id!r} not found")
        return result

    # --- orders ------------------------------------------------------------

    def list_orders(
        self,
        customer_id: str | None = None,
        product_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 5000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List orders, optionally filtered by customer, product and/or an
        order_date range (inclusive, ISO 'YYYY-MM-DD'). No product-category
        filter: joining to products is left to the caller, by design."""
        clauses = []
        params: list[Any] = []
        if customer_id is not None:
            clauses.append("customer_id = ?")
            params.append(customer_id)
        if product_id is not None:
            clauses.append("product_id = ?")
            params.append(product_id)
        if date_from is not None:
            clauses.append("order_date >= ?")
            params.append(date_from)
        if date_to is not None:
            clauses.append("order_date <= ?")
            params.append(date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = self.conn.execute(
            f"SELECT * FROM orders {where} ORDER BY id LIMIT ? OFFSET ?",  # noqa: S608
            params,
        ).fetchall()
        return _rows_to_dicts(rows)

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch one order by id."""
        row = self.conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        result = _row_to_dict(row)
        if result is None:
            raise NotFoundError(f"order {order_id!r} not found")
        return result

    # --- invoices -------------------------------------------------------

    def list_invoices(
        self,
        customer_id: str | None = None,
        status: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List invoices, optionally filtered by customer and/or exact status
        ('paid', 'unpaid' or 'overdue')."""
        clauses = []
        params: list[Any] = []
        if customer_id is not None:
            clauses.append("customer_id = ?")
            params.append(customer_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = self.conn.execute(
            f"SELECT * FROM invoices {where} ORDER BY id LIMIT ? OFFSET ?",  # noqa: S608
            params,
        ).fetchall()
        return _rows_to_dicts(rows)

    def list_invoices_page(
        self,
        customer_id: str | None = None,
        status: str | None = None,
        limit: int = 200,
        offset: int = 0,
        max_limit: int = 200,
    ) -> dict[str, Any]:
        """Like `list_invoices`, but for a well-designed paginated tool: the
        server clamps `limit` to `max_limit` and reports `total_count` /
        `has_more` so a caller can tell it needs another page, instead of
        silently truncating (ANALYSE-CLAUDE.md point A)."""
        effective_limit = min(limit, max_limit)
        clauses = []
        params: list[Any] = []
        if customer_id is not None:
            clauses.append("customer_id = ?")
            params.append(customer_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total = self.conn.execute(
            f"SELECT COUNT(*) AS n FROM invoices {where}",  # noqa: S608
            params,
        ).fetchone()["n"]
        rows = self.conn.execute(
            f"SELECT * FROM invoices {where} ORDER BY id LIMIT ? OFFSET ?",  # noqa: S608
            [*params, effective_limit, offset],
        ).fetchall()
        items = _rows_to_dicts(rows)
        return {
            "items": items,
            "returned": len(items),
            "offset": offset,
            "limit": effective_limit,
            "total_count": int(total),
            "has_more": offset + len(items) < int(total),
        }

    def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Fetch one invoice by id."""
        row = self.conn.execute("SELECT * FROM invoices WHERE id = ?", (invoice_id,)).fetchone()
        result = _row_to_dict(row)
        if result is None:
            raise NotFoundError(f"invoice {invoice_id!r} not found")
        return result

    # --- tickets ---------------------------------------------------------

    def search_tickets(
        self,
        customer_id: str | None = None,
        category: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        """Search tickets by exact customer/category/status. Sorted by
        created_at descending by default, so `limit=3` with a customer_id
        gets that customer's 3 most recent tickets."""
        clauses = []
        params: list[Any] = []
        if customer_id is not None:
            clauses.append("customer_id = ?")
            params.append(customer_id)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "created_at DESC, id DESC" if newest_first else "created_at ASC, id ASC"
        params.extend([limit, offset])
        rows = self.conn.execute(
            f"SELECT * FROM tickets {where} ORDER BY {order} LIMIT ? OFFSET ?",  # noqa: S608
            params,
        ).fetchall()
        return _rows_to_dicts(rows)

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Fetch one ticket by id."""
        row = self.conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        result = _row_to_dict(row)
        if result is None:
            raise NotFoundError(f"ticket {ticket_id!r} not found")
        return result

    # --- writes ------------------------------------------------------------

    def update_customer_note(self, customer_id: str, note: str) -> dict[str, Any]:
        """Overwrite a customer's notes field. Returns the updated customer."""
        cur = self.conn.execute(
            "UPDATE customers SET notes = ? WHERE id = ?", (note, customer_id)
        )
        if cur.rowcount == 0:
            raise NotFoundError(f"customer {customer_id!r} not found")
        self.conn.commit()
        return self.get_customer(customer_id)

    def send_email(self, customer_id: str, subject: str, body: str) -> dict[str, Any]:
        """Simulate sending an email: appends a row to `outbox`, does not
        actually send anything over the network."""
        self.get_customer(customer_id)  # raises NotFoundError if unknown
        sent_at = datetime.now(UTC).isoformat()
        cur = self.conn.execute(
            "INSERT INTO outbox (customer_id, sent_at, subject, body) VALUES (?, ?, ?, ?)",
            (customer_id, sent_at, subject, body),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM outbox WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        result = _row_to_dict(row)
        assert result is not None
        return result
