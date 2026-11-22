"""MCP server exposing the mini-CRM as 12 tools over stdio.

`classic` mode gives the model these tool definitions directly and lets
it call them one at a time. `codemode` mode instead generates a typed
`tools.py` stub from this same tool set and runs the model's code in a
sandbox that forwards `tools.xxx()` calls to this server through the
bridge (see `cmbench.sandbox.bridge`). Either way, the actual tool
behaviour comes from `cmbench.mcp_server.tools.CrmTools`, so both modes
see identical semantics.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from cmbench.mcp_server.tools import CrmTools

logger = logging.getLogger(__name__)


def build_server(db_path: Path, *, with_noise_tools: bool = False) -> MCPServer:
    """Build the MCP server bound to the SQLite file at `db_path`.

    Call `.run()` on the result to serve it over stdio, or drive it
    directly (e.g. via an in-process MCP client session) in tests.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    crm = CrmTools(conn)
    server = MCPServer("cmbench-crm")

    @server.tool()
    def list_customers(
        region: str | None = None,
        segment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List customers, optionally filtered by exact region and/or segment
        ('VIP' or 'standard')."""
        return crm.list_customers(region=region, segment=segment, limit=limit, offset=offset)

    @server.tool()
    def get_customer(customer_id: str) -> dict[str, Any]:
        """Fetch one customer by id, e.g. 'C0042'."""
        return crm.get_customer(customer_id)

    @server.tool()
    def list_products(
        category: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List products, optionally filtered by exact category (e.g. 'Electronics')."""
        return crm.list_products(category=category, limit=limit, offset=offset)

    @server.tool()
    def get_product(product_id: str) -> dict[str, Any]:
        """Fetch one product by id, e.g. 'P001'."""
        return crm.get_product(product_id)

    @server.tool()
    def list_orders(
        customer_id: str | None = None,
        product_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 5000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List orders, optionally filtered by customer id, product id and/or
        an order_date range (inclusive, ISO 'YYYY-MM-DD'). Returns up to
        `limit` rows (default 5000); use `offset` to page further."""
        return crm.list_orders(
            customer_id=customer_id,
            product_id=product_id,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )

    @server.tool()
    def get_order(order_id: str) -> dict[str, Any]:
        """Fetch one order by id, e.g. 'O1337'."""
        return crm.get_order(order_id)

    @server.tool()
    def list_invoices(
        customer_id: str | None = None,
        status: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List invoices, optionally filtered by customer id and/or exact
        status ('paid', 'unpaid' or 'overdue')."""
        return crm.list_invoices(customer_id=customer_id, status=status, limit=limit, offset=offset)

    @server.tool()
    def get_invoice(invoice_id: str) -> dict[str, Any]:
        """Fetch one invoice by id, e.g. 'I00001'."""
        return crm.get_invoice(invoice_id)

    @server.tool()
    def search_tickets(
        customer_id: str | None = None,
        category: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        """Search tickets by exact customer id, category and/or status
        ('open' or 'closed'). Sorted by creation date descending by
        default, so `customer_id=...` with `limit=3` returns that
        customer's 3 most recent tickets."""
        return crm.search_tickets(
            customer_id=customer_id,
            category=category,
            status=status,
            limit=limit,
            offset=offset,
            newest_first=newest_first,
        )

    @server.tool()
    def get_ticket(ticket_id: str) -> dict[str, Any]:
        """Fetch one ticket by id, e.g. 'K0001'."""
        return crm.get_ticket(ticket_id)

    @server.tool()
    def update_customer_note(customer_id: str, note: str) -> dict[str, Any]:
        """Overwrite a customer's notes field. Returns the updated customer."""
        return crm.update_customer_note(customer_id, note)

    @server.tool()
    def send_email(customer_id: str, subject: str, body: str) -> dict[str, Any]:
        """Simulate sending an email to a customer (writes to the `outbox`
        table; nothing is actually sent over the network)."""
        return crm.send_email(customer_id, subject, body)

    if with_noise_tools:
        from cmbench.mcp_server.extra_tools import register_noise_tools

        register_noise_tools(server)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="cmbench MCP CRM server (stdio transport)")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--with-noise-tools", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    server = build_server(args.db, with_noise_tools=args.with_noise_tools)
    server.run("stdio")


if __name__ == "__main__":
    main()
