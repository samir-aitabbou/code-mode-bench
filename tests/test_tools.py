import sqlite3
from pathlib import Path

import pytest

from cmbench.mcp_server.tools import CrmTools, NotFoundError


def _crm(db_path: Path) -> CrmTools:
    return CrmTools(sqlite3.connect(db_path))


def test_get_customer_found(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    customer = crm.get_customer("C0042")
    assert customer["id"] == "C0042"
    assert customer["segment"] in ("VIP", "standard")


def test_get_customer_not_found_raises(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    with pytest.raises(NotFoundError):
        crm.get_customer("C9999")


def test_list_customers_filters_by_region_and_segment(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    rows = crm.list_customers(region="South", segment="VIP", limit=1000)
    assert rows
    assert all(r["region"] == "South" and r["segment"] == "VIP" for r in rows)


def test_list_orders_respects_limit_and_date_range(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    rows = crm.list_orders(date_from="2026-10-01", date_to="2026-10-31", limit=5000)
    assert rows
    assert all("2026-10-01" <= r["order_date"] <= "2026-10-31" for r in rows)

    limited = crm.list_orders(limit=7)
    assert len(limited) == 7


def test_get_order_found(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    order = crm.get_order("O1337")
    assert order["id"] == "O1337"


def test_list_invoices_filters_by_status(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    rows = crm.list_invoices(status="overdue", limit=10000)
    assert rows
    assert all(r["status"] == "overdue" for r in rows)


def test_search_tickets_newest_first_for_customer(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    rows = crm.search_tickets(customer_id="C0107", limit=3)
    assert all(r["customer_id"] == "C0107" for r in rows)
    dates = [r["created_at"] for r in rows]
    assert dates == sorted(dates, reverse=True)


def test_update_customer_note_persists(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    updated = crm.update_customer_note("C0001", "Q4 follow-up")
    assert updated["notes"] == "Q4 follow-up"

    # a fresh connection must see the committed write
    crm2 = _crm(seeded_db)
    assert crm2.get_customer("C0001")["notes"] == "Q4 follow-up"


def test_update_customer_note_unknown_customer_raises(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    with pytest.raises(NotFoundError):
        crm.update_customer_note("C9999", "note")


def test_send_email_writes_to_outbox(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    result = crm.send_email("C0001", subject="Reminder", body="Please pay your invoice.")
    assert result["customer_id"] == "C0001"
    assert result["subject"] == "Reminder"

    rows = crm.conn.execute("SELECT * FROM outbox WHERE customer_id = 'C0001'").fetchall()
    assert len(rows) == 1


def test_send_email_unknown_customer_raises(seeded_db: Path) -> None:
    crm = _crm(seeded_db)
    with pytest.raises(NotFoundError):
        crm.send_email("C9999", subject="x", body="y")
