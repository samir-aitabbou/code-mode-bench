"""Shared vocabulary for the synthetic mini-CRM, used by both the data
generator and the MCP server / graders so they agree on valid values.
"""

from __future__ import annotations

REGIONS: tuple[str, ...] = ("North", "South", "East", "West", "Central")
SEGMENTS: tuple[str, ...] = ("VIP", "standard")

PRODUCT_CATEGORIES: tuple[str, ...] = (
    "Electronics",
    "Clothing",
    "Home",
    "Books",
    "Toys",
    "Sports",
    "Food",
    "Office",
)

INVOICE_STATUSES: tuple[str, ...] = ("paid", "unpaid", "overdue")
UNPAID_STATUSES: tuple[str, ...] = ("unpaid", "overdue")

TICKET_CATEGORIES: tuple[str, ...] = (
    "billing",
    "shipping",
    "product_defect",
    "support",
    "other",
)
TICKET_STATUSES: tuple[str, ...] = ("open", "closed")
