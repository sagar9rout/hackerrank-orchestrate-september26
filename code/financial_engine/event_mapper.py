from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from financial_engine.models import FinancialEvent
from financial_engine.loader import parse_date


def _decimal(value: Any):
    if value is None or str(value).strip() == "":
        return None

    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def _get(row: dict, *names: str, default=None):
    for name in names:
        if name in row:
            return row[name]
    return default


def map_event(row: dict) -> FinancialEvent:
    return FinancialEvent(
        event_id=str(_get(row, "event_id", default="")).strip(),
        user_id=str(_get(row, "user_id", default="")).strip(),
        event_date=parse_date(_get(row, "event_date", "date")),
        event_type=str(_get(row, "event_type", default="")).strip(),
        description=str(_get(row, "description", default="")).strip(),
        amount=_decimal(_get(row, "amount")),
        currency=str(_get(row, "currency", default="")).strip(),
        direction=str(_get(row, "direction", default="")).strip().lower(),
        status=str(_get(row, "status", default="")).strip().lower(),
        category=str(_get(row, "category", default="")).strip(),
        flexibility=str(_get(row, "flexibility", default="")).strip().lower(),
        recurring_group=_get(
            row,
            "recurring_group",
            "recurrence_group",
            default=None,
        ),
    )


def map_events(rows: list[dict]) -> list[FinancialEvent]:
    return [map_event(row) for row in rows]
