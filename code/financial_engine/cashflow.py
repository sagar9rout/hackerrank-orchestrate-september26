from __future__ import annotations

from financial_engine.models import FinancialEvent


EXCLUDED_STATUSES = {
    "cancelled",
    "failed",
    "unrealized",
}


def is_cashflow_event(event: FinancialEvent) -> bool:
    """
    Return True only when the event represents usable/realized cash movement.

    Rules:
    - cancelled, failed and unrealized events are ignored
    - non-cash events are ignored
    - events with missing amounts are unresolved and ignored here
    """
    if event.status in EXCLUDED_STATUSES:
        return False

    if event.direction not in {"debit", "credit"}:
        return False

    if event.amount is None:
        return False

    return True


def is_cashflow_credit(event: FinancialEvent) -> bool:
    return is_cashflow_event(event) and event.direction == "credit"


def is_cashflow_debit(event: FinancialEvent) -> bool:
    return is_cashflow_event(event) and event.direction == "debit"


def filter_cashflow(events: list[FinancialEvent]) -> list[FinancialEvent]:
    return [event for event in events if is_cashflow_event(event)]
