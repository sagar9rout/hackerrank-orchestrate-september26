from __future__ import annotations

from financial_engine.models import FinancialEvent


def is_settled(event: FinancialEvent) -> bool:
    return (
        event.status == "settled"
        and event.direction in {"debit", "credit"}
        and event.amount is not None
    )


def is_pending(event: FinancialEvent) -> bool:
    return (
        event.status == "pending"
        and event.direction in {"debit", "credit"}
        and event.amount is not None
    )


def is_scheduled(event: FinancialEvent) -> bool:
    return (
        event.status == "scheduled"
        and event.direction in {"debit", "credit"}
        and event.amount is not None
    )


def historical_settled(events: list[FinancialEvent]):
    return [e for e in events if is_settled(e)]


def future_confirmed_cashflow(events: list[FinancialEvent]):
    """
    Scheduled cash movements are future commitments/credits.

    They are NOT treated as historical settled transactions.
    """
    return [e for e in events if is_scheduled(e)]


def pending_events(events: list[FinancialEvent]):
    """
    Pending transactions are tracked separately.

    In particular, pending credits must NOT increase available cash.
    """
    return [e for e in events if is_pending(e)]
