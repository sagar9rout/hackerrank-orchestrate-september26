from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from financial_engine.models import FinancialEvent
from financial_engine.lifecycle import (
    is_pending,
    is_scheduled,
    is_settled,
)


@dataclass(frozen=True)
class CashPoint:
    day: date
    balance: Decimal


@dataclass(frozen=True)
class SimulationResult:
    start_date: date
    end_date: date
    starting_balance: Decimal
    minimum_balance: Decimal
    lowest_balance: Decimal
    lowest_balance_date: date
    daily_balances: tuple[CashPoint, ...]

    @property
    def stays_safe(self) -> bool:
        return self.lowest_balance >= self.minimum_balance


def simulate(
    *,
    starting_balance: Decimal,
    minimum_balance: Decimal,
    start_date: date,
    events: list[FinancialEvent],
    days: int = 90,
) -> SimulationResult:
    """
    Deterministic 90-day cash-flow simulation.

    Included:
    - settled cash events
    - scheduled cash events
    - pending debits as future obligations

    Excluded:
    - pending credits
    - cancelled events
    - failed events
    - unrealized investment events
    - non-cash/internal-transfer events
    - events without a resolved amount

    Events occurring on the same day are processed deterministically:
    debits first, then credits, then event_id.
    """

    if days < 0:
        raise ValueError("days must be non-negative")

    if starting_balance < 0:
        raise ValueError("starting_balance must be non-negative")

    if minimum_balance < 0:
        raise ValueError("minimum_balance must be non-negative")

    end_date = start_date + timedelta(days=days)

    usable_events: list[FinancialEvent] = []

    for event in events:
        # Blank amounts are unresolved, not zero.
        if event.amount is None:
            continue

        # Ignore lifecycle states that cannot represent a real
        # future cash movement.
        if event.status in {"cancelled", "failed", "unrealized"}:
            continue

        # Ignore events outside the simulation window.
        if event.event_date < start_date or event.event_date > end_date:
            continue

        # Only real cash directions participate.
        if event.direction not in {"debit", "credit"}:
            continue

        # Pending credits cannot be counted as available money.
        if is_pending(event) and event.direction == "credit":
            continue

        # Only settled, scheduled and pending events are usable.
        if not (
            is_settled(event)
            or is_scheduled(event)
            or is_pending(event)
        ):
            continue

        usable_events.append(event)

    by_date: dict[date, list[FinancialEvent]] = {}

    for event in usable_events:
        by_date.setdefault(event.event_date, []).append(event)

    balance = Decimal(starting_balance)
    lowest = balance
    lowest_date = start_date

    points: list[CashPoint] = [
        CashPoint(
            day=start_date,
            balance=balance,
        )
    ]

    for offset in range(days + 1):
        day = start_date + timedelta(days=offset)

        day_events = sorted(
            by_date.get(day, []),
            key=lambda event: (
                0 if event.direction == "debit" else 1,
                event.event_id,
            ),
        )

        for event in day_events:
            amount = event.amount

            if amount is None:
                continue

            if amount < 0:
                raise ValueError(
                    f"Negative event amount: {event.event_id}"
                )

            if event.direction == "debit":
                balance -= amount
            else:
                balance += amount

            if balance < lowest:
                lowest = balance
                lowest_date = day

        if offset > 0:
            points.append(
                CashPoint(
                    day=day,
                    balance=balance,
                )
            )

    return SimulationResult(
        start_date=start_date,
        end_date=end_date,
        starting_balance=starting_balance,
        minimum_balance=minimum_balance,
        lowest_balance=lowest,
        lowest_balance_date=lowest_date,
        daily_balances=tuple(points),
    )
