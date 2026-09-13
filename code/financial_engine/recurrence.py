from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from statistics import median
from typing import Optional

from financial_engine.models import FinancialEvent


@dataclass(frozen=True)
class RecurrenceSeries:
    key: str
    user_id: str
    description: str
    direction: str
    currency: str
    category: str
    flexibility: str
    interval_days: int
    typical_amount: Decimal
    confidence: float
    last_date: date


# Strong recurring-obligation indicators.
# These are intentionally conservative.
_OBLIGATION_KEYWORDS = (
    "rent",
    "lease",
    "utility",
    "utilities",
    "water",
    "power",
    "electric",
    "electricity",
    "internet",
    "phone",
    "mobile",
    "streaming",
    "subscription",
    "cloud storage",
    "insurance",
    "loan",
    "mortgage",
    "salary",
    "payroll",
    "pay",
    "stipend",
    "pension",
)


def _normal_key(event: FinancialEvent) -> str:
    return "|".join(
        [
            event.user_id,
            event.description.strip().lower(),
            event.direction,
            event.currency,
        ]
    )


def _looks_like_obligation(event: FinancialEvent) -> bool:
    text = f"{event.description} {event.category}".lower()

    return any(keyword in text for keyword in _OBLIGATION_KEYWORDS)


def _median_decimal(values: list[Decimal]) -> Decimal:
    if not values:
        raise ValueError("Cannot calculate median of empty list")

    ordered = sorted(values)
    n = len(ordered)

    if n % 2:
        return ordered[n // 2]

    return (ordered[n // 2 - 1] + ordered[n // 2]) / Decimal("2")


def _regularity_score(gaps: list[int]) -> float:
    if not gaps:
        return 0.0

    target = median(gaps)

    if target <= 0:
        return 0.0

    close = sum(
        1
        for gap in gaps
        if abs(gap - target) <= max(2, round(target * 0.10))
    )

    return close / len(gaps)


def detect_recurrencies(
    events: list[FinancialEvent],
    as_of: date,
) -> list[RecurrenceSeries]:
    """
    Detect conservative recurring cash-flow series.

    Only settled historical events are used as evidence.
    Generic repeated discretionary purchases are deliberately excluded.
    """

    groups: dict[str, list[FinancialEvent]] = {}

    for event in events:
        if event.event_date > as_of:
            continue

        if event.status != "settled":
            continue

        if event.amount is None:
            continue

        if event.direction not in {"debit", "credit"}:
            continue

        # Important:
        # Do not turn ordinary repeated shopping/food/transport
        # transactions into mandatory future cash flows.
        if not _looks_like_obligation(event):
            continue

        groups.setdefault(_normal_key(event), []).append(event)

    result: list[RecurrenceSeries] = []

    for key, group in groups.items():
        group.sort(key=lambda e: (e.event_date, e.event_id))

        if len(group) < 3:
            continue

        gaps = [
            (group[i].event_date - group[i - 1].event_date).days
            for i in range(1, len(group))
        ]

        positive_gaps = [gap for gap in gaps if gap > 0]

        if len(positive_gaps) < 2:
            continue

        interval = int(round(median(positive_gaps)))

        # Monthly-ish obligations.
        if not 25 <= interval <= 35:
            continue

        regularity = _regularity_score(positive_gaps)

        if regularity < 0.67:
            continue

        # Require reasonably recent evidence.
        if (as_of - group[-1].event_date).days > 120:
            continue

        amounts = [event.amount for event in group if event.amount is not None]

        if not amounts:
            continue

        typical_amount = _median_decimal(amounts)

        # Confidence is deliberately simple and deterministic.
        confidence = min(
            0.99,
            0.70
            + 0.20 * regularity
            + 0.05 * min(len(group), 6) / 6,
        )

        latest = group[-1]

        result.append(
            RecurrenceSeries(
                key=key,
                user_id=latest.user_id,
                description=latest.description,
                direction=latest.direction,
                currency=latest.currency,
                category=latest.category,
                flexibility=latest.flexibility,
                interval_days=interval,
                typical_amount=typical_amount,
                confidence=round(confidence, 2),
                last_date=latest.event_date,
            )
        )

    result.sort(
        key=lambda series: (
            series.user_id,
            series.direction,
            series.description.lower(),
        )
    )

    return result


def project_recurrencies(
    series: list[RecurrenceSeries],
    start_date: date,
    days: int = 90,
) -> list[FinancialEvent]:
    """
    Project detected recurring series into the forecast window.

    This is only a forecast. Explicit future events must later override
    or suppress projections when the adapter/reconciliation layer is added.
    """

    end_date = start_date + timedelta(days=days)

    projected: list[FinancialEvent] = []

    for item in series:
        next_date = item.last_date + timedelta(days=item.interval_days)

        while next_date <= end_date:
            if next_date >= start_date:
                projected.append(
                    FinancialEvent(
                        event_id=(
                            f"forecast:{item.key}:"
                            f"{next_date.isoformat()}"
                        ),
                        user_id=item.user_id,
                        event_date=next_date,
                        event_type=(
                            "income"
                            if item.direction == "credit"
                            else "subscription"
                        ),
                        description=item.description,
                        amount=item.typical_amount,
                        currency=item.currency,
                        direction=item.direction,
                        status="scheduled",
                        category=item.category,
                        flexibility=item.flexibility,
                        recurring_group=item.key,
                    )
                )

            next_date += timedelta(days=item.interval_days)

    projected.sort(
        key=lambda event: (
            event.event_date,
            0 if event.direction == "debit" else 1,
            event.event_id,
        )
    )

    return projected
