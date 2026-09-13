from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_date: date
    event_type: str
    description: str
    amount: Optional[Decimal]
    currency: str
    direction: str
    status: str
    category: str
    flexibility: str
    recurring_group: Optional[str] = None

    @property
    def is_cash(self) -> bool:
        return self.direction in {"debit", "credit"}

    @property
    def is_outflow(self) -> bool:
        return self.direction == "debit"

    @property
    def is_inflow(self) -> bool:
        return self.direction == "credit"

    @property
    def is_valid_for_cashflow(self) -> bool:
        return (
            self.status not in {"cancelled", "failed", "unrealized"}
            and self.direction in {"debit", "credit"}
            and self.amount is not None
        )
