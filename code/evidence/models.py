"""
Evidence-layer data models.

This module defines normalized evidence extracted from:
- user messages
- financial messages
- images / OCR
- linked financial events

The evidence layer does NOT make affordability decisions.
It only records what the evidence says, how trustworthy it is,
and what financial fact it may support.

All monetary values use Decimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional


class EvidenceSource(str, Enum):
    """Origin of an evidence item."""

    MESSAGE = "message"
    IMAGE = "image"
    DATASET = "dataset"
    USER_INPUT = "user_input"


class EvidenceStrength(str, Enum):
    """How strongly the evidence supports a financial fact."""

    CONFIRMED = "confirmed"
    LIKELY = "likely"
    POSSIBLE = "possible"
    UNUSABLE = "unusable"


class EvidenceEffect(str, Enum):
    """How evidence should affect financial interpretation."""

    INCLUDE = "include"
    EXCLUDE = "exclude"
    MODIFY = "modify"
    LINK = "link"
    INFORMATIONAL = "informational"


@dataclass(frozen=True)
class FinancialEvidence:
    """
    Normalized piece of evidence.

    Examples:
    - Employer confirms next salary.
    - Commission is unapproved and must be excluded.
    - Refund was initiated but has not reached the account.
    - Image shows an unpaid bill.
    """

    evidence_id: str
    user_id: str

    source: EvidenceSource
    strength: EvidenceStrength
    effect: EvidenceEffect

    statement: str

    event_id: Optional[str] = None
    request_id: Optional[str] = None

    evidence_date: Optional[date] = None

    amount: Optional[Decimal] = None
    currency: Optional[str] = None

    category: Optional[str] = None

    # Optional structured fields for financial interpretation.
    status: Optional[str] = None
    direction: Optional[str] = None

    # Keeps track of whether an instruction-like message was treated
    # as untrusted content rather than as a rule for the agent.
    contains_instruction: bool = False
    instruction_ignored: bool = False

    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the normalized evidence object."""

        if not self.evidence_id:
            raise ValueError("evidence_id cannot be empty")

        if not self.user_id:
            raise ValueError("user_id cannot be empty")

        if not self.statement.strip():
            raise ValueError("statement cannot be empty")

        if self.amount is not None:
            if not isinstance(self.amount, Decimal):
                raise TypeError("amount must be Decimal")

            if self.amount < Decimal("0"):
                raise ValueError("amount cannot be negative")

        if self.currency is not None and not self.currency.strip():
            raise ValueError("currency cannot be blank")

        if self.event_id is not None and not self.event_id.strip():
            raise ValueError("event_id cannot be blank")

        if self.request_id is not None and not self.request_id.strip():
            raise ValueError("request_id cannot be blank")

    @property
    def is_usable(self) -> bool:
        """Return whether this evidence is strong enough to use."""

        return self.strength != EvidenceStrength.UNUSABLE

    @property
    def is_confirmed(self) -> bool:
        """Return whether the evidence represents a confirmed fact."""

        return self.strength == EvidenceStrength.CONFIRMED

    @property
    def should_affect_ledger(self) -> bool:
        """
        Return whether the evidence is intended to alter financial
        interpretation.

        Informational and unusable evidence must not directly modify
        the financial ledger.
        """

        return (
            self.is_usable
            and self.effect in {
                EvidenceEffect.INCLUDE,
                EvidenceEffect.EXCLUDE,
                EvidenceEffect.MODIFY,
                EvidenceEffect.LINK,
            }
        )


@dataclass(frozen=True)
class EvidenceBundle:
    """
    Collection of evidence belonging to one user/request context.
    """

    user_id: str
    request_id: Optional[str]
    items: tuple[FinancialEvidence, ...] = ()

    def for_event(self, event_id: str) -> tuple[FinancialEvidence, ...]:
        """Return evidence explicitly linked to an event."""

        return tuple(
            item
            for item in self.items
            if item.event_id == event_id
        )

    def confirmed(self) -> tuple[FinancialEvidence, ...]:
        """Return only confirmed evidence."""

        return tuple(
            item
            for item in self.items
            if item.strength == EvidenceStrength.CONFIRMED
        )

    def affecting_ledger(self) -> tuple[FinancialEvidence, ...]:
        """Return evidence that may affect ledger interpretation."""

        return tuple(
            item
            for item in self.items
            if item.should_affect_ledger
        )


def decimal_or_none(value: object) -> Optional[Decimal]:
    """
    Safely convert a value into Decimal.

    Empty values become None.
    Existing Decimal values are returned unchanged.
    """

    if value is None:
        return None

    if isinstance(value, Decimal):
        return value

    text = str(value).strip()

    if not text:
        return None

    return Decimal(text)


def normalize_currency(value: object) -> Optional[str]:
    """Normalize a currency code without inventing one."""

    if value is None:
        return None

    text = str(value).strip().upper()

    return text or None
