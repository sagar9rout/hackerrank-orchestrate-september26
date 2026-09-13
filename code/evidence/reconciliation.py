"""
Evidence reconciliation.

This module connects normalized evidence to dataset financial events.

Design goals:
- Evidence is untrusted input and never becomes an instruction.
- Explicit cancellation/exclusion wins over generic confirmation.
- Newer same-source evidence wins over older evidence.
- Settled facts win over estimates/forecasts.
- Evidence may exclude or modify an event, but does not perform
  affordability calculations.
- Conflicting evidence is resolved conservatively.
- Reconciliation is deterministic.

The output is a normalized interpretation layer that downstream
financial-state and planning code can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Iterable, Optional

from financial_engine.models import FinancialEvent

from .models import (
    EvidenceBundle,
    EvidenceEffect,
    EvidenceStrength,
    FinancialEvidence,
)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventInterpretation:
    """
    Reconciled interpretation of one financial event.

    include:
        Whether the event should participate in the financial ledger.

    amount:
        Effective amount after evidence-based modification.

    currency:
        Effective currency.

    status:
        Effective lifecycle status.

    rationale:
        Human-readable deterministic explanation of the reconciliation.
    """

    event_id: str
    include: bool
    amount: Optional[Decimal]
    currency: Optional[str]
    status: Optional[str]
    rationale: str

    evidence_ids: tuple[str, ...] = ()

    @property
    def excluded(self) -> bool:
        return not self.include


@dataclass(frozen=True)
class ReconciledLedger:
    """Collection of reconciled event interpretations."""

    user_id: str
    interpretations: tuple[EventInterpretation, ...] = ()

    def for_event(self, event_id: str) -> Optional[EventInterpretation]:
        for item in self.interpretations:
            if item.event_id == event_id:
                return item
        return None

    def included(self) -> tuple[EventInterpretation, ...]:
        return tuple(item for item in self.interpretations if item.include)

    def excluded(self) -> tuple[EventInterpretation, ...]:
        return tuple(item for item in self.interpretations if not item.include)


# ---------------------------------------------------------------------------
# Evidence ranking
# ---------------------------------------------------------------------------


_STRENGTH_RANK = {
    EvidenceStrength.UNUSABLE: 0,
    EvidenceStrength.POSSIBLE: 1,
    EvidenceStrength.LIKELY: 2,
    EvidenceStrength.CONFIRMED: 3,
}


def _evidence_date(item: FinancialEvidence) -> date:
    """
    Missing evidence dates sort before dated evidence.

    Reconciliation must never invent a date.
    """

    return item.evidence_date or date.min


def _source_rank(item: FinancialEvidence) -> int:
    """
    Rank sources for deterministic conflict resolution.

    Dataset evidence represents an actual structured financial record.
    User-provided evidence can explain or amend it.
    Messages and images are useful corroborating evidence.

    This rank is intentionally only a tie-breaker. Explicit exclusion
    and lifecycle precedence are handled separately.
    """

    ranks = {
        "dataset": 4,
        "user_input": 3,
        "message": 2,
        "image": 1,
    }

    return ranks.get(item.source.value, 0)


def _is_explicit_exclusion(item: FinancialEvidence) -> bool:
    """
    Determine whether evidence explicitly says the financial event
    should not be counted.
    """

    if item.effect == EvidenceEffect.EXCLUDE:
        return True

    status = (item.status or "").strip().lower()

    if status in {"cancelled", "canceled", "failed"}:
        return True

    text = item.statement.lower()

    exclusion_terms = (
        "not credited",
        "not received",
        "not withdrawable",
        "unapproved",
        "unearned",
        "cancelled",
        "canceled",
        "failed",
        "do not count",
        "exclude",
        "not reached",
        "still processing",
    )

    return any(term in text for term in exclusion_terms)


def _is_settled(item: FinancialEvidence) -> bool:
    status = (item.status or "").strip().lower()

    if status == "settled":
        return True

    return "settled" in item.statement.lower()


def _sort_evidence(items: Iterable[FinancialEvidence]) -> list[FinancialEvidence]:
    """
    Sort evidence from strongest/newest/most authoritative to weakest.
    """

    usable = [
        item
        for item in items
        if item.is_usable and not item.instruction_ignored
    ]

    return sorted(
        usable,
        key=lambda item: (
            1 if _is_explicit_exclusion(item) else 0,
            _STRENGTH_RANK[item.strength],
            1 if _is_settled(item) else 0,
            _evidence_date(item),
            _source_rank(item),
            item.evidence_id,
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Event reconciliation
# ---------------------------------------------------------------------------


def reconcile_event(
    event: FinancialEvent,
    evidence: Iterable[FinancialEvidence],
) -> EventInterpretation:
    """
    Reconcile one event against linked evidence.

    Important:
    - blank event amounts remain blank unless evidence supplies a
      trustworthy amount;
    - explicit cancellation/exclusion removes an event;
    - a message cannot arbitrarily turn an unrelated event into income;
    - evidence without an event link is not applied here.
    """

    linked = [
        item
        for item in evidence
        if item.event_id == event.event_id
        and item.user_id == event.user_id
        and item.is_usable
        and not item.instruction_ignored
    ]

    if not linked:
        return EventInterpretation(
            event_id=event.event_id,
            include=event.valid_cashflow,
            amount=event.amount,
            currency=event.currency,
            status=event.status,
            rationale=(
                "No linked evidence; retain the normalized dataset "
                "interpretation."
            ),
            evidence_ids=(),
        )

    ordered = _sort_evidence(linked)

    # ---------------------------------------------------------------
    # 1. Explicit exclusions have highest semantic precedence.
    # ---------------------------------------------------------------

    explicit_exclusions = [
        item for item in ordered if _is_explicit_exclusion(item)
    ]

    if explicit_exclusions:
        winner = explicit_exclusions[0]

        return EventInterpretation(
            event_id=event.event_id,
            include=False,
            amount=event.amount,
            currency=event.currency,
            status=winner.status or event.status,
            rationale=(
                "Excluded because linked evidence explicitly indicates "
                "that the event is cancelled, failed, uncredited, "
                "unapproved, or otherwise not realized."
            ),
            evidence_ids=tuple(item.evidence_id for item in ordered),
        )

    # ---------------------------------------------------------------
    # 2. Determine effective amount.
    # ---------------------------------------------------------------

    effective_amount = event.amount

    # Only confirmed/likely evidence with a concrete amount can fill
    # a blank dataset amount.
    amount_candidates = [
        item
        for item in ordered
        if item.amount is not None
        and item.strength
        in {
            EvidenceStrength.CONFIRMED,
            EvidenceStrength.LIKELY,
        }
        and item.effect
        in {
            EvidenceEffect.INCLUDE,
            EvidenceEffect.MODIFY,
            EvidenceEffect.LINK,
        }
    ]

    if effective_amount is None and amount_candidates:
        effective_amount = amount_candidates[0].amount

    # Evidence should not casually overwrite an existing structured
    # dataset amount. An explicit MODIFY evidence item may do so.
    modify_candidates = [
        item
        for item in ordered
        if item.effect == EvidenceEffect.MODIFY
        and item.amount is not None
        and item.strength
        in {
            EvidenceStrength.CONFIRMED,
            EvidenceStrength.LIKELY,
        }
    ]

    if modify_candidates:
        effective_amount = modify_candidates[0].amount

    # ---------------------------------------------------------------
    # 3. Determine effective currency.
    # ---------------------------------------------------------------

    effective_currency = event.currency

    currency_candidates = [
        item
        for item in ordered
        if item.currency
        and item.strength
        in {
            EvidenceStrength.CONFIRMED,
            EvidenceStrength.LIKELY,
        }
        and item.effect
        in {
            EvidenceEffect.INCLUDE,
            EvidenceEffect.MODIFY,
            EvidenceEffect.LINK,
        }
    ]

    if effective_currency is None and currency_candidates:
        effective_currency = currency_candidates[0].currency

    # ---------------------------------------------------------------
    # 4. Determine effective lifecycle status.
    # ---------------------------------------------------------------

    effective_status = event.status

    status_candidates = [
        item
        for item in ordered
        if item.status
        and item.strength
        in {
            EvidenceStrength.CONFIRMED,
            EvidenceStrength.LIKELY,
        }
    ]

    if status_candidates:
        # Explicit evidence of a lifecycle state can amend a weaker
        # dataset interpretation.
        effective_status = status_candidates[0].status

    # ---------------------------------------------------------------
    # 5. Inclusion decision.
    # ---------------------------------------------------------------

    include = event.valid_cashflow

    if any(item.effect == EvidenceEffect.INCLUDE for item in ordered):
        include = True

    if any(item.effect == EvidenceEffect.MODIFY for item in ordered):
        include = True

    # Lifecycle evidence can still make the event unusable.
    if (effective_status or "").lower() in {
        "cancelled",
        "canceled",
        "failed",
        "unrealized",
    }:
        include = False

    rationale_parts: list[str] = []

    if effective_amount != event.amount:
        rationale_parts.append("effective amount supplied or modified by evidence")

    if effective_currency != event.currency:
        rationale_parts.append("effective currency supplied by evidence")

    if effective_status != event.status:
        rationale_parts.append("lifecycle status amended by evidence")

    if not rationale_parts:
        rationale_parts.append(
            "linked evidence corroborates the normalized dataset event"
        )

    return EventInterpretation(
        event_id=event.event_id,
        include=include,
        amount=effective_amount,
        currency=effective_currency,
        status=effective_status,
        rationale="; ".join(rationale_parts) + ".",
        evidence_ids=tuple(item.evidence_id for item in ordered),
    )


# ---------------------------------------------------------------------------
# User-level reconciliation
# ---------------------------------------------------------------------------


def reconcile_events(
    events: Iterable[FinancialEvent],
    evidence: Iterable[FinancialEvidence],
    user_id: Optional[str] = None,
) -> ReconciledLedger:
    """
    Reconcile all events for a user.

    Evidence that is not linked to a particular event is intentionally
    not applied to individual ledger events. Such evidence belongs to
    a separate financial-state extraction stage.

    This prevents an employer message such as "salary confirmed"
    from accidentally changing an unrelated historical event.
    """

    event_list = [
        event
        for event in events
        if user_id is None or event.user_id == user_id
    ]

    evidence_list = [
        item
        for item in evidence
        if user_id is None or item.user_id == user_id
    ]

    interpretations = tuple(
        reconcile_event(event, evidence_list)
        for event in sorted(
            event_list,
            key=lambda item: (
                item.event_date,
                item.event_id,
            ),
        )
    )

    resolved_user_id = user_id

    if resolved_user_id is None:
        users = {event.user_id for event in event_list}
        if len(users) == 1:
            resolved_user_id = next(iter(users))
        else:
            resolved_user_id = "unknown"

    return ReconciledLedger(
        user_id=resolved_user_id,
        interpretations=interpretations,
    )


# ---------------------------------------------------------------------------
# Evidence-level helpers
# ---------------------------------------------------------------------------


def select_user_financial_facts(
    evidence: Iterable[FinancialEvidence],
    user_id: str,
) -> tuple[FinancialEvidence, ...]:
    """
    Return evidence that describes a user's financial state but is not
    directly attached to a particular event.

    Examples:
    - employer confirms next salary;
    - commission is unapproved;
    - temporary pay continues;
    - a future bonus is not approved.

    This output is consumed by the financial-state layer.
    """

    items = [
        item
        for item in evidence
        if item.user_id == user_id
        and item.event_id is None
        and item.is_usable
        and not item.instruction_ignored
    ]

    return tuple(_sort_evidence(items))


def select_confirmed_income_evidence(
    evidence: Iterable[FinancialEvidence],
    user_id: str,
) -> tuple[FinancialEvidence, ...]:
    """
    Select confirmed income evidence that can potentially support
    future-income modeling.

    Pending/unrealized/uncertain evidence is excluded.
    """

    return tuple(
        item
        for item in select_user_financial_facts(evidence, user_id)
        if item.strength == EvidenceStrength.CONFIRMED
        and item.effect == EvidenceEffect.INCLUDE
        and item.category in {"salary", "income", "payroll"}
        and item.amount is not None
    )


def select_excluded_financial_evidence(
    evidence: Iterable[FinancialEvidence],
    user_id: str,
) -> tuple[FinancialEvidence, ...]:
    """
    Return explicit financial exclusions for auditing/debugging.
    """

    return tuple(
        item
        for item in evidence
        if item.user_id == user_id
        and item.is_usable
        and not item.instruction_ignored
        and _is_explicit_exclusion(item)
    )


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def reconcile_bundle(
    events: Iterable[FinancialEvent],
    bundle: EvidenceBundle,
) -> ReconciledLedger:
    """
    Convenience wrapper for an EvidenceBundle.
    """

    return reconcile_events(
        events=events,
        evidence=bundle.items,
        user_id=bundle.user_id,
    )
