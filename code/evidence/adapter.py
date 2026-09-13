from __future__ import annotations

"""
Evidence-to-financial-engine adapter.

This module converts the project's current FinancialEvidence model into
small, deterministic override structures that can be consumed by the
financial engine.

Important design rule:
    Reconciliation remains the source of truth for individual events.

This adapter is primarily for user-level financial facts such as:
    - confirmed salary changes
    - temporary salary amounts
    - confirmed future income
    - expense changes
    - event amount disclosures
    - explicit pending/refund exclusions
    - internal-transfer annotations

It deliberately does NOT introduce the older EvidenceFact/FactType model.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Tuple

from .models import (
    EvidenceEffect,
    EvidenceStrength,
    FinancialEvidence,
)
from .reconciliation import (
    select_excluded_financial_evidence,
    select_user_financial_facts,
)


# ---------------------------------------------------------------------------
# Override models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SalaryOverride:
    """
    User-level salary/pay override.

    new_recurring_amount:
        Replacement recurring amount when evidence explicitly establishes
        a new recurring salary/pay amount.

    currency:
        Currency of the replacement amount.

    effective_from:
        Date from which the replacement amount should be considered active.

    termination_date:
        Optional date after which the recurring salary should no longer
        be projected.

    one_time_additions:
        Confirmed future income events represented as:
            (date, amount, currency)
    """

    new_recurring_amount: Optional[Decimal] = None
    currency: Optional[str] = None
    effective_from: Optional[date] = None
    termination_date: Optional[date] = None

    one_time_additions: Tuple[
        Tuple[Optional[date], Decimal, str],
        ...,
    ] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExpenseOverride:
    """
    User-level expense modification.

    relative_multiplier:
        Example:
            Decimal("0.5") means reduce the recurring amount to 50%.

    absolute_amount:
        Explicit replacement amount.

    effective_from:
        Date from which the change applies.
    """

    category: str
    relative_multiplier: Optional[Decimal] = None
    absolute_amount: Optional[Decimal] = None
    effective_from: Optional[date] = None


@dataclass(frozen=True)
class EvidenceOverrides:
    """
    Deterministic collection of evidence-derived engine overrides.
    """

    salary_overrides: Dict[str, SalaryOverride] = field(default_factory=dict)

    expense_overrides: Dict[
        Tuple[str, str],
        ExpenseOverride,
    ] = field(default_factory=dict)

    event_amount_fills: Dict[
        str,
        Tuple[Decimal, str, float],
    ] = field(default_factory=dict)

    excluded_as_pending: List[str] = field(default_factory=list)

    internal_transfer_flags: List[str] = field(default_factory=list)

    rejected_or_unusable: List[FinancialEvidence] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usable(item: FinancialEvidence) -> bool:
    return item.is_usable and not item.instruction_ignored


def _confirmed(item: FinancialEvidence) -> bool:
    return (
        _usable(item)
        and item.strength == EvidenceStrength.CONFIRMED
    )


def _likely_or_confirmed(item: FinancialEvidence) -> bool:
    return (
        _usable(item)
        and item.strength
        in {
            EvidenceStrength.CONFIRMED,
            EvidenceStrength.LIKELY,
        }
    )


def _normalise_currency(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    value = value.strip().upper()

    return value or None


def _safe_amount(value: Optional[Decimal]) -> Optional[Decimal]:
    if value is None:
        return None

    if value < Decimal("0"):
        return None

    return value


def _evidence_date(item: FinancialEvidence) -> Optional[date]:
    return item.evidence_date


def _category(item: FinancialEvidence) -> str:
    return (item.category or "").strip().lower()


def _statement(item: FinancialEvidence) -> str:
    return (item.statement or "").strip().lower()


def _metadata(item: FinancialEvidence, key: str) -> Optional[str]:
    value = item.metadata.get(key)

    if value is None:
        return None

    value = str(value).strip()

    return value or None


def _is_salary_fact(item: FinancialEvidence) -> bool:
    category = _category(item)

    if category in {
        "salary",
        "payroll",
        "income",
    }:
        return True

    text = _statement(item)

    salary_terms = (
        "salary",
        "pay",
        "payroll",
        "monthly pay",
        "monthly salary",
        "base salary",
        "net salary",
    )

    return any(term in text for term in salary_terms)


def _is_expense_change(item: FinancialEvidence) -> bool:
    category = _category(item)

    if category in {
        "expense_change",
        "recurring_expense_change",
        "subscription_change",
        "expense",
    }:
        return True

    text = _statement(item)

    return any(
        term in text
        for term in (
            "reduce",
            "reduced",
            "decrease",
            "lower",
            "stop",
            "cancel",
            "expense change",
        )
    )


def _is_future_income(item: FinancialEvidence) -> bool:
    text = _statement(item)

    if _category(item) in {
        "future_income",
        "confirmed_future_income",
    }:
        return True

    return any(
        term in text
        for term in (
            "next salary",
            "next payroll",
            "future salary",
            "future income",
            "upcoming salary",
            "confirmed salary",
        )
    )


def _is_internal_transfer(item: FinancialEvidence) -> bool:
    if _category(item) in {
        "internal_transfer",
        "transfer",
    }:
        return True

    text = _statement(item)

    return (
        "internal transfer" in text
        or "same-account transfer" in text
        or "matching debit and credit" in text
    )


def _is_pending_or_uncredited(item: FinancialEvidence) -> bool:
    text = _statement(item)

    exclusion_words = (
        "pending",
        "not credited",
        "not received",
        "not reached",
        "uncredited",
        "processing",
        "still processing",
        "not withdrawable",
        "not approved",
        "unapproved",
        "unearned",
    )

    return any(word in text for word in exclusion_words)


def _is_amount_disclosure(item: FinancialEvidence) -> bool:
    if item.amount is None:
        return False

    if item.event_id is None:
        return False

    if item.effect not in {
        EvidenceEffect.INCLUDE,
        EvidenceEffect.MODIFY,
        EvidenceEffect.LINK,
    }:
        return False

    return True


def _parse_multiplier(item: FinancialEvidence) -> Optional[Decimal]:
    """
    Read an explicitly supplied relative multiplier.

    Supported metadata examples:
        multiplier = "0.5"
        relative_multiplier = "0.75"
    """

    raw = (
        _metadata(item, "relative_multiplier")
        or _metadata(item, "multiplier")
    )

    if raw is None:
        return None

    try:
        value = Decimal(raw)
    except Exception:
        return None

    if value < Decimal("0"):
        return None

    return value


def _parse_effective_date(item: FinancialEvidence) -> Optional[date]:
    raw = (
        _metadata(item, "effective_from")
        or _metadata(item, "effective_date")
    )

    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass

    return item.evidence_date


def _sort_items(items: Iterable[FinancialEvidence]) -> List[FinancialEvidence]:
    return sorted(
        items,
        key=lambda item: (
            item.evidence_date or date.min,
            item.evidence_id,
        ),
    )


# ---------------------------------------------------------------------------
# Salary handling
# ---------------------------------------------------------------------------


def _build_salary_override(
    user_id: str,
    items: Iterable[FinancialEvidence],
) -> Optional[SalaryOverride]:
    candidates = [
        item
        for item in items
        if item.user_id == user_id
        and _confirmed(item)
        and _is_salary_fact(item)
        and item.amount is not None
        and item.effect == EvidenceEffect.INCLUDE
    ]

    candidates = _sort_items(candidates)

    recurring: Optional[FinancialEvidence] = None
    future: List[FinancialEvidence] = []

    for item in candidates:
        if _is_future_income(item):
            future.append(item)
        else:
            recurring = item

    amount: Optional[Decimal] = None
    currency: Optional[str] = None
    effective_from: Optional[date] = None

    if recurring is not None:
        amount = _safe_amount(recurring.amount)
        currency = _normalise_currency(recurring.currency)
        effective_from = _parse_effective_date(recurring)

    one_time_additions: List[
        Tuple[Optional[date], Decimal, str]
    ] = []

    for item in future:
        safe_amount = _safe_amount(item.amount)

        if safe_amount is None:
            continue

        item_currency = _normalise_currency(item.currency)

        if item_currency is None:
            continue

        one_time_additions.append(
            (
                _parse_effective_date(item),
                safe_amount,
                item_currency,
            )
        )

    if (
        amount is None
        and not one_time_additions
    ):
        return None

    return SalaryOverride(
        new_recurring_amount=amount,
        currency=currency,
        effective_from=effective_from,
        one_time_additions=tuple(
            sorted(
                one_time_additions,
                key=lambda value: (
                    value[0] or date.max,
                    value[1],
                    value[2],
                ),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Expense handling
# ---------------------------------------------------------------------------


def _build_expense_override(
    user_id: str,
    item: FinancialEvidence,
) -> Optional[ExpenseOverride]:
    if item.user_id != user_id:
        return None

    if not _confirmed(item):
        return None

    if not _is_expense_change(item):
        return None

    category = _category(item)

    if not category:
        return None

    multiplier = _parse_multiplier(item)

    absolute_amount = _safe_amount(item.amount)

    if multiplier is None and absolute_amount is None:
        return None

    return ExpenseOverride(
        category=category,
        relative_multiplier=multiplier,
        absolute_amount=absolute_amount,
        effective_from=_parse_effective_date(item),
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_overrides(
    all_evidence: Iterable[FinancialEvidence],
) -> EvidenceOverrides:
    """
    Build deterministic engine overrides from current FinancialEvidence.

    No LLM calls occur here.

    The function is intentionally conservative:
        - only usable evidence is applied;
        - only confirmed evidence changes financial state;
        - amounts must be non-negative;
        - unrelated unlinked evidence is not forced onto events;
        - explicit exclusions are retained for auditing.
    """

    evidence = list(all_evidence)

    user_ids = sorted(
        {
            item.user_id
            for item in evidence
            if item.user_id
        }
    )

    salary_overrides: Dict[str, SalaryOverride] = {}

    for user_id in user_ids:
        override = _build_salary_override(
            user_id=user_id,
            items=evidence,
        )

        if override is not None:
            salary_overrides[user_id] = override

    expense_overrides: Dict[
        Tuple[str, str],
        ExpenseOverride,
    ] = {}

    expense_candidates = _sort_items(
        item
        for item in evidence
        if _is_expense_change(item)
        and _confirmed(item)
    )

    for item in expense_candidates:
        override = _build_expense_override(
            user_id=item.user_id,
            item=item,
        )

        if override is None:
            continue

        key = (
            item.user_id,
            override.category,
        )

        # Newer confirmed evidence supersedes older evidence.
        existing = expense_overrides.get(key)

        if existing is None:
            expense_overrides[key] = override
            continue

        existing_date = existing.effective_from or date.min
        new_date = override.effective_from or date.min

        if new_date >= existing_date:
            expense_overrides[key] = override

    # ---------------------------------------------------------------
    # Event amount fills
    # ---------------------------------------------------------------

    event_amount_fills: Dict[
        str,
        Tuple[Decimal, str, float],
    ] = {}

    amount_candidates = _sort_items(
        item
        for item in evidence
        if _is_amount_disclosure(item)
        and _likely_or_confirmed(item)
        and item.amount is not None
        and item.currency is not None
    )

    for item in amount_candidates:
        amount = _safe_amount(item.amount)
        currency = _normalise_currency(item.currency)

        if amount is None or currency is None:
            continue

        confidence = (
            1.0
            if item.strength == EvidenceStrength.CONFIRMED
            else 0.75
        )

        candidate = (
            amount,
            currency,
            confidence,
        )

        existing = event_amount_fills.get(item.event_id)

        if existing is None:
            event_amount_fills[item.event_id] = candidate
            continue

        # Prefer stronger evidence.
        if confidence > existing[2]:
            event_amount_fills[item.event_id] = candidate

    # ---------------------------------------------------------------
    # Pending / uncredited exclusions
    # ---------------------------------------------------------------

    excluded_as_pending: List[str] = []

    for item in _sort_items(evidence):
        if not _usable(item):
            continue

        if item.event_id is None:
            continue

        if item.effect == EvidenceEffect.EXCLUDE:
            excluded_as_pending.append(item.event_id)
            continue

        if _is_pending_or_uncredited(item):
            excluded_as_pending.append(item.event_id)

    excluded_as_pending = sorted(
        set(excluded_as_pending)
    )

    # ---------------------------------------------------------------
    # Internal transfers
    # ---------------------------------------------------------------

    internal_transfer_flags: List[str] = []

    for item in _sort_items(evidence):
        if not _usable(item):
            continue

        if item.event_id is None:
            continue

        if _is_internal_transfer(item):
            internal_transfer_flags.append(item.event_id)

    internal_transfer_flags = sorted(
        set(internal_transfer_flags)
    )

    # ---------------------------------------------------------------
    # Explicit exclusions / unusable evidence for audit
    # ---------------------------------------------------------------

    rejected_or_unusable: List[FinancialEvidence] = []

    rejected_or_unusable.extend(
        item
        for item in evidence
        if not item.is_usable
        or item.instruction_ignored
    )

    rejected_or_unusable.extend(
        select_excluded_financial_evidence(
            evidence=evidence,
            user_id=user_id,
        )
        for user_id in []
    )

    # The comprehension above intentionally does not duplicate exclusion
    # selection. Build it directly for deterministic uniqueness.
    rejected_or_unusable = sorted(
        {
            item.evidence_id: item
            for item in rejected_or_unusable
        }.values(),
        key=lambda item: item.evidence_id,
    )

    return EvidenceOverrides(
        salary_overrides=salary_overrides,
        expense_overrides=expense_overrides,
        event_amount_fills=event_amount_fills,
        excluded_as_pending=excluded_as_pending,
        internal_transfer_flags=internal_transfer_flags,
        rejected_or_unusable=rejected_or_unusable,
    )


# ---------------------------------------------------------------------------
# User/request helpers
# ---------------------------------------------------------------------------


def build_overrides_for_user(
    all_evidence: Iterable[FinancialEvidence],
    user_id: str,
) -> EvidenceOverrides:
    """
    Build overrides using evidence belonging to one user.
    """

    evidence = [
        item
        for item in all_evidence
        if item.user_id == user_id
    ]

    return build_overrides(evidence)


def build_overrides_for_request(
    all_evidence: Iterable[FinancialEvidence],
    request_id: str,
) -> EvidenceOverrides:
    """
    Build overrides from evidence associated with one request.

    User-level facts without a request_id are intentionally not included
    because they cannot safely be assigned to this request in isolation.
    """

    evidence = [
        item
        for item in all_evidence
        if item.request_id == request_id
    ]

    return build_overrides(evidence)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_overrides(
    overrides: EvidenceOverrides,
) -> List[str]:
    """
    Return deterministic validation errors.

    An empty list means the structure is internally valid.
    """

    errors: List[str] = []

    for user_id, salary in overrides.salary_overrides.items():
        if salary.new_recurring_amount is not None:
            if salary.new_recurring_amount < Decimal("0"):
                errors.append(
                    f"negative salary override for {user_id}"
                )

        if salary.effective_from is not None:
            if not isinstance(salary.effective_from, date):
                errors.append(
                    f"invalid salary effective date for {user_id}"
                )

        for (
            addition_date,
            amount,
            currency,
        ) in salary.one_time_additions:
            if amount < Decimal("0"):
                errors.append(
                    f"negative future income for {user_id}"
                )

            if not currency:
                errors.append(
                    f"missing currency for future income for {user_id}"
                )

            if (
                addition_date is not None
                and not isinstance(addition_date, date)
            ):
                errors.append(
                    f"invalid future income date for {user_id}"
                )

    for (
        user_id,
        category,
    ), expense in overrides.expense_overrides.items():

        if not user_id:
            errors.append("expense override missing user_id")

        if not category:
            errors.append(
                f"expense override missing category for {user_id}"
            )

        if expense.relative_multiplier is not None:
            if expense.relative_multiplier < Decimal("0"):
                errors.append(
                    f"negative expense multiplier for "
                    f"{user_id}/{category}"
                )

        if expense.absolute_amount is not None:
            if expense.absolute_amount < Decimal("0"):
                errors.append(
                    f"negative expense amount for "
                    f"{user_id}/{category}"
                )

    for event_id, (
        amount,
        currency,
        confidence,
    ) in overrides.event_amount_fills.items():

        if amount < Decimal("0"):
            errors.append(
                f"negative event amount fill for {event_id}"
            )

        if not currency:
            errors.append(
                f"missing currency for event amount fill {event_id}"
            )

        if not 0.0 <= confidence <= 1.0:
            errors.append(
                f"invalid confidence for event amount fill {event_id}"
            )

    overlap = (
        set(overrides.excluded_as_pending)
        & set(overrides.internal_transfer_flags)
    )

    for event_id in sorted(overlap):
        errors.append(
            f"event appears in both pending exclusion and "
            f"internal-transfer flags: {event_id}"
        )

    return errors


# ---------------------------------------------------------------------------
# Module smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Evidence adapter module loaded successfully.")
    print("Use build_overrides(all_evidence) from the full pipeline.")
