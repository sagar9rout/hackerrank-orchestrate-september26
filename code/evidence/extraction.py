"""
Deterministic extraction of financial evidence from untrusted text.

The extractor does not make affordability decisions.
It converts messages into normalized FinancialEvidence objects.

Important safety rule:
message text is evidence, not executable instructions.
Instruction-like text is recorded but never treated as an agent rule.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Iterable, Optional

from evidence.models import (
    EvidenceEffect,
    EvidenceSource,
    EvidenceStrength,
    FinancialEvidence,
    normalize_currency,
)


_CONFIRMED_PATTERNS = (
    r"\bconfirmed\b",
    r"\bapproved\b",
    r"\bverified\b",
    r"\bguaranteed\b",
    r"\bregular salary\b",
    r"\bnext payroll\b",
    r"\bnext salary\b",
    r"\bwill be paid\b",
    r"\bpayment is scheduled\b",
    r"\bsalary continues\b",
)

_UNCERTAIN_PATTERNS = (
    r"\bmay\b",
    r"\bmight\b",
    r"\bcan change\b",
    r"\bexpected\b",
    r"\bestimated\b",
    r"\bforecast\b",
    r"\bpotential\b",
    r"\bunapproved\b",
    r"\bunearned\b",
    r"\bnot approved\b",
    r"\bawaiting\b",
    r"\bpending\b",
    r"\bprocessing\b",
    r"\bnot credited\b",
    r"\bnot withdrawable\b",
)

_EXCLUSION_PATTERNS = (
    r"\bunapproved\b",
    r"\bnot approved\b",
    r"\bunearned\b",
    r"\bnot credited\b",
    r"\bnot received\b",
    r"\bhas not reached\b",
    r"\bnot withdrawable\b",
    r"\bfailed\b",
    r"\bcancelled\b",
    r"\bcanceled\b",
    r"\bvoid\b",
)

_INSTRUCTION_PATTERNS = (
    r"\bignore previous instructions\b",
    r"\bignore all instructions\b",
    r"\bsystem prompt\b",
    r"\bdeveloper message\b",
    r"\bdo not follow\b",
    r"\boverride\b",
    r"\bdisregard\b",
    r"\bassistant must\b",
    r"\byou must approve\b",
    r"\bapprove this\b",
    r"\bmark this as affordable\b",
)

_INCOME_TERMS = (
    "salary",
    "payroll",
    "wage",
    "income",
    "commission",
    "bonus",
    "payout",
    "stipend",
    "pension",
    "refund",
    "prize",
)

_EXPENSE_TERMS = (
    "bill",
    "rent",
    "lease",
    "payment",
    "purchase",
    "expense",
    "charge",
    "debit",
    "subscription",
)

_CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "₹": "INR",
    "R": "ZAR",
}


def _contains_pattern(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def _clean_text(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _parse_decimal(value: object) -> Optional[Decimal]:
    if value is None:
        return None

    if isinstance(value, Decimal):
        return value

    text = str(value).strip()

    if not text:
        return None

    for symbol in _CURRENCY_SYMBOLS:
        text = text.replace(symbol, "")

    text = text.replace(",", "").strip()

    try:
        amount = Decimal(text)
    except (InvalidOperation, ValueError):
        return None

    if amount < Decimal("0"):
        return None

    return amount


def _extract_amount(text: str) -> Optional[Decimal]:
    patterns = (
        r"(?:₹|€|\$|R)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"\b(?:INR|EUR|USD|ZAR|IDR)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"\b(?:amount|salary|pay|payment|refund|bonus|commission|payout)"
        r"\s*(?:of|is|was|:)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    )

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return _parse_decimal(match.group(1))

    return None


def _extract_currency(
    text: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    upper = text.upper()

    for code in ("INR", "USD", "EUR", "ZAR", "IDR"):
        if re.search(rf"\b{code}\b", upper):
            return code

    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code

    return normalize_currency(fallback)


def _extract_date(text: str) -> Optional[date]:
    patterns = (
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{4}/\d{2}/\d{2})\b",
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b(\d{2}-\d{2}-\d{4})\b",
    )

    for pattern in patterns:
        match = re.search(pattern, text)

        if not match:
            continue

        value = match.group(1)

        for fmt in (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%d-%m-%Y",
        ):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                pass

    return None


def _classify_category(text: str) -> Optional[str]:
    lowered = text.lower()

    if any(term in lowered for term in _INCOME_TERMS):
        if "commission" in lowered:
            return "commission"
        if "bonus" in lowered:
            return "bonus"
        if "refund" in lowered:
            return "refund"
        if "payout" in lowered:
            return "payout"
        if "salary" in lowered or "payroll" in lowered:
            return "salary"
        return "income"

    if any(term in lowered for term in _EXPENSE_TERMS):
        if "rent" in lowered or "lease" in lowered:
            return "rent"
        if "subscription" in lowered:
            return "subscription"
        if "bill" in lowered:
            return "bill"
        return "expense"

    return None


def _classify(
    text: str,
) -> tuple[EvidenceStrength, EvidenceEffect]:
    if _contains_pattern(text, _EXCLUSION_PATTERNS):
        return EvidenceStrength.CONFIRMED, EvidenceEffect.EXCLUDE

    if _contains_pattern(text, _CONFIRMED_PATTERNS):
        return EvidenceStrength.CONFIRMED, EvidenceEffect.INCLUDE

    if _contains_pattern(text, _UNCERTAIN_PATTERNS):
        return EvidenceStrength.POSSIBLE, EvidenceEffect.INFORMATIONAL

    return EvidenceStrength.LIKELY, EvidenceEffect.INFORMATIONAL


def _parse_message_date(value: object) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    # Dataset uses ISO timestamps such as:
    # 2025-07-29T09:30:00Z
    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass

    return _extract_date(text)


def extract_message_evidence(
    *,
    evidence_id: str,
    user_id: str,
    message: str,
    request_id: Optional[str] = None,
    event_id: Optional[str] = None,
    message_date: Optional[date] = None,
    source_type: Optional[str] = None,
) -> FinancialEvidence:
    """
    Convert one raw message into normalized FinancialEvidence.

    Message instructions are always treated as untrusted content.
    """

    text = _clean_text(message)

    if not text:
        raise ValueError("message cannot be empty")

    strength, effect = _classify(text)

    contains_instruction = _contains_pattern(
        text,
        _INSTRUCTION_PATTERNS,
    )

    instruction_ignored = contains_instruction

    if contains_instruction:
        if strength == EvidenceStrength.LIKELY:
            strength = EvidenceStrength.POSSIBLE

        if effect == EvidenceEffect.INCLUDE:
            effect = EvidenceEffect.INFORMATIONAL

    amount = _extract_amount(text)
    currency = _extract_currency(text)
    category = _classify_category(text)

    evidence_date = (
        message_date
        if message_date is not None
        else _extract_date(text)
    )

    status: Optional[str] = None
    direction: Optional[str] = None

    lowered = text.lower()

    if any(
        term in lowered
        for term in (
            "pending",
            "processing",
            "awaiting",
            "not credited",
            "not withdrawable",
        )
    ):
        status = "pending"

    elif any(
        term in lowered
        for term in (
            "confirmed",
            "approved",
            "verified",
            "guaranteed",
        )
    ):
        status = "confirmed"

    if any(
        term in lowered
        for term in (
            "salary",
            "income",
            "payout",
            "refund",
            "bonus",
            "commission",
            "prize",
            "wage",
        )
    ):
        direction = "credit"

    if any(
        term in lowered
        for term in (
            "bill",
            "rent",
            "lease",
            "payment",
            "purchase",
            "expense",
            "charge",
            "debit",
            "subscription",
        )
    ):
        direction = "debit"

    metadata = {
        "extractor": "deterministic_v2",
    }

    if source_type:
        metadata["source_type"] = source_type

    if contains_instruction:
        metadata["instruction_policy"] = (
            "ignored_as_untrusted_content"
        )

    return FinancialEvidence(
        evidence_id=evidence_id,
        user_id=user_id,
        source=EvidenceSource.MESSAGE,
        strength=strength,
        effect=effect,
        statement=text,
        event_id=event_id,
        request_id=request_id,
        evidence_date=evidence_date,
        amount=amount,
        currency=currency,
        category=category,
        status=status,
        direction=direction,
        contains_instruction=contains_instruction,
        instruction_ignored=instruction_ignored,
        metadata=metadata,
    )


def extract_many_messages(
    messages: Iterable[dict],
) -> tuple[FinancialEvidence, ...]:
    """
    Extract evidence from the actual dataset message schema.

    Supported dataset keys:

    message_id
    user_id
    request_id
    related_event_id
    sent_at
    source_type
    message_text

    Legacy aliases are also accepted for compatibility.
    """

    results: list[FinancialEvidence] = []

    for index, raw in enumerate(messages, start=1):
        if not isinstance(raw, dict):
            continue

        # Actual HackerRank dataset field.
        message = (
            raw.get("message_text")
            or raw.get("message")
            or raw.get("text")
            or raw.get("content")
        )

        if message is None:
            continue

        evidence_id = str(
            raw.get("evidence_id")
            or raw.get("message_id")
            or f"message_evidence_{index}"
        )

        user_id = str(
            raw.get("user_id") or ""
        ).strip()

        if not user_id:
            continue

        request_id_raw = raw.get("request_id")

        request_id = (
            str(request_id_raw).strip()
            if request_id_raw
            else None
        )

        # Actual dataset uses related_event_id.
        event_id_raw = (
            raw.get("related_event_id")
            or raw.get("event_id")
        )

        event_id = (
            str(event_id_raw).strip()
            if event_id_raw
            else None
        )

        message_date = _parse_message_date(
            raw.get("sent_at")
            or raw.get("message_date")
            or raw.get("date")
        )

        results.append(
            extract_message_evidence(
                evidence_id=evidence_id,
                user_id=user_id,
                message=str(message),
                request_id=request_id,
                event_id=event_id,
                message_date=message_date,
                source_type=(
                    str(raw["source_type"])
                    if raw.get("source_type") is not None
                    else None
                ),
            )
        )

    return tuple(results)
