#!/usr/bin/env python3

import csv
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


EXPECTED_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

VALID_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

VALID_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}


def decimal(value: str) -> Decimal:
    try:
        return Decimal(value.strip())
    except (InvalidOperation, AttributeError):
        raise ValueError(f"Invalid decimal value: {value!r}")


def validate_output(output_path: Path, requests_path: Path) -> list[str]:
    errors = []

    with requests_path.open(newline="", encoding="utf-8-sig") as f:
        requests = list(csv.DictReader(f))

    with output_path.open(newline="", encoding="utf-8-sig") as f:
        output = list(csv.DictReader(f))

    # Schema
    actual_columns = list(output[0].keys()) if output else []

    if actual_columns != EXPECTED_COLUMNS:
        errors.append(
            f"Wrong columns. Expected {EXPECTED_COLUMNS}, got {actual_columns}"
        )

    # Row count
    if len(output) != len(requests):
        errors.append(
            f"Wrong row count: expected {len(requests)}, got {len(output)}"
        )

    request_map = {r["request_id"]: r for r in requests}
    seen = set()

    for row_number, row in enumerate(output, start=2):
        request_id = row.get("request_id", "")

        if request_id in seen:
            errors.append(
                f"Row {row_number}: duplicate request_id {request_id}"
            )

        seen.add(request_id)

        if request_id not in request_map:
            errors.append(
                f"Row {row_number}: unknown request_id {request_id}"
            )
            continue

        request = request_map[request_id]

        # Amount bounds
        try:
            safe = decimal(row["amount_safe_to_pay"])
            requested = decimal(request["requested_amount"])

            if safe < 0:
                errors.append(
                    f"Row {row_number}: negative safe amount"
                )

            if safe > requested:
                errors.append(
                    f"Row {row_number}: safe amount {safe} exceeds "
                    f"requested amount {requested}"
                )

        except (ValueError, KeyError) as exc:
            errors.append(
                f"Row {row_number}: amount validation failed: {exc}"
            )

        # Vocabulary
        status = row.get("affordability_status", "")

        if status not in VALID_STATUSES:
            errors.append(
                f"Row {row_number}: invalid status {status!r}"
            )

        method = row.get("recommended_payment_method", "")

        if method not in VALID_METHODS:
            errors.append(
                f"Row {row_number}: invalid payment method {method!r}"
            )

        # Required explanation
        if not row.get("decision_explanation", "").strip():
            errors.append(
                f"Row {row_number}: missing decision explanation"
            )

        # not_recommended must have no payment plan
        if (
            method == "not_recommended"
            and row.get("payment_plan", "").strip()
        ):
            errors.append(
                f"Row {row_number}: not_recommended must have empty "
                f"payment_plan"
            )

        # Basic payment-plan syntax
        plan = row.get("payment_plan", "").strip()

        if plan:
            payments = plan.split("|")

            for payment in payments:
                if ":" not in payment:
                    errors.append(
                        f"Row {row_number}: malformed payment plan "
                        f"entry {payment!r}"
                    )
                    continue

                payment_date, amount = payment.split(":", 1)

                if not payment_date.strip():
                    errors.append(
                        f"Row {row_number}: payment plan has empty date"
                    )

                try:
                    if decimal(amount) <= 0:
                        errors.append(
                            f"Row {row_number}: payment amount must be positive"
                        )
                except ValueError:
                    errors.append(
                        f"Row {row_number}: invalid payment amount "
                        f"{amount!r}"
                    )

    # Missing IDs
    expected_ids = set(request_map)
    missing = expected_ids - seen

    for request_id in sorted(missing):
        errors.append(f"Missing request_id {request_id}")

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    output_path = repo_root / "output.csv"
    requests_path = repo_root / "dataset" / "requests.csv"

    if not output_path.exists():
        print(f"ERROR: {output_path} does not exist")
        print("Run the main prediction pipeline first.")
        return 1

    if not requests_path.exists():
        print(f"ERROR: {requests_path} does not exist")
        return 1

    errors = validate_output(output_path, requests_path)

    if errors:
        print(f"VALIDATION FAILED: {len(errors)} error(s)")

        for error in errors:
            print(f" - {error}")

        return 1

    print("VALIDATION PASSED")
    print(
        "Output schema, row count, IDs, bounds and basic plan syntax "
        "are valid."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
