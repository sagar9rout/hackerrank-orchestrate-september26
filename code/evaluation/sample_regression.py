#!/usr/bin/env python3

import csv
import sys
from pathlib import Path


COMPARE_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    root = Path(__file__).resolve().parents[2]

    expected_path = root / "dataset" / "sample_requests.csv"
    actual_path = root / "output.csv"

    if not expected_path.exists():
        print(f"ERROR: missing {expected_path}")
        return 1

    if not actual_path.exists():
        print(f"ERROR: missing {actual_path}")
        print("Run the prediction engine first.")
        return 1

    expected = load_csv(expected_path)
    actual = load_csv(actual_path)

    expected_map = {row["request_id"]: row for row in expected}
    actual_map = {row["request_id"]: row for row in actual}

    failures = 0

    for request_id, expected_row in expected_map.items():
        actual_row = actual_map.get(request_id)

        if actual_row is None:
            print(f"[FAIL] {request_id}: missing from output")
            failures += 1
            continue

        row_failed = False

        for column in COMPARE_COLUMNS:
            expected_value = expected_row.get(column, "")
            actual_value = actual_row.get(column, "")

            if expected_value != actual_value:
                if not row_failed:
                    print(f"\n[FAIL] {request_id}")
                    row_failed = True
                    failures += 1

                print(f"  {column}")
                print(f"    expected: {expected_value!r}")
                print(f"    actual:   {actual_value!r}")

        if not row_failed:
            print(f"[PASS] {request_id}")

    extra = set(actual_map) - set(expected_map)

    for request_id in sorted(extra):
        print(f"[WARN] {request_id}: not part of 25-row sample set")

    print()
    print("=" * 60)
    print(f"Sample requests checked: {len(expected)}")
    print(f"Failed requests:         {failures}")
    print("=" * 60)

    if failures:
        return 1

    print("ALL SAMPLE REQUESTS MATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
