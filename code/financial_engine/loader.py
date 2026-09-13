from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


def parse_decimal(value: str | None) -> Decimal | None:
    if value is None or value.strip() == "":
        return None
    return Decimal(value.strip())


def parse_date(value: str | None) -> date | None:
    if value is None or value.strip() == "":
        return None
    text = value.strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def parse_datetime(value: str | None) -> datetime | None:
    if value is None or value.strip() == "":
        return None

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def read_csv(name: str) -> list[dict[str, str]]:
    path = DATASET_DIR / name

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class Dataset:
    requests: list[dict[str, str]]
    profiles: list[dict[str, str]]
    events: list[dict[str, str]]
    payment_options: list[dict[str, str]]
    messages: list[dict[str, str]]
    images: list[dict[str, str]]
    exchange_rates: list[dict[str, str]]
    samples: list[dict[str, str]]


def load_dataset(dataset_dir: Path | None = None) -> Dataset:
    global DATASET_DIR

    if dataset_dir is not None:
        base = dataset_dir

        def load(name: str) -> list[dict[str, str]]:
            with (base / name).open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                return list(csv.DictReader(handle))

    else:
        load = read_csv

    return Dataset(
        requests=load("requests.csv"),
        profiles=load("financial_profiles.csv"),
        events=load("financial_events.csv"),
        payment_options=load("request_payment_options.csv"),
        messages=load("messages.csv"),
        images=load("images.csv"),
        exchange_rates=load("exchange_rates.csv"),
        samples=load("sample_requests.csv"),
    )


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        value = row.get(key)
        if value:
            result.setdefault(value, []).append(row)

    return result
