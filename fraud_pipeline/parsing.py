from __future__ import annotations

import csv
import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Iterator

from .config import PipelineConfig
from .models import AccountStateUpdate, TransactionEvent


class ParseError(ValueError):
    pass


def _optional(row: dict[str, str], key: str, default: str) -> str:
    value = row.get(key)
    return default if value is None or str(value).strip() == "" else str(value).strip()


def build_event_id(row: dict[str, str]) -> str:
    raw = "|".join(
        [
            row["step"].strip(),
            row["type"].strip(),
            row["amount"].strip(),
            row["nameOrig"].strip(),
            row["nameDest"].strip(),
            row["oldbalanceOrg"].strip(),
            row["oldbalanceDest"].strip(),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def parse_csv_row(row: dict[str, str], config: PipelineConfig | None = None) -> TransactionEvent:
    config = config or PipelineConfig()
    try:
        step = int(row["step"])
        amount = float(row["amount"])
        oldbalance_org = float(row["oldbalanceOrg"])
        oldbalance_dest = float(row["oldbalanceDest"])
        txn_type = row["type"].strip()
        newbalance_orig = float(
            _optional(row, "newbalanceOrig", str(max(oldbalance_org - amount, 0.0)))
        )
        newbalance_dest = float(
            _optional(row, "newbalanceDest", str(oldbalance_dest + amount))
        )
        is_fraud = int(_optional(row, "isFraud", "0"))
        name_orig = row["nameOrig"].strip()
        name_dest = row["nameDest"].strip()
        hour = int(_optional(row, "hour_of_day", str(step % 24)))
        is_night = int(
            _optional(
                row,
                "is_night_transaction",
                str(int(hour >= 22 or hour <= 6)),
            )
        )
        customer_account_age_days = float(
            _optional(row, "customer_account_age_days", "0.0")
        )
        new_device = int(_optional(row, "new_device_flag", "0"))
        ip_distance = float(_optional(row, "ip_billing_distance_km", "0.0"))
        ip_mismatch = int(_optional(row, "ip_billing_country_mismatch", "0"))
        shipping_mismatch = int(_optional(row, "shipping_billing_mismatch", "0"))
        failed_attempts = float(
            _optional(row, "failed_payment_attempts_24h", "0.0")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ParseError(f"Invalid PaySim row: {exc}") from exc

    event_time = config.base_event_time + timedelta(seconds=step * config.step_seconds)
    event_id = build_event_id(row)
    return TransactionEvent(
        event_id=event_id,
        event_time=event_time,
        producer_ts=event_time,
        step=step,
        txn_type=txn_type,
        amount=amount,
        name_orig=name_orig,
        oldbalance_org=oldbalance_org,
        newbalance_orig=newbalance_orig,
        name_dest=name_dest,
        oldbalance_dest=oldbalance_dest,
        newbalance_dest=newbalance_dest,
        is_fraud=is_fraud,
        schema_version=config.schema_version,
        hour_of_day=hour,
        is_night_transaction=is_night,
        customer_account_age_days=customer_account_age_days,
        browser=_optional(row, "browser", "unknown"),
        device_type=_optional(row, "device_type", "unknown"),
        new_device_flag=new_device,
        billing_country=_optional(row, "billing_country", "unknown"),
        ip_country=_optional(row, "ip_country", "unknown"),
        ip_billing_distance_km=ip_distance,
        ip_billing_country_mismatch=ip_mismatch,
        shipping_billing_mismatch=shipping_mismatch,
        failed_payment_attempts_24h=failed_attempts,
    )


def derive_account_state_updates(event: TransactionEvent) -> list[AccountStateUpdate]:
    return [
        AccountStateUpdate(
            event_id=f"{event.event_id}:sender",
            source_event_id=event.event_id,
            account_id=event.name_orig,
            role="sender",
            step=event.step,
            balance_before=event.oldbalance_org,
            balance_after=event.newbalance_orig,
            event_time=event.event_time,
        ),
        AccountStateUpdate(
            event_id=f"{event.event_id}:receiver",
            source_event_id=event.event_id,
            account_id=event.name_dest,
            role="receiver",
            step=event.step,
            balance_before=event.oldbalance_dest,
            balance_after=event.newbalance_dest,
            event_time=event.event_time,
        ),
    ]


def iter_transaction_events(
    csv_path: str | Path,
    config: PipelineConfig | None = None,
    limit: int | None = None,
) -> Iterator[TransactionEvent]:
    config = config or PipelineConfig()
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            yield parse_csv_row(row, config=config)



