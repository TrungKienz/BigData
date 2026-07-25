from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from .models import TransactionEvent


class IntegrationError(ValueError):
    pass


def _value(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise IntegrationError(f"Thieu truong bat buoc: {key}")
    return payload[key]


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    raise IntegrationError(f"Gia tri thoi gian khong hop le: {value!r}")


def integrate_logical_streams(
    transaction_payload: Mapping[str, Any],
    sender_payload: Mapping[str, Any],
    receiver_payload: Mapping[str, Any],
) -> dict[str, Any]:
    tx_event_id = str(_value(transaction_payload, "event_id"))
    sender_source_event_id = str(_value(sender_payload, "source_event_id"))
    receiver_source_event_id = str(_value(receiver_payload, "source_event_id"))

    if sender_source_event_id != tx_event_id:
        raise IntegrationError("Sender stream khong khop source_event_id voi transaction stream")
    if receiver_source_event_id != tx_event_id:
        raise IntegrationError("Receiver stream khong khop source_event_id voi transaction stream")

    tx_step = int(_value(transaction_payload, "step"))
    sender_step = int(_value(sender_payload, "step"))
    receiver_step = int(_value(receiver_payload, "step"))
    if tx_step != sender_step or tx_step != receiver_step:
        raise IntegrationError("Step giua 3 logical streams khong dong nhat")

    tx_name_orig = str(_value(transaction_payload, "nameOrig"))
    tx_name_dest = str(_value(transaction_payload, "nameDest"))
    sender_name_orig = str(_value(sender_payload, "nameOrig"))
    receiver_name_dest = str(_value(receiver_payload, "nameDest"))
    if tx_name_orig != sender_name_orig:
        raise IntegrationError("nameOrig giua transaction va sender_state khong khop")
    if tx_name_dest != receiver_name_dest:
        raise IntegrationError("nameDest giua transaction va receiver_state khong khop")

    tx_event_time = _parse_datetime(_value(transaction_payload, "event_time"))
    sender_event_time = _parse_datetime(_value(sender_payload, "event_time"))
    receiver_event_time = _parse_datetime(_value(receiver_payload, "event_time"))
    if tx_event_time != sender_event_time or tx_event_time != receiver_event_time:
        raise IntegrationError("event_time giua 3 logical streams khong khop")

    producer_ts = _parse_datetime(transaction_payload.get("producer_ts", tx_event_time))
    return {
        "event_id": tx_event_id,
        "event_time": tx_event_time,
        "producer_ts": producer_ts,
        "step": tx_step,
        "type": str(_value(transaction_payload, "type")),
        "amount": float(_value(transaction_payload, "amount")),
        "nameOrig": tx_name_orig,
        "nameDest": tx_name_dest,
        "oldbalanceOrg": float(_value(sender_payload, "oldbalanceOrg")),
        "newbalanceOrig": float(sender_payload.get("newbalanceOrig") if sender_payload.get("newbalanceOrig") is not None else max(float(_value(sender_payload, "oldbalanceOrg")) - float(_value(transaction_payload, "amount")), 0.0)),
        "oldbalanceDest": float(_value(receiver_payload, "oldbalanceDest")),
        "newbalanceDest": float(receiver_payload.get("newbalanceDest") if receiver_payload.get("newbalanceDest") is not None else float(_value(receiver_payload, "oldbalanceDest")) + float(_value(transaction_payload, "amount"))),
        "hour_of_day": transaction_payload.get("hour_of_day"),
        "is_night_transaction": transaction_payload.get("is_night_transaction"),
        "customer_account_age_days": transaction_payload.get("customer_account_age_days", 0.0),
        "browser": transaction_payload.get("browser", "unknown"),
        "device_type": transaction_payload.get("device_type", "unknown"),
        "new_device_flag": transaction_payload.get("new_device_flag", 0),
        "billing_country": transaction_payload.get("billing_country", "unknown"),
        "ip_country": transaction_payload.get("ip_country", "unknown"),
        "ip_billing_distance_km": transaction_payload.get("ip_billing_distance_km", 0.0),
        "ip_billing_country_mismatch": transaction_payload.get("ip_billing_country_mismatch", 0),
        "shipping_billing_mismatch": transaction_payload.get("shipping_billing_mismatch", 0),
        "failed_payment_attempts_24h": transaction_payload.get("failed_payment_attempts_24h", 0.0),
        "isFraud": int(transaction_payload.get("isFraud") or 0),
        "schema_version": int(transaction_payload.get("schema_version", 1)),
    }


def integrated_payload_to_transaction_event(payload: Mapping[str, Any]) -> TransactionEvent:
    return TransactionEvent(
        event_id=str(_value(payload, "event_id")),
        event_time=_parse_datetime(_value(payload, "event_time")),
        producer_ts=_parse_datetime(payload.get("producer_ts", _value(payload, "event_time"))),
        step=int(_value(payload, "step")),
        txn_type=str(_value(payload, "type")),
        amount=float(_value(payload, "amount")),
        name_orig=str(_value(payload, "nameOrig")),
        oldbalance_org=float(_value(payload, "oldbalanceOrg")),
        newbalance_orig=float(payload.get("newbalanceOrig") if payload.get("newbalanceOrig") is not None else max(float(_value(payload, "oldbalanceOrg")) - float(_value(payload, "amount")), 0.0)),
        name_dest=str(_value(payload, "nameDest")),
        oldbalance_dest=float(_value(payload, "oldbalanceDest")),
        newbalance_dest=float(payload.get("newbalanceDest") if payload.get("newbalanceDest") is not None else float(_value(payload, "oldbalanceDest")) + float(_value(payload, "amount"))),
        is_fraud=int(payload.get("isFraud") or 0),
        schema_version=int(payload.get("schema_version", 1)),
        hour_of_day=(
            int(payload["hour_of_day"])
            if payload.get("hour_of_day") is not None
            else int(_value(payload, "step")) % 24
        ),
        is_night_transaction=(
            int(payload["is_night_transaction"])
            if payload.get("is_night_transaction") is not None
            else None
        ),
        customer_account_age_days=float(payload.get("customer_account_age_days") or 0.0),
        browser=str(payload.get("browser") or "unknown"),
        device_type=str(payload.get("device_type") or "unknown"),
        new_device_flag=int(payload.get("new_device_flag") or 0),
        billing_country=str(payload.get("billing_country") or "unknown"),
        ip_country=str(payload.get("ip_country") or "unknown"),
        ip_billing_distance_km=float(payload.get("ip_billing_distance_km") or 0.0),
        ip_billing_country_mismatch=int(payload.get("ip_billing_country_mismatch") or 0),
        shipping_billing_mismatch=int(payload.get("shipping_billing_mismatch") or 0),
        failed_payment_attempts_24h=float(payload.get("failed_payment_attempts_24h") or 0.0),
    )

