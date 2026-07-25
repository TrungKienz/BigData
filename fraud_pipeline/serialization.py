from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from typing import Any

from .config import PipelineConfig
from .models import AccountStateUpdate, FraudDecision, PredictionRecord, TransactionEvent, WindowMetric


def _dt(value: datetime) -> str:
    return value.isoformat()


def transaction_to_dict(event: TransactionEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_time": _dt(event.event_time),
        "producer_ts": _dt(event.producer_ts),
        "step": event.step,
        "type": event.txn_type,
        "amount": event.amount,
        "nameOrig": event.name_orig,
        "nameDest": event.name_dest,
        "hour_of_day": event.hour_of_day,
        "is_night_transaction": event.is_night_transaction,
        "customer_account_age_days": event.customer_account_age_days,
        "browser": event.browser,
        "device_type": event.device_type,
        "new_device_flag": event.new_device_flag,
        "billing_country": event.billing_country,
        "ip_country": event.ip_country,
        "ip_billing_distance_km": event.ip_billing_distance_km,
        "ip_billing_country_mismatch": event.ip_billing_country_mismatch,
        "shipping_billing_mismatch": event.shipping_billing_mismatch,
        "failed_payment_attempts_24h": event.failed_payment_attempts_24h,
        "isFraud": event.is_fraud,
        "schema_version": event.schema_version,
    }


def account_state_to_dict(update: AccountStateUpdate) -> dict[str, Any]:
    return {
        "event_id": update.event_id,
        "source_event_id": update.source_event_id,
        "account_id": update.account_id,
        "role": update.role,
        "step": update.step,
        "balance_before": update.balance_before,
        "balance_after": update.balance_after,
        "event_time": _dt(update.event_time),
    }


def sender_state_to_dict(update: AccountStateUpdate) -> dict[str, Any]:
    if update.role != "sender":
        raise ValueError("sender_state_to_dict chi nhan sender update")
    return {
        "event_id": update.event_id,
        "source_event_id": update.source_event_id,
        "event_time": _dt(update.event_time),
        "step": update.step,
        "nameOrig": update.account_id,
        "oldbalanceOrg": update.balance_before,
        "newbalanceOrig": update.balance_after,
    }


def receiver_state_to_dict(update: AccountStateUpdate) -> dict[str, Any]:
    if update.role != "receiver":
        raise ValueError("receiver_state_to_dict chi nhan receiver update")
    return {
        "event_id": update.event_id,
        "source_event_id": update.source_event_id,
        "event_time": _dt(update.event_time),
        "step": update.step,
        "nameDest": update.account_id,
        "oldbalanceDest": update.balance_before,
        "newbalanceDest": update.balance_after,
    }


def fraud_decision_to_dict(event: TransactionEvent, decision: FraudDecision) -> dict[str, Any]:
    return {
        "alert_id": f"alert:{decision.event_id}",
        "event_id": decision.event_id,
        "account_id": event.name_orig,
        "nameDest": event.name_dest,
        "event_time": _dt(event.event_time),
        "txn_type": event.txn_type,
        "amount": event.amount,
        "risk_score": decision.risk_score,
        "rule_score": decision.rule_score,
        "ml_score": decision.ml_score,
        "hybrid_score": decision.risk_score,
        "threshold": decision.decision_threshold,
        "decision": "alert" if decision.is_alert else "allow",
        "severity": decision.severity,
        "ml_model_version": decision.ml_model_version,
        "model_version": decision.ml_model_version,
        "model_tag": decision.model_tag,
        "feature_configuration": decision.feature_configuration,
        "rule_weight": decision.rule_weight,
        "ml_weight": decision.ml_weight,
        "triggered_rules": list(decision.triggered_rules),
        "is_alert": decision.is_alert,
    }


def prediction_record_from_decision(event: TransactionEvent, decision: FraudDecision) -> PredictionRecord:
    return PredictionRecord(
        event_id=decision.event_id,
        account_id=event.name_orig,
        name_dest=event.name_dest,
        event_time=event.event_time,
        txn_type=event.txn_type,
        amount=event.amount,
        risk_score=decision.risk_score,
        severity=decision.severity,
        rule_score=decision.rule_score,
        ml_score=decision.ml_score,
        hybrid_score=decision.risk_score,
        decision_threshold=decision.decision_threshold,
        ml_model_version=decision.ml_model_version,
        model_tag=decision.model_tag,
        feature_configuration=decision.feature_configuration,
        rule_weight=decision.rule_weight,
        ml_weight=decision.ml_weight,
        triggered_rules=decision.triggered_rules,
        is_alert=decision.is_alert,
        alert_id=f"alert:{decision.event_id}" if decision.is_alert else None,
    )


def prediction_record_to_dict(record: PredictionRecord) -> dict[str, Any]:
    return {
        "event_id": record.event_id,
        "account_id": record.account_id,
        "nameDest": record.name_dest,
        "event_time": _dt(record.event_time),
        "txn_type": record.txn_type,
        "amount": record.amount,
        "risk_score": record.risk_score,
        "rule_score": record.rule_score,
        "ml_score": record.ml_score,
        "hybrid_score": record.hybrid_score,
        "threshold": record.decision_threshold,
        "decision": "alert" if record.is_alert else "allow",
        "severity": record.severity,
        "ml_model_version": record.ml_model_version,
        "model_version": record.ml_model_version,
        "model_tag": record.model_tag,
        "feature_configuration": record.feature_configuration,
        "rule_weight": record.rule_weight,
        "ml_weight": record.ml_weight,
        "triggered_rules": list(record.triggered_rules),
        "is_alert": record.is_alert,
        "alert_id": record.alert_id,
    }


def window_metric_to_dict(metric: WindowMetric, window_type: str) -> dict[str, Any]:
    return {
        "window_type": window_type,
        "window_start": _dt(metric.window_start),
        "window_end": _dt(metric.window_end),
        "event_count": metric.event_count,
        "fraud_count": metric.fraud_count,
        "total_amount": metric.total_amount,
        "fraud_rate": metric.fraud_rate,
    }


def risk_rule_event() -> list[dict[str, Any]]:
    config = PipelineConfig()
    return [
        {
            "rule_id": "account_drain_near_zero_v1",
            "rule_type": "account_drain_threshold",
            "min_balance_floor": config.account_drain_min_balance_floor,
            "ratio_threshold": config.account_drain_ratio_threshold,
            "near_zero_balance": config.account_drain_near_zero_balance,
            "weight": config.account_drain_weight,
            "severity": "high",
        },
        {
            "rule_id": "sender_fan_out_burst_v1",
            "rule_type": "fan_out_threshold",
            "window_seconds": config.fan_out_window_seconds,
            "distinct_receiver_threshold": config.fan_out_distinct_receiver_threshold,
            "total_amount_threshold": config.fan_out_total_amount_threshold,
            "weight": config.sender_fan_out_weight,
            "severity": "medium",
        },
        {
            "rule_id": "receiver_fan_in_burst_v1",
            "rule_type": "fan_in_threshold",
            "window_seconds": config.fan_in_window_seconds,
            "distinct_sender_threshold": config.fan_in_distinct_sender_threshold,
            "total_amount_threshold": config.fan_in_total_amount_threshold,
            "weight": config.receiver_fan_in_weight,
            "severity": "medium",
        },
        {
            "rule_id": "structured_split_transfer_v1",
            "rule_type": "structuring_threshold",
            "window_seconds": config.structuring_window_seconds,
            "count_threshold": config.structuring_count_threshold,
            "min_amount": config.structuring_min_amount,
            "max_amount": config.structuring_max_amount,
            "total_amount_threshold": config.structuring_total_amount_threshold,
            "weight": config.structured_split_weight,
            "severity": "high",
        },
        {
            "rule_id": "new_counterparty_large_transfer_v1",
            "rule_type": "new_counterparty_threshold",
            "amount_threshold": config.new_counterparty_amount_threshold,
            "weight": config.new_counterparty_weight,
            "severity": "medium",
        },
        {
            "rule_id": "cashout_after_inbound_chain_v1",
            "rule_type": "cashout_after_inbound_threshold",
            "window_seconds": config.cashout_after_inbound_window_seconds,
            "ratio_threshold": config.cashout_after_inbound_ratio_threshold,
            "weight": config.cashout_after_inbound_weight,
            "severity": "medium",
        },
    ]


def dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


