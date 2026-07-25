from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class TransactionEvent:
    event_id: str
    event_time: datetime
    producer_ts: datetime
    step: int
    txn_type: str
    amount: float
    name_orig: str
    oldbalance_org: float
    newbalance_orig: float
    name_dest: str
    oldbalance_dest: float
    newbalance_dest: float
    is_fraud: int
    schema_version: int = 1
    hour_of_day: int | None = None
    is_night_transaction: int | None = None
    customer_account_age_days: float = 0.0
    browser: str = "unknown"
    device_type: str = "unknown"
    new_device_flag: int = 0
    billing_country: str = "unknown"
    ip_country: str = "unknown"
    ip_billing_distance_km: float = 0.0
    ip_billing_country_mismatch: int = 0
    shipping_billing_mismatch: int = 0
    failed_payment_attempts_24h: float = 0.0


@dataclass(frozen=True)
class AccountStateUpdate:
    event_id: str
    source_event_id: str
    account_id: str
    role: str
    step: int
    balance_before: float
    balance_after: float
    event_time: datetime


@dataclass(frozen=True)
class FraudDecision:
    event_id: str
    is_alert: bool
    risk_score: float
    severity: str
    ml_score: float = 0.0
    ml_model_version: str = "v0"
    triggered_rules: tuple[str, ...] = field(default_factory=tuple)
    rule_score: float = 0.0
    decision_threshold: float = 0.5
    model_tag: str = "unknown"
    feature_configuration: str = "unknown"
    rule_weight: float = 0.6
    ml_weight: float = 0.4


@dataclass(frozen=True)
class PredictionRecord:
    event_id: str
    account_id: str
    name_dest: str
    event_time: datetime
    txn_type: str
    amount: float
    risk_score: float
    severity: str
    rule_score: float
    ml_score: float
    hybrid_score: float
    decision_threshold: float
    ml_model_version: str
    model_tag: str = "unknown"
    feature_configuration: str = "unknown"
    rule_weight: float = 0.6
    ml_weight: float = 0.4
    triggered_rules: tuple[str, ...] = field(default_factory=tuple)
    is_alert: bool = False
    alert_id: str | None = None


@dataclass(frozen=True)
class WindowMetric:
    window_start: datetime
    window_end: datetime
    event_count: int
    fraud_count: int
    total_amount: float
    fraud_rate: float



