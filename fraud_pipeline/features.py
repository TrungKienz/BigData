from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import fsum
from typing import AbstractSet, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .models import TransactionEvent


SENDER_DEBIT_TYPES = frozenset({"PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"})
RECEIVER_CREDIT_TYPES = frozenset({"TRANSFER", "CASH_IN"})
OUTBOUND_TYPES = frozenset({"TRANSFER", "CASH_OUT"})

TIME_SINCE_LAST_TRANSACTION_DEFAULT = 86_400.0


def credited_account(event: TransactionEvent) -> str | None:
    if event.txn_type == "TRANSFER":
        return event.name_dest
    if event.txn_type == "CASH_IN":
        return event.name_orig
    return None


def funding_account(event: TransactionEvent) -> str | None:
    if event.txn_type == "TRANSFER":
        return event.name_orig
    if event.txn_type == "CASH_IN":
        return event.name_dest
    return None

REQUIRED_CLEANED_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "hour_of_day",
    "is_night_transaction",
    "customer_account_age_days",
    "browser",
    "device_type",
    "new_device_flag",
    "billing_country",
    "ip_country",
    "ip_billing_distance_km",
    "ip_billing_country_mismatch",
    "shipping_billing_mismatch",
    "failed_payment_attempts_24h",
)

FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "isFraud",
        "label_is_fraud",
        "label",
        "isFlaggedFraud",
        "row_id",
        "split",
        "step",
        "nameOrig",
        "nameDest",
        "device_id",
        "event_id",
    }
)

CATEGORICAL_FEATURES = (
    "txn_type",
    "browser",
    "device_type",
    "billing_country",
    "ip_country",
)

BASE_TRANSACTION_FEATURES = (
    "txn_type",
    "amount",
    "oldbalance_org",
    "oldbalance_dest",
    "sender_depletion_ratio",
    "amount_to_balance_ratio",
    "is_same_sender_receiver",
    "dest_is_merchant",
    "hour_of_day",
    "is_night_transaction",
)

CONTEXTUAL_FEATURES = (
    "customer_account_age_days",
    "browser",
    "device_type",
    "new_device_flag",
    "billing_country",
    "ip_country",
    "ip_billing_distance_km",
    "ip_billing_country_mismatch",
    "shipping_billing_mismatch",
    "failed_payment_attempts_24h",
)

DYNAMIC_FEATURES = (
    "sender_recent_txn_count",
    "sender_recent_total_amount",
    "receiver_recent_txn_count",
    "receiver_recent_total_amount",
    "is_new_counterparty",
    "inbound_to_cashout_ratio",
    "velocity_transactions_1h",
    "time_since_last_transaction",
)

POST_TRANSACTION_FEATURES = (
    "newbalance_orig",
    "newbalance_dest",
    "sender_balance_delta",
    "receiver_balance_delta",
    "is_zero_balance_after",
    "sender_balance_inconsistent",
    "receiver_balance_inconsistent",
    "sender_balance_discrepancy",
    "receiver_balance_discrepancy",
)

DEPLOYMENT_SAFE_FEATURES = tuple(
    dict.fromkeys(BASE_TRANSACTION_FEATURES + CONTEXTUAL_FEATURES + DYNAMIC_FEATURES)
)
FULL_PAYSIM_FEATURES = tuple(
    dict.fromkeys(DEPLOYMENT_SAFE_FEATURES + POST_TRANSACTION_FEATURES)
)

FEATURE_CONFIGURATIONS = {
    "deployment_safe": DEPLOYMENT_SAFE_FEATURES,
    "full_paysim": FULL_PAYSIM_FEATURES,
}

FEATURE_GROUPS = {
    "base_transaction_features": BASE_TRANSACTION_FEATURES,
    "base_plus_synthetic_contextual_features": tuple(
        dict.fromkeys(BASE_TRANSACTION_FEATURES + CONTEXTUAL_FEATURES)
    ),
    "base_plus_dynamic_features": tuple(
        dict.fromkeys(BASE_TRANSACTION_FEATURES + DYNAMIC_FEATURES)
    ),
    "full_paysim": FULL_PAYSIM_FEATURES,
    "deployment_safe": DEPLOYMENT_SAFE_FEATURES,
}

INFERENCE_AVAILABLE_FEATURES = frozenset(DEPLOYMENT_SAFE_FEATURES)

# Compatibility aliases. They now describe raw model inputs; categorical encoding is
# owned by the fitted model preprocessor rather than hand-written one-hot columns.
FEATURE_COLUMNS = list(DEPLOYMENT_SAFE_FEATURES)
TXN_TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
BROWSER_CATEGORIES = ["chrome", "safari", "firefox", "edge"]
DEVICE_TYPE_CATEGORIES = ["desktop", "mobile", "tablet"]
COUNTRY_CATEGORIES = ["US", "VN", "SG", "PH", "TH"]


def assert_no_forbidden_features(feature_columns: Iterable[str]) -> None:
    columns = list(feature_columns)
    duplicates = sorted({name for name in columns if columns.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate selected features are not allowed: {duplicates}")
    forbidden = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(columns))
    if forbidden:
        raise ValueError(f"Forbidden/leakage-prone features selected: {forbidden}")


def validate_cleaned_schema(frame: pd.DataFrame) -> None:
    missing = [name for name in REQUIRED_CLEANED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            "Cleaned transaction dataset is missing required columns: "
            + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("Cleaned transaction dataset contains no rows")
    if frame["step"].isna().any():
        raise ValueError("Cleaned transaction dataset contains null values in 'step'")
    labels = pd.to_numeric(frame["isFraud"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        invalid = frame.loc[labels.isna() | ~labels.isin([0, 1]), "isFraud"]
        raise ValueError(
            "Column 'isFraud' must contain non-null binary 0/1 values; "
            f"found examples {invalid.head(5).tolist()}"
        )


def _numeric(
    frame: pd.DataFrame,
    column: str,
    dtype: type[np.floating] = np.float32,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        count = int(values.isna().sum())
        raise ValueError(f"Column {column!r} contains {count} non-numeric/null values")
    return values.astype(dtype)


def _binary(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric(frame, column)
    invalid = ~values.isin([0.0, 1.0])
    if invalid.any():
        raise ValueError(f"Column {column!r} must contain only 0/1 values")
    return values.astype(np.int8)


def build_static_features_frame(
    frame: pd.DataFrame,
    config: PipelineConfig | None = None,
) -> pd.DataFrame:
    """Build all static features with vectorized, label-free transformations."""

    validate_cleaned_schema(frame)
    config = config or PipelineConfig()
    result = pd.DataFrame(index=frame.index)

    txn_type = frame["type"].fillna("unknown").astype(str)
    # Keep monetary values used by business-cost evaluation in float64.
    amount = _numeric(frame, "amount", np.float64)
    old_org = _numeric(frame, "oldbalanceOrg")
    new_org = _numeric(frame, "newbalanceOrig")
    old_dest = _numeric(frame, "oldbalanceDest")
    new_dest = _numeric(frame, "newbalanceDest")

    result["txn_type"] = txn_type
    result["amount"] = amount
    result["oldbalance_org"] = old_org
    result["oldbalance_dest"] = old_dest

    result["sender_depletion_ratio"] = np.where(
        old_org > 0,
        np.minimum(amount / old_org.replace(0, np.nan), 1.0),
        (amount > 0).astype(np.float32),
    ).astype(np.float32)
    denominator = old_org + old_dest
    result["amount_to_balance_ratio"] = np.where(
        denominator > 0,
        np.minimum(amount / denominator.replace(0, np.nan), 1.0),
        (amount > 0).astype(np.float32),
    ).astype(np.float32)
    result["is_same_sender_receiver"] = (
        frame["nameOrig"].astype(str) == frame["nameDest"].astype(str)
    ).astype(np.int8)
    result["dest_is_merchant"] = frame["nameDest"].astype(str).str.startswith("M").astype(np.int8)
    result["hour_of_day"] = _numeric(frame, "hour_of_day").astype(np.int8)
    result["is_night_transaction"] = _binary(frame, "is_night_transaction")

    result["customer_account_age_days"] = _numeric(
        frame, "customer_account_age_days"
    )
    result["browser"] = frame["browser"].fillna("unknown").astype(str)
    result["device_type"] = frame["device_type"].fillna("unknown").astype(str)
    result["new_device_flag"] = _binary(frame, "new_device_flag")
    result["billing_country"] = frame["billing_country"].fillna("unknown").astype(str)
    result["ip_country"] = frame["ip_country"].fillna("unknown").astype(str)
    result["ip_billing_distance_km"] = _numeric(frame, "ip_billing_distance_km")
    result["ip_billing_country_mismatch"] = _binary(
        frame, "ip_billing_country_mismatch"
    )
    result["shipping_billing_mismatch"] = _binary(
        frame, "shipping_billing_mismatch"
    )
    result["failed_payment_attempts_24h"] = _numeric(
        frame, "failed_payment_attempts_24h"
    )

    result["newbalance_orig"] = new_org
    result["newbalance_dest"] = new_dest
    result["sender_balance_delta"] = (old_org - new_org).astype(np.float32)
    result["receiver_balance_delta"] = (new_dest - old_dest).astype(np.float32)
    result["is_zero_balance_after"] = (
        (new_org == 0) & txn_type.isin(SENDER_DEBIT_TYPES)
    ).astype(np.int8)

    sender_expected = old_org - amount
    receiver_expected = old_dest + amount
    sender_discrepancy = sender_expected - new_org
    receiver_discrepancy = receiver_expected - new_dest
    sender_applicable = txn_type.isin(SENDER_DEBIT_TYPES)
    receiver_applicable = txn_type.isin(RECEIVER_CREDIT_TYPES)
    result["sender_balance_inconsistent"] = (
        sender_applicable & (sender_discrepancy.abs() > config.balance_tolerance)
    ).astype(np.int8)
    result["receiver_balance_inconsistent"] = (
        receiver_applicable & (receiver_discrepancy.abs() > config.balance_tolerance)
    ).astype(np.int8)
    result["sender_balance_discrepancy"] = sender_discrepancy.where(
        sender_applicable, 0.0
    ).astype(np.float32)
    result["receiver_balance_discrepancy"] = receiver_discrepancy.where(
        receiver_applicable, 0.0
    ).astype(np.float32)
    return result


def _event_source_record(event: TransactionEvent) -> dict[str, object]:
    hour = event.hour_of_day if event.hour_of_day is not None else event.step % 24
    is_night = (
        event.is_night_transaction
        if event.is_night_transaction is not None
        else int(hour >= 22 or hour <= 6)
    )
    return {
        "step": event.step,
        "type": event.txn_type,
        "amount": event.amount,
        "nameOrig": event.name_orig,
        "nameDest": event.name_dest,
        "oldbalanceOrg": event.oldbalance_org,
        "newbalanceOrig": event.newbalance_orig,
        "oldbalanceDest": event.oldbalance_dest,
        "newbalanceDest": event.newbalance_dest,
        "isFraud": event.is_fraud,
        "hour_of_day": hour,
        "is_night_transaction": is_night,
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
    }


def build_feature_record(
    event: TransactionEvent,
    config: PipelineConfig | None = None,
    dynamic_features: Mapping[str, float] | None = None,
) -> dict[str, float | int | str]:
    """Build the same raw feature record consumed by offline and runtime inference."""

    static = build_static_features_frame(
        pd.DataFrame([_event_source_record(event)]), config=config
    ).iloc[0]
    record: dict[str, float | int | str] = static.to_dict()
    values = dynamic_features or {}
    for name in DYNAMIC_FEATURES:
        default = (
            TIME_SINCE_LAST_TRANSACTION_DEFAULT
            if name == "time_since_last_transaction"
            else 0.0
        )
        record[name] = float(values.get(name, default))
    return record


@dataclass(frozen=True)
class FeatureContext:
    recent_sender_events: Sequence[TransactionEvent] = ()
    recent_receiver_events: Sequence[TransactionEvent] = ()
    recent_inbound_events: Sequence[TransactionEvent] = ()
    known_counterparties: AbstractSet[str] = field(default_factory=frozenset)
    last_sender_timestamp: float | None = None


def _strict_prior_window(
    events: Sequence[TransactionEvent],
    current_step: int,
    current_timestamp: float,
    window_seconds: int,
) -> list[TransactionEvent]:
    lower_bound = current_timestamp - window_seconds
    return [
        item
        for item in events
        if item.step < current_step
        and lower_bound <= item.event_time.timestamp() < current_timestamp
    ]


def calculate_dynamic_features(
    event: TransactionEvent,
    context: FeatureContext | None = None,
    config: PipelineConfig | None = None,
) -> dict[str, float]:
    """Calculate prior-only features; same-timestamp events are always excluded."""

    context = context or FeatureContext()
    config = config or PipelineConfig()
    current_timestamp = event.event_time.timestamp()

    sender_window: list[TransactionEvent] = []
    if event.txn_type in OUTBOUND_TYPES:
        sender_window = [
            item
            for item in _strict_prior_window(
                context.recent_sender_events,
                event.step,
                current_timestamp,
                config.fan_out_window_seconds,
            )
            if item.name_orig == event.name_orig and item.txn_type in OUTBOUND_TYPES
        ]

    receiver_window: list[TransactionEvent] = []
    current_credit_account = credited_account(event)
    if current_credit_account is not None:
        receiver_window = [
            item
            for item in _strict_prior_window(
                context.recent_receiver_events,
                event.step,
                current_timestamp,
                config.fan_in_window_seconds,
            )
            if credited_account(item) == current_credit_account
        ]

    inbound_ratio = 0.0
    if event.txn_type == "CASH_OUT":
        matching_inbound = [
            item
            for item in _strict_prior_window(
                context.recent_inbound_events,
                event.step,
                current_timestamp,
                config.cashout_after_inbound_window_seconds,
            )
            if item.txn_type in RECEIVER_CREDIT_TYPES
            and credited_account(item) == event.name_orig
        ]
        inbound_total = fsum(item.amount for item in matching_inbound)
        if inbound_total > 0:
            inbound_ratio = event.amount / inbound_total

    sender_hour = [
        item
        for item in _strict_prior_window(
            context.recent_sender_events, event.step, current_timestamp, 3_600
        )
        if item.name_orig == event.name_orig
    ]

    last_timestamp = context.last_sender_timestamp
    if last_timestamp is None:
        prior_timestamps = [
            item.event_time.timestamp()
            for item in context.recent_sender_events
            if item.name_orig == event.name_orig
            and item.step < event.step
            and item.event_time.timestamp() < current_timestamp
        ]
        last_timestamp = max(prior_timestamps) if prior_timestamps else None
    time_since_last = (
        current_timestamp - last_timestamp
        if last_timestamp is not None and last_timestamp < current_timestamp
        else TIME_SINCE_LAST_TRANSACTION_DEFAULT
    )

    values = {
        "sender_recent_txn_count": len(sender_window),
        "sender_recent_total_amount": fsum(item.amount for item in sender_window),
        "receiver_recent_txn_count": len(receiver_window),
        "receiver_recent_total_amount": fsum(item.amount for item in receiver_window),
        "is_new_counterparty": (
            event.txn_type == "TRANSFER"
            and event.name_dest not in context.known_counterparties
        ),
        "inbound_to_cashout_ratio": inbound_ratio,
        "velocity_transactions_1h": len(sender_hour),
        "time_since_last_transaction": time_since_last,
    }
    # Offline Parquet and online event scoring use the same canonical precision.
    return {name: float(np.float32(value)) for name, value in values.items()}


class DynamicFeatureState:
    """Compact chronological state with an explicit update barrier per PaySim step."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.sender_history: dict[str, deque[TransactionEvent]] = defaultdict(deque)
        self.receiver_history: dict[str, deque[TransactionEvent]] = defaultdict(deque)
        self.inbound_history: dict[str, deque[TransactionEvent]] = defaultdict(deque)
        self.counterparties: dict[str, set[str]] = defaultdict(set)
        self.last_sender_timestamp: dict[str, float] = {}

    @staticmethod
    def _prune(history: deque[TransactionEvent], lower_bound: float) -> None:
        while history and history[0].event_time.timestamp() < lower_bound:
            history.popleft()

    def context_for(self, event: TransactionEvent) -> FeatureContext:
        timestamp = event.event_time.timestamp()
        sender_horizon = max(
            3_600,
            self.config.fan_out_window_seconds,
            self.config.structuring_window_seconds,
            self.config.rapid_outflow_window_seconds,
        )
        sender = self.sender_history[event.name_orig]
        credit_account = credited_account(event)
        receiver = (
            self.receiver_history[credit_account]
            if credit_account is not None
            else deque()
        )
        inbound = self.inbound_history[event.name_orig]
        self._prune(sender, timestamp - sender_horizon)
        self._prune(receiver, timestamp - self.config.fan_in_window_seconds)
        self._prune(
            inbound, timestamp - self.config.cashout_after_inbound_window_seconds
        )
        return FeatureContext(
            recent_sender_events=tuple(sender),
            recent_receiver_events=tuple(receiver),
            recent_inbound_events=tuple(inbound),
            known_counterparties=self.counterparties[event.name_orig],
            last_sender_timestamp=self.last_sender_timestamp.get(event.name_orig),
        )

    def calculate_step(
        self, events: Sequence[TransactionEvent]
    ) -> tuple[list[FeatureContext], list[dict[str, float]]]:
        if not events:
            return [], []
        steps = {event.step for event in events}
        if len(steps) != 1:
            raise ValueError("calculate_step requires events from exactly one step")
        contexts = [self.context_for(event) for event in events]
        features = [
            calculate_dynamic_features(event, context, self.config)
            for event, context in zip(events, contexts)
        ]
        return contexts, features

    def update_after_step(self, events: Sequence[TransactionEvent]) -> None:
        if not events:
            return
        steps = {event.step for event in events}
        if len(steps) != 1:
            raise ValueError("update_after_step requires events from exactly one step")
        for event in events:
            timestamp = event.event_time.timestamp()
            self.sender_history[event.name_orig].append(event)
            credit_account = credited_account(event)
            if credit_account is not None:
                self.receiver_history[credit_account].append(event)
                self.inbound_history[credit_account].append(event)
            self.counterparties[event.name_orig].add(event.name_dest)
            previous = self.last_sender_timestamp.get(event.name_orig)
            if previous is None or timestamp > previous:
                self.last_sender_timestamp[event.name_orig] = timestamp

    def process_step(self, events: Sequence[TransactionEvent]) -> list[dict[str, float]]:
        _, features = self.calculate_step(events)
        self.update_after_step(events)
        return features
