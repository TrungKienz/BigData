from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    base_event_time: datetime = field(
        default_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    # A PaySim step represents one hour. Tests may override this for synthetic data.
    step_seconds: int = 3_600
    high_amount_transfer_threshold: float = 200_000.0
    high_amount_cash_out_threshold: float = 200_000.0
    rapid_outflow_window_seconds: int = 3_600
    rapid_outflow_count_threshold: int = 3
    rapid_outflow_amount_threshold: float = 300_000.0
    balance_tolerance: float = 1.0
    account_drain_min_balance_floor: float = 50_000.0
    account_drain_ratio_threshold: float = 0.8
    account_drain_near_zero_balance: float = 1_000.0
    fan_out_window_seconds: int = 10_800
    fan_out_distinct_receiver_threshold: int = 3
    fan_out_total_amount_threshold: float = 250_000.0
    fan_in_window_seconds: int = 10_800
    fan_in_distinct_sender_threshold: int = 3
    fan_in_total_amount_threshold: float = 250_000.0
    structuring_window_seconds: int = 21_600
    structuring_count_threshold: int = 4
    structuring_min_amount: float = 40_000.0
    structuring_max_amount: float = 90_000.0
    structuring_total_amount_threshold: float = 250_000.0
    new_counterparty_amount_threshold: float = 150_000.0
    cashout_after_inbound_window_seconds: int = 10_800
    cashout_after_inbound_ratio_threshold: float = 0.8
    account_drain_weight: float = 0.35
    sender_fan_out_weight: float = 0.25
    receiver_fan_in_weight: float = 0.25
    structured_split_weight: float = 0.30
    new_counterparty_weight: float = 0.15
    cashout_after_inbound_weight: float = 0.40
    schema_version: int = 2
    default_csv_path: Path = Path("data/raw/paysim.csv")


RULE_SCORING_CONFIG_FIELDS = (
    "account_drain_min_balance_floor",
    "account_drain_ratio_threshold",
    "account_drain_near_zero_balance",
    "fan_out_window_seconds",
    "fan_out_distinct_receiver_threshold",
    "fan_out_total_amount_threshold",
    "fan_in_window_seconds",
    "fan_in_distinct_sender_threshold",
    "fan_in_total_amount_threshold",
    "structuring_window_seconds",
    "structuring_count_threshold",
    "structuring_min_amount",
    "structuring_max_amount",
    "structuring_total_amount_threshold",
    "new_counterparty_amount_threshold",
    "cashout_after_inbound_window_seconds",
    "cashout_after_inbound_ratio_threshold",
    "account_drain_weight",
    "sender_fan_out_weight",
    "receiver_fan_in_weight",
    "structured_split_weight",
    "new_counterparty_weight",
    "cashout_after_inbound_weight",
)


def rule_scoring_config(config: PipelineConfig | None = None) -> dict[str, float | int]:
    value = config or PipelineConfig()
    return {name: getattr(value, name) for name in RULE_SCORING_CONFIG_FIELDS}
