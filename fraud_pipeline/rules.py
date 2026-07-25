from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
from math import fsum, prod
from typing import AbstractSet, Any, Sequence

import numpy as np

from .config import PipelineConfig
from .features import (
    FeatureContext,
    OUTBOUND_TYPES,
    RECEIVER_CREDIT_TYPES,
    calculate_dynamic_features,
    credited_account,
    funding_account,
)
from .models import FraudDecision, TransactionEvent


LOGGER = logging.getLogger(__name__)
DEFAULT_RULE_WEIGHT = 0.6
DEFAULT_ML_WEIGHT = 0.4

def _model_runtime() -> Any | None:
    try:
        return importlib.import_module("model.model_utils")
    except ImportError:
        LOGGER.exception("ML runtime unavailable: failed to import model utilities")
        return None


def combine_risk_scores(
    rule_score: float,
    ml_score: float,
    rule_weight: float = DEFAULT_RULE_WEIGHT,
    ml_weight: float = DEFAULT_ML_WEIGHT,
) -> float:
    if rule_weight < 0 or ml_weight < 0:
        raise ValueError("Hybrid score weights must be non-negative")
    total_weight = rule_weight + ml_weight
    if total_weight <= 0:
        raise ValueError("At least one hybrid score weight must be positive")
    normalized_rule_weight = rule_weight / total_weight
    normalized_ml_weight = ml_weight / total_weight
    score = normalized_rule_weight * float(rule_score) + normalized_ml_weight * float(ml_score)
    return float(min(max(score, 0.0), 1.0))


def combine_risk_score_arrays(
    rule_scores: np.ndarray,
    ml_scores: np.ndarray,
    rule_weight: float = DEFAULT_RULE_WEIGHT,
    ml_weight: float = DEFAULT_ML_WEIGHT,
) -> np.ndarray:
    if rule_weight < 0 or ml_weight < 0:
        raise ValueError("Hybrid score weights must be non-negative")
    total_weight = rule_weight + ml_weight
    if total_weight <= 0:
        raise ValueError("At least one hybrid score weight must be positive")
    rule_values = np.asarray(rule_scores, dtype=np.float64)
    ml_values = np.asarray(ml_scores, dtype=np.float64)
    if rule_values.shape != ml_values.shape:
        raise ValueError("Rule and ML score arrays must have the same shape")
    return np.clip(
        (rule_weight / total_weight) * rule_values
        + (ml_weight / total_weight) * ml_values,
        0.0,
        1.0,
    )


@dataclass(frozen=True)
class RuleEvaluation:
    score: float
    triggered_rules: tuple[str, ...]


@dataclass
class RuleEngine:
    config: PipelineConfig

    def evaluate_rules(
        self,
        event: TransactionEvent,
        context: FeatureContext | None = None,
    ) -> RuleEvaluation:
        context = context or FeatureContext()
        triggered_rules: list[str] = []
        weights: list[float] = []

        if self._is_account_drain_near_zero(event):
            triggered_rules.append("account_drain_near_zero")
            weights.append(self.config.account_drain_weight)
        if self._has_sender_fan_out_burst(event, context.recent_sender_events):
            triggered_rules.append("sender_fan_out_burst")
            weights.append(self.config.sender_fan_out_weight)
        if self._has_receiver_fan_in_burst(event, context.recent_receiver_events):
            triggered_rules.append("receiver_fan_in_burst")
            weights.append(self.config.receiver_fan_in_weight)
        if self._has_structured_split_transfer(event, context.recent_sender_events):
            triggered_rules.append("structured_split_transfer")
            weights.append(self.config.structured_split_weight)
        if self._is_new_counterparty_large_transfer(
            event, context.known_counterparties
        ):
            triggered_rules.append("new_counterparty_large_transfer")
            weights.append(self.config.new_counterparty_weight)
        if self._has_cashout_after_inbound_chain(
            event, context.recent_inbound_events
        ):
            triggered_rules.append("cashout_after_inbound_chain")
            weights.append(self.config.cashout_after_inbound_weight)

        for weight in weights:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Rule weight must be in [0, 1], got {weight}")
        score = 1.0 - prod(1.0 - weight for weight in weights) if weights else 0.0
        return RuleEvaluation(float(score), tuple(triggered_rules))

    def evaluate(
        self,
        event: TransactionEvent,
        recent_sender_events: list[TransactionEvent] | None = None,
        recent_receiver_events: list[TransactionEvent] | None = None,
        recent_inbound_events: list[TransactionEvent] | None = None,
        known_counterparties: set[str] | None = None,
        watchlisted_accounts: set[str] | None = None,
        context: FeatureContext | None = None,
    ) -> FraudDecision:
        del watchlisted_accounts  # Reserved for a separately versioned rule.
        if context is None:
            sender_events = recent_sender_events or []
            last_timestamps = [
                item.event_time.timestamp()
                for item in sender_events
                if item.name_orig == event.name_orig
                and item.step < event.step
                and item.event_time.timestamp() < event.event_time.timestamp()
            ]
            context = FeatureContext(
                recent_sender_events=sender_events,
                recent_receiver_events=recent_receiver_events or [],
                recent_inbound_events=recent_inbound_events or [],
                known_counterparties=known_counterparties or set(),
                last_sender_timestamp=max(last_timestamps) if last_timestamps else None,
            )

        rule_evaluation = self.evaluate_rules(event, context)
        dynamic_features = calculate_dynamic_features(event, context, self.config)
        runtime = _model_runtime()
        if runtime is None:
            raise RuntimeError("ML runtime unavailable: model.model_utils could not be imported")
        scoring = runtime.get_scoring_config(strict=True)
        model_info = runtime.get_model_info(strict=True)
        ml_score = self._predict_ml_score(event, dynamic_features, runtime)
        combined_risk_score = combine_risk_scores(
            rule_evaluation.score,
            ml_score,
            float(scoring["rule_weight"]),
            float(scoring["ml_weight"]),
        )
        threshold = float(scoring["hybrid_threshold"])
        is_alert = combined_risk_score >= threshold
        severity = (
            "high"
            if combined_risk_score >= 0.65
            else "medium"
            if combined_risk_score >= 0.35
            else "low"
        )

        return FraudDecision(
            event_id=event.event_id,
            is_alert=is_alert,
            risk_score=combined_risk_score,
            severity=severity,
            ml_score=ml_score,
            ml_model_version=str(model_info["model_version"]),
            triggered_rules=rule_evaluation.triggered_rules,
            rule_score=rule_evaluation.score,
            decision_threshold=threshold,
            model_tag=str(model_info["model_tag"]),
            feature_configuration=str(model_info["feature_configuration"]),
            rule_weight=float(scoring["rule_weight"]),
            ml_weight=float(scoring["ml_weight"]),
        )

    def _predict_ml_score(
        self,
        event: TransactionEvent,
        dynamic_features: dict[str, float] | None = None,
        runtime: Any | None = None,
    ) -> float:
        if runtime is None:
            raise RuntimeError("ML runtime unavailable")
        score = float(runtime.predict_proba(event, dynamic_features=dynamic_features))
        if not np.isfinite(score):
            raise ValueError(f"Model returned a non-finite score: {score}")
        return score

    @staticmethod
    def _strict_prior(
        event: TransactionEvent,
        history: Sequence[TransactionEvent],
        window_seconds: int,
    ) -> list[TransactionEvent]:
        timestamp = event.event_time.timestamp()
        lower_bound = timestamp - window_seconds
        return [
            item
            for item in history
            if item.step < event.step
            and lower_bound <= item.event_time.timestamp() < timestamp
        ]

    def _is_account_drain_near_zero(self, event: TransactionEvent) -> bool:
        if event.txn_type not in OUTBOUND_TYPES:
            return False
        if event.oldbalance_org < self.config.account_drain_min_balance_floor:
            return False
        if event.oldbalance_org <= 0:
            return False
        drain_ratio = event.amount / event.oldbalance_org
        projected_balance = max(event.oldbalance_org - event.amount, 0.0)
        return (
            drain_ratio >= self.config.account_drain_ratio_threshold
            and projected_balance <= self.config.account_drain_near_zero_balance
        )

    def _matching_sender_window(
        self,
        event: TransactionEvent,
        recent_sender_events: Sequence[TransactionEvent],
    ) -> list[TransactionEvent]:
        if event.txn_type not in OUTBOUND_TYPES:
            return []
        return [
            item
            for item in self._strict_prior(
                event, recent_sender_events, self.config.fan_out_window_seconds
            )
            if item.name_orig == event.name_orig and item.txn_type in OUTBOUND_TYPES
        ]

    def _has_sender_fan_out_burst(
        self,
        event: TransactionEvent,
        recent_sender_events: Sequence[TransactionEvent],
    ) -> bool:
        matching = self._matching_sender_window(event, recent_sender_events)
        if event.txn_type not in OUTBOUND_TYPES:
            return False
        distinct_receivers = {item.name_dest for item in matching}
        distinct_receivers.add(event.name_dest)
        total_amount = fsum(item.amount for item in matching) + event.amount
        return (
            len(distinct_receivers)
            >= self.config.fan_out_distinct_receiver_threshold
            and total_amount >= self.config.fan_out_total_amount_threshold
        )

    def _has_receiver_fan_in_burst(
        self,
        event: TransactionEvent,
        recent_receiver_events: Sequence[TransactionEvent],
    ) -> bool:
        current_credit_account = credited_account(event)
        if current_credit_account is None:
            return False
        matching = [
            item
            for item in self._strict_prior(
                event, recent_receiver_events, self.config.fan_in_window_seconds
            )
            if credited_account(item) == current_credit_account
        ]
        distinct_senders = {
            account
            for item in matching
            if (account := funding_account(item)) is not None
        }
        current_funding_account = funding_account(event)
        if current_funding_account is not None:
            distinct_senders.add(current_funding_account)
        total_amount = fsum(item.amount for item in matching) + event.amount
        return (
            len(distinct_senders) >= self.config.fan_in_distinct_sender_threshold
            and total_amount >= self.config.fan_in_total_amount_threshold
        )

    def _has_structured_split_transfer(
        self,
        event: TransactionEvent,
        recent_sender_events: Sequence[TransactionEvent],
    ) -> bool:
        if event.txn_type != "TRANSFER":
            return False
        if not (
            self.config.structuring_min_amount
            <= event.amount
            <= self.config.structuring_max_amount
        ):
            return False
        matching = [
            item
            for item in self._strict_prior(
                event, recent_sender_events, self.config.structuring_window_seconds
            )
            if item.name_orig == event.name_orig
            and item.txn_type == "TRANSFER"
            and self.config.structuring_min_amount
            <= item.amount
            <= self.config.structuring_max_amount
        ]
        total_amount = fsum(item.amount for item in matching) + event.amount
        return (
            len(matching) + 1 >= self.config.structuring_count_threshold
            and total_amount >= self.config.structuring_total_amount_threshold
        )

    def _is_new_counterparty_large_transfer(
        self, event: TransactionEvent, known_counterparties: AbstractSet[str]
    ) -> bool:
        return (
            event.txn_type == "TRANSFER"
            and event.amount >= self.config.new_counterparty_amount_threshold
            and event.name_dest not in known_counterparties
        )

    def _has_cashout_after_inbound_chain(
        self,
        event: TransactionEvent,
        recent_inbound_events: Sequence[TransactionEvent],
    ) -> bool:
        if event.txn_type != "CASH_OUT":
            return False
        matching = [
            item
            for item in self._strict_prior(
                event,
                recent_inbound_events,
                self.config.cashout_after_inbound_window_seconds,
            )
            if item.txn_type in RECEIVER_CREDIT_TYPES
            and credited_account(item) == event.name_orig
        ]
        inbound_total = fsum(item.amount for item in matching)
        return bool(
            inbound_total > 0
            and event.amount / inbound_total
            >= self.config.cashout_after_inbound_ratio_threshold
        )


