import unittest
from unittest.mock import patch

from fraud_pipeline import PipelineConfig, RuleEngine, parse_csv_row


def sample_row(**overrides: str) -> dict[str, str]:
    row = {
        "step": "1",
        "type": "TRANSFER",
        "amount": "250000.0",
        "nameOrig": "C1",
        "oldbalanceOrg": "300000.0",
        "newbalanceOrig": "50000.0",
        "nameDest": "C2",
        "oldbalanceDest": "1000.0",
        "newbalanceDest": "251000.0",
        "isFraud": "1",
    }
    row.update(overrides)
    return row


def parse_test_row(**overrides: str):
    return parse_csv_row(
        sample_row(**overrides), config=PipelineConfig(step_seconds=60)
    )


class RuleEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.patchers = [
            patch("model.model_utils.get_scoring_config", return_value={"rule_weight": 0.6, "ml_weight": 0.4, "hybrid_threshold": 0.236128568649292}),
            patch("model.model_utils.get_model_info", return_value={"model_version": "test-xgb", "model_tag": "xgb", "feature_configuration": "deployment_safe"}),
            patch("model.model_utils.predict_proba", return_value=0.0),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()
    def test_account_drain_near_zero_triggers_alert(self) -> None:
        engine = RuleEngine(
            PipelineConfig(
                account_drain_min_balance_floor=50_000.0,
                account_drain_ratio_threshold=0.8,
                account_drain_near_zero_balance=1_000.0,
            )
        )
        event = parse_test_row(
            amount="299900.0",
            oldbalanceOrg="300000.0",
            newbalanceOrig="100.0",
        )

        decision = engine.evaluate(event)

        self.assertTrue(decision.is_alert)
        self.assertIn("account_drain_near_zero", decision.triggered_rules)

    def test_sender_fan_out_and_structuring_increase_risk(self) -> None:
        engine = RuleEngine(
            PipelineConfig(
                fan_out_window_seconds=600,
                fan_out_distinct_receiver_threshold=3,
                fan_out_total_amount_threshold=200_000.0,
                structuring_window_seconds=600,
                structuring_count_threshold=4,
                structuring_min_amount=40_000.0,
                structuring_max_amount=90_000.0,
                structuring_total_amount_threshold=220_000.0,
                new_counterparty_amount_threshold=999_999.0,
            )
        )
        history = [
            parse_test_row(step="1", amount="60000.0", nameDest="C2", oldbalanceOrg="500000.0", newbalanceOrig="440000.0"),
            parse_test_row(step="2", amount="55000.0", nameDest="C3", oldbalanceOrg="440000.0", newbalanceOrig="385000.0"),
            parse_test_row(step="3", amount="50000.0", nameDest="C4", oldbalanceOrg="385000.0", newbalanceOrig="335000.0"),
        ]
        event = parse_test_row(step="4", amount="70000.0", nameDest="C5", oldbalanceOrg="335000.0", newbalanceOrig="265000.0")

        decision = engine.evaluate(event, recent_sender_events=history)

        self.assertIn("sender_fan_out_burst", decision.triggered_rules)
        self.assertIn("structured_split_transfer", decision.triggered_rules)
        self.assertGreater(decision.risk_score, 0.28)

    def test_receiver_fan_in_triggers_alert(self) -> None:
        config = PipelineConfig(
            fan_in_window_seconds=600,
            fan_in_distinct_sender_threshold=3,
            fan_in_total_amount_threshold=200_000.0,
        )
        engine = RuleEngine(config)
        first = parse_test_row(step="1", amount="70000.0", nameOrig="S1", nameDest="C9")
        second = parse_test_row(step="2", amount="80000.0", nameOrig="S2", nameDest="C9")
        current = parse_test_row(step="3", amount="90000.0", nameOrig="S3", nameDest="C9")

        decision = engine.evaluate(current, recent_receiver_events=[first, second])

        self.assertIn("receiver_fan_in_burst", decision.triggered_rules)

    def test_new_counterparty_large_transfer_triggers_alert(self) -> None:
        engine = RuleEngine(PipelineConfig(new_counterparty_amount_threshold=150_000.0))
        event = parse_test_row(amount="200000.0", nameDest="C999")

        decision = engine.evaluate(event, known_counterparties={"C2", "C3"})

        self.assertIn("new_counterparty_large_transfer", decision.triggered_rules)

    def test_cashout_after_inbound_chain_triggers_alert(self) -> None:
        engine = RuleEngine(
            PipelineConfig(
                cashout_after_inbound_window_seconds=1800,
                cashout_after_inbound_ratio_threshold=0.75,
                new_counterparty_amount_threshold=999_999.0,
            )
        )
        inbound = parse_test_row(
            step="1",
            type="TRANSFER",
            amount="120000.0",
            nameOrig="UPSTREAM",
            nameDest="C1",
            oldbalanceDest="1000.0",
            newbalanceDest="121000.0",
        )
        cashout = parse_test_row(
            step="2",
            type="CASH_OUT",
            amount="100000.0",
            nameOrig="C1",
            nameDest="MERCHANT",
            oldbalanceOrg="121000.0",
            newbalanceOrig="21000.0",
        )

        decision = engine.evaluate(cashout, recent_inbound_events=[inbound])

        self.assertIn("cashout_after_inbound_chain", decision.triggered_rules)


if __name__ == "__main__":
    unittest.main()


