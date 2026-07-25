import unittest
from unittest.mock import patch

from fraud_pipeline import (
    PipelineConfig,
    RuleEngine,
    derive_account_state_updates,
    parse_csv_row,
    prediction_record_from_decision,
    prediction_record_to_dict,
    receiver_state_to_dict,
    sender_state_to_dict,
)


def sample_row(**overrides: str) -> dict[str, str]:
    row = {
        "step": "5",
        "type": "TRANSFER",
        "amount": "700.0",
        "nameOrig": "C1",
        "oldbalanceOrg": "1000.0",
        "newbalanceOrig": "300.0",
        "nameDest": "C2",
        "oldbalanceDest": "10.0",
        "newbalanceDest": "710.0",
        "isFraud": "0",
    }
    row.update(overrides)
    return row


class SerializationTests(unittest.TestCase):
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
    def test_sender_state_payload_contains_correlation_key(self) -> None:
        event = parse_csv_row(sample_row())
        sender_update = derive_account_state_updates(event)[0]

        payload = sender_state_to_dict(sender_update)

        self.assertEqual(payload["source_event_id"], event.event_id)
        self.assertEqual(payload["nameOrig"], event.name_orig)
        self.assertEqual(payload["oldbalanceOrg"], event.oldbalance_org)

    def test_receiver_state_payload_contains_correlation_key(self) -> None:
        event = parse_csv_row(sample_row())
        receiver_update = derive_account_state_updates(event)[1]

        payload = receiver_state_to_dict(receiver_update)

        self.assertEqual(payload["source_event_id"], event.event_id)
        self.assertEqual(payload["nameDest"], event.name_dest)
        self.assertEqual(payload["newbalanceDest"], event.newbalance_dest)

    def test_prediction_record_serialization_uses_shared_contract(self) -> None:
        event = parse_csv_row(sample_row(amount="250000.0", oldbalanceOrg="300000.0", newbalanceOrig="100.0", isFraud="1"))
        decision = RuleEngine(PipelineConfig()).evaluate(event)

        record = prediction_record_from_decision(event, decision)
        payload = prediction_record_to_dict(record)

        self.assertEqual(payload["event_id"], event.event_id)
        self.assertEqual(payload["account_id"], event.name_orig)
        self.assertEqual(payload["nameDest"], event.name_dest)
        self.assertEqual(payload["txn_type"], event.txn_type)
        self.assertIn("triggered_rules", payload)
        self.assertEqual(payload["is_alert"], decision.is_alert)


if __name__ == "__main__":
    unittest.main()


