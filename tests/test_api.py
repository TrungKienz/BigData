import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app
from api.service import KafkaPublishError


def sample_payload(**overrides):
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 299900.0,
        "nameOrig": "C1",
        "oldbalanceOrg": 300000.0,
        "nameDest": "C2",
        "oldbalanceDest": 1000.0,
    }
    payload.update(overrides)
    return payload


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.patchers = [
            patch("model.model_utils.get_scoring_config", return_value={"rule_weight": 0.6, "ml_weight": 0.4, "hybrid_threshold": 0.236128568649292}),
            patch("model.model_utils.get_model_info", return_value={"model_version": "test-xgb", "model_tag": "xgb", "feature_configuration": "deployment_safe"}),
            patch("model.model_utils.predict_proba", return_value=0.0),
            patch("api.service.model_is_loaded", return_value=True),
            patch("api.service.get_model_version", return_value="test-xgb"),
            patch("api.service.get_model_info", return_value={"artifact_path": "test", "model_loaded": True, "model_version": "test-xgb", "model_tag": "xgb", "feature_configuration": "deployment_safe", "feature_count": 24, "hybrid_threshold": 0.236128568649292, "rule_weight": 0.6, "ml_weight": 0.4}),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()

    def test_health_endpoint_reports_api_status(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("model_loaded", body)
        self.assertIn("model_version", body)
        self.assertIn("model_type", body)
        self.assertIn("prediction_logging_enabled", body)

    def test_model_info_endpoint_reports_artifact_contract(self):
        response = self.client.get("/model-info")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["model_tag"], "xgb")
        self.assertEqual(body["feature_configuration"], "deployment_safe")
        self.assertEqual(body["feature_count"], 24)

    def test_score_endpoint_returns_decision_payload(self):
        response = self.client.post("/score", json=sample_payload())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["is_alert"])
        self.assertEqual(body["decision"], "alert")
        self.assertIn("event_id", body)
        self.assertIn("risk_score", body)
        self.assertIn("rule_score", body)
        self.assertIn("ml_score", body)
        self.assertIn("hybrid_score", body)
        self.assertEqual(body["threshold"], 0.236128568649292)
        self.assertEqual(body["model_version"], "test-xgb")
        self.assertIn("account_drain_near_zero", body["triggered_rules"])

    def test_score_endpoint_validates_required_fields(self):
        response = self.client.post("/score", json={"step": 1})

        self.assertEqual(response.status_code, 422)

    def test_batch_score_returns_predictions(self):
        response = self.client.post(
            "/score/batch",
            json={"transactions": [sample_payload(), sample_payload(nameDest="C3", event_id="evt-2")]},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["predictions"]), 2)

    @patch.dict("api.service.os.environ", {"KAFKA_BOOTSTRAP_SERVERS": "localhost:9092"}, clear=False)
    @patch("api.service.publish_transaction_bundle")
    def test_score_endpoint_publishes_to_kafka_when_ingest_is_configured(self, mock_publish):
        response = self.client.post("/score", json=sample_payload())

        self.assertEqual(response.status_code, 200)
        mock_publish.assert_called_once()

    @patch.dict("api.service.os.environ", {"KAFKA_BOOTSTRAP_SERVERS": "localhost:9092"}, clear=False)
    @patch("api.service.publish_transaction_bundle", side_effect=KafkaPublishError("kafka unavailable"))
    def test_score_endpoint_returns_503_when_kafka_publish_fails(self, _mock_publish):
        response = self.client.post("/score", json=sample_payload())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "kafka unavailable")


if __name__ == "__main__":
    unittest.main()

