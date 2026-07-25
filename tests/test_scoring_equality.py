from datetime import datetime, timezone
from unittest.mock import patch

from api.schemas import ScoreRequest
from api.service import score_transaction
from fraud_pipeline import PipelineConfig, RuleEngine, parse_csv_row


SCORING = {"rule_weight": 0.6, "ml_weight": 0.4, "hybrid_threshold": 0.236128568649292}
MODEL_INFO = {"model_version": "test-xgb", "model_tag": "xgb", "feature_configuration": "deployment_safe"}


def sample_row() -> dict[str, str]:
    return {
        "step": "1",
        "type": "TRANSFER",
        "amount": "299900.0",
        "nameOrig": "C1",
        "oldbalanceOrg": "300000.0",
        "nameDest": "C2",
        "oldbalanceDest": "1000.0",
    }


def test_offline_api_and_spark_style_scoring_are_equal() -> None:
    with patch("model.model_utils.get_scoring_config", return_value=SCORING), patch(
        "model.model_utils.get_model_info", return_value=MODEL_INFO
    ), patch("model.model_utils.predict_proba", return_value=0.123456789), patch(
        "api.service.persist_prediction_record"
    ), patch("api.service.publish_transaction_bundle"):
        event = parse_csv_row(sample_row(), config=PipelineConfig(step_seconds=60))
        offline_decision = RuleEngine(PipelineConfig()).evaluate(event)

        payload = ScoreRequest(
            event_id=event.event_id,
            event_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            step=event.step,
            type=event.txn_type,
            amount=event.amount,
            nameOrig=event.name_orig,
            oldbalanceOrg=event.oldbalance_org,
            nameDest=event.name_dest,
            oldbalanceDest=event.oldbalance_dest,
        )
        api_decision = score_transaction(payload)

        spark_style_decision = RuleEngine(PipelineConfig()).evaluate(event)

    assert api_decision.rule_score == offline_decision.rule_score == spark_style_decision.rule_score
    assert api_decision.ml_score == offline_decision.ml_score == spark_style_decision.ml_score
    assert api_decision.hybrid_score == offline_decision.risk_score == spark_style_decision.risk_score
    assert api_decision.threshold == offline_decision.decision_threshold == spark_style_decision.decision_threshold
    assert api_decision.is_alert == offline_decision.is_alert == spark_style_decision.is_alert
    assert api_decision.model_version == offline_decision.ml_model_version == spark_style_decision.ml_model_version
