import inspect
import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from fraud_pipeline import PipelineConfig, parse_csv_row
from fraud_pipeline.features import (
    DYNAMIC_FEATURES,
    FORBIDDEN_FEATURE_COLUMNS,
    DynamicFeatureState,
    FeatureContext,
    assert_no_forbidden_features,
    build_feature_record,
)
from fraud_pipeline.models import TransactionEvent
from fraud_pipeline.rules import RuleEngine, combine_risk_score_arrays, combine_risk_scores
from model import model_utils
from model.prepare_features import (
    _event_from_row,
    create_split_manifest,
    prepare_feature_artifacts,
    select_features,
    validate_split_manifest,
)
from model.train_models import (
    build_estimator,
    build_preprocessor,
    calculate_scale_pos_weight,
    train_and_export_models,
    tune_threshold_by_business_cost,
    validate_test_prediction_consistency,
)


def cleaned_frame(step_count: int = 15, rows_per_step: int = 4) -> pd.DataFrame:
    rows = []
    transaction_types = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"]
    browsers = ["chrome", "firefox", "edge", "safari"]
    devices = ["desktop", "mobile", "tablet", "mobile"]
    countries = ["VN", "US", "SG", "TH"]
    row_id = 0
    for step in range(step_count):
        for offset in range(rows_per_step):
            amount = float(100 + step * 11 + offset * 17)
            old_org = float(5_000 + offset * 1_000)
            old_dest = float(2_000 + step * 3)
            txn_type = transaction_types[(step + offset) % len(transaction_types)]
            rows.append(
                {
                    "row_id": row_id,
                    "step": step,
                    "type": txn_type,
                    "amount": amount,
                    "nameOrig": f"C{offset}",
                    "nameDest": f"D{(step + offset) % 7}",
                    "oldbalanceOrg": old_org,
                    "newbalanceOrig": max(old_org - amount, 0.0),
                    "oldbalanceDest": old_dest,
                    "newbalanceDest": old_dest + amount,
                    "isFraud": int(offset == 0),
                    "hour_of_day": step % 24,
                    "is_night_transaction": int(step % 24 >= 22 or step % 24 <= 6),
                    "customer_account_age_days": float(30 + step + offset),
                    "browser": browsers[offset],
                    "device_type": devices[offset],
                    "new_device_flag": int((step + offset) % 5 == 0),
                    "billing_country": countries[offset],
                    "ip_country": countries[(offset + step) % len(countries)],
                    "ip_billing_distance_km": float(step * 10 + offset),
                    "ip_billing_country_mismatch": int(step % 3 == 0 and offset == 1),
                    "shipping_billing_mismatch": int(step % 4 == 0 and offset == 2),
                    "failed_payment_attempts_24h": float((step + offset) % 3),
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def event(
    event_id: str,
    step: int,
    *,
    txn_type: str = "TRANSFER",
    amount: float = 100.0,
    name_orig: str = "C1",
    name_dest: str = "C2",
) -> TransactionEvent:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=step)
    return TransactionEvent(
        event_id=event_id,
        event_time=timestamp,
        producer_ts=timestamp,
        step=step,
        txn_type=txn_type,
        amount=amount,
        name_orig=name_orig,
        oldbalance_org=1_000.0,
        newbalance_orig=max(1_000.0 - amount, 0.0),
        name_dest=name_dest,
        oldbalance_dest=100.0,
        newbalance_dest=100.0 + amount,
        is_fraud=0,
        hour_of_day=step % 24,
        is_night_transaction=int(step % 24 >= 22 or step % 24 <= 6),
        customer_account_age_days=120.0,
        browser="chrome",
        device_type="mobile",
        new_device_flag=0,
        billing_country="VN",
        ip_country="VN",
        ip_billing_distance_km=5.0,
        ip_billing_country_mismatch=0,
        shipping_billing_mismatch=0,
        failed_payment_attempts_24h=0.0,
    )


class SplitManifestTests(unittest.TestCase):
    def test_manifest_is_chronological_deterministic_and_step_atomic(self) -> None:
        frame = cleaned_frame()
        first = create_split_manifest(frame)
        second = create_split_manifest(frame.sample(frac=1.0, random_state=7))

        validate_split_manifest(first, frame)
        self.assertTrue(
            first.sort_values("row_id").reset_index(drop=True).equals(
                second.sort_values("row_id").reset_index(drop=True)
            )
        )
        self.assertEqual(len(first), len(frame))
        self.assertEqual(first["row_id"].nunique(), len(frame))
        self.assertEqual(int(first.groupby("step")["split"].nunique().max()), 1)
        ratios = first["split"].value_counts(normalize=True)
        self.assertAlmostEqual(float(ratios["train"]), 0.60, delta=0.07)
        self.assertAlmostEqual(float(ratios["validation"]), 0.20, delta=0.07)
        self.assertAlmostEqual(float(ratios["test"]), 0.20, delta=0.07)
        self.assertLess(
            first.loc[first["split"] == "train", "step"].max(),
            first.loc[first["split"] == "validation", "step"].min(),
        )
        self.assertLess(
            first.loc[first["split"] == "validation", "step"].max(),
            first.loc[first["split"] == "test", "step"].min(),
        )

    def test_schema_validation_reports_all_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "incomplete.parquet"
            pd.DataFrame({"step": [1], "amount": [1.0]}).to_parquet(path)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                prepare_feature_artifacts(path, Path(temp_dir) / "artifacts")

    def test_schema_validation_rejects_fractional_labels(self) -> None:
        frame = cleaned_frame()
        frame["isFraud"] = frame["isFraud"].astype(float)
        frame.loc[0, "isFraud"] = 0.5
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fractional-label.parquet"
            frame.to_parquet(path, index=False)

            with self.assertRaisesRegex(ValueError, "binary 0/1"):
                prepare_feature_artifacts(path, Path(temp_dir) / "artifacts")

    def test_full_csv_uses_complete_step_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transactions_cleaned.csv"
            cleaned_frame().to_csv(path, index=False)
            artifacts = Path(temp_dir) / "artifacts"

            result = prepare_feature_artifacts(path, artifacts, chunk_size=3)

            self.assertEqual(result["metadata"]["source_format"], "csv")
            self.assertEqual(
                result["metadata"]["preparation_mode"],
                "streaming_complete_step_batches",
            )
            manifest = pd.read_parquet(artifacts / "split_manifest.parquet")
            self.assertEqual(int(manifest.groupby("step")["split"].nunique().max()), 1)


class FeatureSemanticsTests(unittest.TestCase):
    def test_same_step_events_do_not_see_each_other(self) -> None:
        config = PipelineConfig(fan_out_window_seconds=7_200)
        state = DynamicFeatureState(config)
        first = event("one", 1, name_dest="D1")
        second = event("two", 1, name_dest="D2")

        contexts, values = state.calculate_step([first, second])

        self.assertEqual(values[0]["sender_recent_txn_count"], 0.0)
        self.assertEqual(values[1]["sender_recent_txn_count"], 0.0)
        self.assertEqual(values[0]["velocity_transactions_1h"], 0.0)
        self.assertEqual(values[1]["velocity_transactions_1h"], 0.0)
        self.assertEqual(len(contexts[0].recent_sender_events), 0)
        state.update_after_step([first, second])

        later = event("three", 2, name_dest="D1")
        _, later_values = state.calculate_step([later])
        self.assertEqual(later_values[0]["sender_recent_txn_count"], 2.0)
        self.assertEqual(later_values[0]["velocity_transactions_1h"], 2.0)
        self.assertEqual(later_values[0]["is_new_counterparty"], 0.0)

    def test_last_transaction_timestamp_survives_one_hour_history_truncation(self) -> None:
        state = DynamicFeatureState(PipelineConfig())
        state.process_step([event("first", 1)])

        values = state.process_step([event("later", 4)])

        self.assertEqual(values[0]["time_since_last_transaction"], 10_800.0)

    def test_inbound_ratio_uses_prior_money_received_by_cashout_sender(self) -> None:
        config = PipelineConfig(cashout_after_inbound_window_seconds=10_800)
        state = DynamicFeatureState(config)
        inbound = event(
            "inbound",
            1,
            amount=200.0,
            name_orig="UPSTREAM",
            name_dest="CASHOUT_ACCOUNT",
        )
        state.process_step([inbound])
        cashout = event(
            "cashout",
            2,
            txn_type="CASH_OUT",
            amount=100.0,
            name_orig="CASHOUT_ACCOUNT",
            name_dest="MERCHANT",
        )

        values = state.process_step([cashout])

        self.assertAlmostEqual(values[0]["inbound_to_cashout_ratio"], 0.5)

    def test_cash_in_credits_origin_account_for_inbound_history(self) -> None:
        config = PipelineConfig(cashout_after_inbound_window_seconds=10_800)
        state = DynamicFeatureState(config)
        cash_in = event(
            "cash-in",
            1,
            txn_type="CASH_IN",
            amount=250.0,
            name_orig="CASHOUT_ACCOUNT",
            name_dest="CASH_AGENT",
        )
        state.process_step([cash_in])
        cashout = event(
            "cashout-after-cash-in",
            2,
            txn_type="CASH_OUT",
            amount=100.0,
            name_orig="CASHOUT_ACCOUNT",
            name_dest="MERCHANT",
        )

        values = state.process_step([cashout])

        self.assertAlmostEqual(values[0]["inbound_to_cashout_ratio"], 0.4)

    def test_cash_in_fan_in_counts_distinct_cash_agents(self) -> None:
        config = PipelineConfig(
            fan_in_window_seconds=10_800,
            fan_in_distinct_sender_threshold=3,
            fan_in_total_amount_threshold=1.0,
        )
        engine = RuleEngine(config)
        history = [
            event(
                "cash-agent-1",
                1,
                txn_type="CASH_IN",
                name_orig="ACCOUNT",
                name_dest="AGENT_1",
            ),
            event(
                "cash-agent-2",
                2,
                txn_type="CASH_IN",
                name_orig="ACCOUNT",
                name_dest="AGENT_2",
            ),
        ]
        current = event(
            "cash-agent-3",
            3,
            txn_type="CASH_IN",
            name_orig="ACCOUNT",
            name_dest="AGENT_3",
        )

        evaluation = engine.evaluate_rules(
            current, FeatureContext(recent_receiver_events=history)
        )

        self.assertIn("receiver_fan_in_burst", evaluation.triggered_rules)

    def test_deployment_rule_score_does_not_use_post_transaction_balance(self) -> None:
        engine = RuleEngine(PipelineConfig())
        first = event("first-balance", 2, amount=999.0)
        second = TransactionEvent(
            **{**first.__dict__, "event_id": "second-balance", "newbalance_orig": 900.0}
        )

        first_score = engine.evaluate_rules(first).score
        second_score = engine.evaluate_rules(second).score

        self.assertEqual(first_score, second_score)

    def test_same_step_is_excluded_even_if_timestamps_differ(self) -> None:
        state = DynamicFeatureState(PipelineConfig(fan_out_window_seconds=10_800))
        first = event("same-step-first", 1)
        state.update_after_step([first])
        later_timestamp = TransactionEvent(
            **{
                **event("same-step-later", 1).__dict__,
                "event_time": first.event_time + timedelta(minutes=30),
                "producer_ts": first.event_time + timedelta(minutes=30),
            }
        )

        values = state.calculate_step([later_timestamp])[1]

        self.assertEqual(values[0]["sender_recent_txn_count"], 0.0)
        self.assertEqual(values[0]["velocity_transactions_1h"], 0.0)

    def test_label_does_not_change_event_id_or_inference_features(self) -> None:
        base = {
            "step": "1",
            "type": "TRANSFER",
            "amount": "100.0",
            "nameOrig": "C1",
            "oldbalanceOrg": "1000.0",
            "newbalanceOrig": "900.0",
            "nameDest": "C2",
            "oldbalanceDest": "0.0",
            "newbalanceDest": "100.0",
            "hour_of_day": "1",
            "is_night_transaction": "1",
            "customer_account_age_days": "120",
            "browser": "chrome",
            "device_type": "mobile",
            "new_device_flag": "0",
            "billing_country": "VN",
            "ip_country": "VN",
            "ip_billing_distance_km": "1.0",
            "ip_billing_country_mismatch": "0",
            "shipping_billing_mismatch": "0",
            "failed_payment_attempts_24h": "0",
        }
        legitimate = parse_csv_row({**base, "isFraud": "0"})
        fraudulent = parse_csv_row({**base, "isFraud": "1"})

        self.assertEqual(legitimate.event_id, fraudulent.event_id)
        self.assertEqual(
            build_feature_record(legitimate), build_feature_record(fraudulent)
        )

    def test_forbidden_and_post_transaction_features_are_excluded(self) -> None:
        frame = cleaned_frame()
        manifest = create_split_manifest(frame)
        from model.prepare_features import build_engineered_features

        engineered, _ = build_engineered_features(frame)
        selected, report = select_features(
            engineered, manifest, "deployment_safe"
        )

        assert_no_forbidden_features(selected)
        self.assertFalse(FORBIDDEN_FEATURE_COLUMNS.intersection(selected))
        report_by_name = {row["feature"]: row for row in report}
        self.assertEqual(report_by_name["step"]["reason"], "raw_time_index_not_deployable")
        self.assertIn(
            "post_transaction",
            report_by_name["newbalance_orig"]["reason"],
        )


class ModelDevelopmentUnitTests(unittest.TestCase):
    def test_preprocessor_fits_train_only_and_ignores_unknown_categories(self) -> None:
        train = pd.DataFrame(
            {"amount": [0.0, 2.0, 4.0, 6.0], "txn_type": ["A", "B", "A", "B"]}
        )
        validation = pd.DataFrame({"amount": [1_000.0], "txn_type": ["UNSEEN"]})
        preprocessor = build_preprocessor(
            ["amount", "txn_type"], scale_numeric=True
        )

        preprocessor.fit(train)
        transformed = preprocessor.transform(validation)
        scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]

        self.assertAlmostEqual(float(scaler.mean_[0]), 3.0)
        self.assertEqual(transformed.shape[0], 1)

    def test_default_imbalance_strategies_do_not_use_smote(self) -> None:
        labels = np.array([0, 0, 0, 1], dtype=np.int8)
        self.assertEqual(calculate_scale_pos_weight(labels), 3.0)
        logistic, logistic_strategy = build_estimator("logreg", labels, quick=True)
        forest, forest_strategy = build_estimator("rf", labels, quick=True)
        _, xgb_strategy = build_estimator("xgb", labels, quick=True)
        lgbm_strategy = None
        try:
            _, lgbm_strategy = build_estimator("lgbm", labels, quick=True)
        except RuntimeError as exc:
            if "LightGBM" not in str(exc):
                raise

        self.assertEqual(logistic.class_weight, "balanced")
        self.assertEqual(forest.class_weight, "balanced_subsample")
        for strategy in (
            logistic_strategy,
            forest_strategy,
            xgb_strategy,
        ):
            self.assertFalse(strategy["smote"])
        if lgbm_strategy is not None:
            self.assertFalse(lgbm_strategy["smote"])
        self.assertEqual(xgb_strategy["scale_pos_weight"], 3.0)
        if lgbm_strategy is not None:
            self.assertEqual(lgbm_strategy["scale_pos_weight"], 3.0)

    def test_threshold_tuner_has_validation_only_contract(self) -> None:
        parameters = inspect.signature(tune_threshold_by_business_cost).parameters
        self.assertEqual(
            list(parameters)[:3],
            ["validation_labels", "validation_scores", "validation_amounts"],
        )
        self.assertFalse(any("test" in name for name in parameters))
        result = tune_threshold_by_business_cost(
            [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], [1.0, 100.0, 1.0, 50.0]
        )
        self.assertGreaterEqual(result["threshold"], 0.0)
        self.assertLessEqual(result["threshold"], 1.0)

    def test_threshold_tuner_represents_no_alert_when_score_is_one(self) -> None:
        result = tune_threshold_by_business_cost(
            [0, 1],
            [1.0, 0.0],
            [1.0, 0.1],
            false_alarm_unit_cost=10.0,
        )

        self.assertGreater(result["threshold"], 1.0)
        self.assertEqual(result["business_cost"], 0.1)

    def test_scalar_and_vector_hybrid_scores_match(self) -> None:
        scalar = combine_risk_scores(0.25, 0.75, 0.6, 0.4)
        vector = combine_risk_score_arrays(
            np.array([0.25]), np.array([0.75]), 0.6, 0.4
        )[0]
        self.assertAlmostEqual(scalar, float(vector), places=12)

    def test_prediction_error_fails_fast(self) -> None:
        current = event(
            "prediction-error",
            2,
            amount=200_000.0,
            name_dest="NEW_COUNTERPARTY",
        )
        engine = RuleEngine(PipelineConfig())
        with patch.object(model_utils, "predict_proba", side_effect=RuntimeError("synthetic inference failure")), patch.object(
            model_utils,
            "get_scoring_config",
            return_value={"rule_weight": 0.6, "ml_weight": 0.4, "hybrid_threshold": 0.236128568649292},
        ), patch.object(
            model_utils,
            "get_model_info",
            return_value={"model_version": "test-xgb", "model_tag": "xgb", "feature_configuration": "deployment_safe"},
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
                engine.evaluate(current)


class EndToEndArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data_path = cls.root / "transactions_cleaned.parquet"
        cls.source = cleaned_frame()
        cls.source.to_parquet(cls.data_path, index=False)
        cls.artifacts = cls.root / "artifacts"
        prepare_feature_artifacts(
            cls.data_path,
            cls.artifacts,
            feature_configuration="deployment_safe",
        )
        cls.result = train_and_export_models(
            cls.artifacts,
            model_types=["xgb"],
            false_alarm_unit_cost=5.0,
            quick=True,
            run_feature_ablation=True,
            ablation_max_rows=1_000,
            update_runtime_pointer=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        model_utils.reset_artifact_cache()
        cls.temporary.cleanup()

    def test_required_artifacts_exist_and_predictions_recreate_metrics(self) -> None:
        required = [
            "split_manifest.parquet",
            "train_features.parquet",
            "validation_features.parquet",
            "test_features.parquet",
            "selected_features.json",
            "excluded_features.json",
            "feature_selection_report.csv",
            "ablation_results.csv",
            "model_comparison.csv",
            "model_comparison.json",
            "sensitivity_analysis.csv",
            "evaluation_results.json",
            "test_predictions.parquet",
            "fraud_pipeline_selected.joblib",
            "dataset_metadata.json",
        ]
        self.assertFalse([name for name in required if not (self.artifacts / name).exists()])
        evaluation = json.loads(
            (self.artifacts / "evaluation_results.json").read_text(encoding="utf-8")
        )
        predictions = pd.read_parquet(self.artifacts / "test_predictions.parquet")
        validate_test_prediction_consistency(
            predictions, evaluation["evaluation"]["hybrid"]
        )
        ablation = pd.read_csv(self.artifacts / "ablation_results.csv")
        self.assertEqual(
            set(ablation["feature_set"]),
            {
                "base_transaction_features",
                "base_plus_synthetic_contextual_features",
                "base_plus_dynamic_features",
                "full_paysim",
                "deployment_safe",
            },
        )
        comparison = pd.read_csv(self.artifacts / "model_comparison.csv")
        self.assertEqual(
            set(comparison["model_tag"]), {"xgb"}
        )
        self.assertEqual(int(comparison["selected"].sum()), 1)

    def test_artifact_round_trip_preserves_feature_order_and_predictions(self) -> None:
        artifact_path = self.artifacts / "fraud_pipeline_selected.joblib"
        first = joblib.load(artifact_path)
        second = model_utils.load_model_artifact(
            artifact_path, strict=True, force_reload=True
        )
        test = pd.read_parquet(self.artifacts / "test_features.parquet")
        original = model_utils.predict_frame(first, test)
        reordered = model_utils.predict_frame(
            second, test[test.columns[::-1]]
        )

        np.testing.assert_allclose(original, reordered, rtol=1e-10, atol=1e-12)
        self.assertEqual(first["feature_columns"], second["feature_columns"])

    def test_model_utils_runtime_prediction_matches_offline_prediction(self) -> None:
        artifact_path = self.artifacts / "fraud_pipeline_selected.joblib"
        bundle = joblib.load(artifact_path)
        current = event("runtime", 8, amount=321.0)
        dynamic = {name: float(index + 1) for index, name in enumerate(DYNAMIC_FEATURES)}
        record = build_feature_record(current, dynamic_features=dynamic)
        offline = model_utils.predict_frame(bundle, pd.DataFrame([record]))[0]

        with patch.dict(os.environ, {"FRAUD_MODEL_ARTIFACT": str(artifact_path)}), patch.object(model_utils, "validate_production_model_artifact", model_utils.validate_model_artifact):
            model_utils.reset_artifact_cache()
            runtime = model_utils.predict_proba(current, dynamic_features=dynamic)

        self.assertAlmostEqual(float(offline), runtime, places=12)

    def test_persisted_offline_features_match_runtime_feature_builder(self) -> None:
        config = PipelineConfig()
        state = DynamicFeatureState(config)
        engine = RuleEngine(config)
        runtime_rows = {}
        ordered = self.source.sort_values(["step", "row_id"], kind="stable")
        for _, step_rows in ordered.groupby("step", sort=True):
            events = [
                _event_from_row(row, config)
                for row in step_rows.itertuples(index=False)
            ]
            contexts, dynamic_rows = state.calculate_step(events)
            for source_row, current, context, dynamic in zip(
                step_rows.itertuples(index=False), events, contexts, dynamic_rows
            ):
                runtime_rows[int(source_row.row_id)] = {
                    **build_feature_record(current, dynamic_features=dynamic),
                    "rule_score": engine.evaluate_rules(current, context).score,
                }
            state.update_after_step(events)

        persisted = pd.read_parquet(self.artifacts / "test_features.parquet")
        selected = json.loads(
            (self.artifacts / "selected_features.json").read_text(encoding="utf-8")
        )["selected_features"]
        for row in persisted.itertuples(index=False):
            runtime = runtime_rows[int(row.row_id)]
            for name in selected:
                expected = getattr(row, name)
                actual = runtime[name]
                if isinstance(expected, str):
                    self.assertEqual(actual, expected)
                else:
                    self.assertAlmostEqual(float(actual), float(expected), places=7)
            self.assertAlmostEqual(
                float(runtime["rule_score"]), float(row.rule_score), places=12
            )

    def test_rule_engine_uses_exact_exported_hybrid_contract(self) -> None:
        artifact_path = self.artifacts / "fraud_pipeline_selected.joblib"
        bundle = joblib.load(artifact_path)
        config = replace(PipelineConfig(), **bundle["rule_config"])
        engine = RuleEngine(config)
        current = event("hybrid-runtime", 9, amount=200_000.0)
        context = FeatureContext()

        with patch.dict(os.environ, {"FRAUD_MODEL_ARTIFACT": str(artifact_path)}), patch.object(model_utils, "validate_production_model_artifact", model_utils.validate_model_artifact):
            model_utils.reset_artifact_cache()
            decision = engine.evaluate(current, context=context)

        expected = combine_risk_scores(
            decision.rule_score,
            decision.ml_score,
            bundle["rule_weight"],
            bundle["ml_weight"],
        )
        self.assertAlmostEqual(decision.risk_score, expected, places=12)
        self.assertEqual(decision.decision_threshold, bundle["hybrid_threshold"])
        self.assertEqual(
            decision.is_alert, expected >= bundle["hybrid_threshold"]
        )

    def test_artifact_refresh_detects_atomic_same_path_replacement(self) -> None:
        artifact_path = self.artifacts / "fraud_pipeline_selected.joblib"
        first = model_utils.load_model_artifact(
            artifact_path, strict=True, force_reload=True
        )
        stat = artifact_path.stat()
        os.utime(
            artifact_path,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
        )

        with patch.dict(os.environ, {"FRAUD_MODEL_ARTIFACT": str(artifact_path)}), patch.object(model_utils, "validate_production_model_artifact", model_utils.validate_model_artifact):
            second = model_utils.refresh_model_artifact(strict=True)

        self.assertIsNot(first, second)


if __name__ == "__main__":
    unittest.main()





