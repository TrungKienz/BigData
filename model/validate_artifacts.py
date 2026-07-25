from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fraud_pipeline.features import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_FEATURE_COLUMNS,
    POST_TRANSACTION_FEATURES,
)


SPLIT_NAMES = ("train", "validation", "test")
FEATURE_REQUIRED_ARTIFACTS = (
    "split_manifest.parquet",
    "train_features.parquet",
    "validation_features.parquet",
    "test_features.parquet",
    "selected_features.json",
    "excluded_features.json",
    "feature_selection_report.csv",
    "feature_schema.json",
    "dataset_metadata.json",
)
FINAL_REQUIRED_ARTIFACTS = (
    "model_comparison.csv",
    "model_comparison.json",
    "evaluation_results.json",
    "test_predictions.parquet",
    "fraud_pipeline_selected.joblib",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_feature_artifacts(
    artifacts_dir: str | Path,
    *,
    batch_size: int = 200_000,
) -> dict[str, Any]:
    artifacts = Path(artifacts_dir)
    missing_artifacts = [
        name for name in FEATURE_REQUIRED_ARTIFACTS if not (artifacts / name).exists()
    ]
    _require(not missing_artifacts, f"Missing feature artifacts: {missing_artifacts}")

    selected_payload = _read_json(artifacts / "selected_features.json")
    schema_payload = _read_json(artifacts / "feature_schema.json")
    metadata = _read_json(artifacts / "dataset_metadata.json")
    selected = [str(name) for name in selected_payload["selected_features"]]
    _require(
        selected_payload["feature_configuration"] == "deployment_safe",
        "Feature configuration must be deployment_safe",
    )
    _require(
        schema_payload["selected_features"] == selected,
        "feature_schema.json and selected_features.json disagree on feature order",
    )
    forbidden = sorted(set(selected).intersection(FORBIDDEN_FEATURE_COLUMNS))
    post_transaction = sorted(set(selected).intersection(POST_TRANSACTION_FEATURES))
    _require(not forbidden, f"Selected features contain forbidden columns: {forbidden}")
    _require(
        not post_transaction,
        f"deployment_safe contains post-transaction columns: {post_transaction}",
    )

    manifest = pd.read_parquet(artifacts / "split_manifest.parquet")
    _require(
        {"row_id", "step", "split"}.issubset(manifest.columns),
        "Split manifest lacks row_id, step, or split",
    )
    _require(not manifest["row_id"].isna().any(), "Manifest row_id contains nulls")
    _require(not manifest["row_id"].duplicated().any(), "Manifest row_id is not unique")
    _require(
        set(manifest["split"].unique()) == set(SPLIT_NAMES),
        "Manifest does not contain exactly train/validation/test",
    )
    step_split_counts = manifest.groupby("step", sort=True)["split"].nunique()
    _require(
        int(step_split_counts.max()) == 1,
        "At least one step appears in multiple splits",
    )
    manifest_boundaries = manifest.groupby("split")["step"].agg(["min", "max"])
    _require(
        int(manifest_boundaries.loc["train", "max"])
        < int(manifest_boundaries.loc["validation", "min"])
        < int(manifest_boundaries.loc["validation", "max"])
        < int(manifest_boundaries.loc["test", "min"]),
        "Split manifest is not strictly chronological",
    )
    expected_total = int(metadata["row_count"])
    _require(
        len(manifest) == expected_total,
        f"Manifest has {len(manifest)} rows, expected {expected_total}",
    )

    common_schema = None
    row_ids_by_split: dict[str, np.ndarray] = {}
    steps_by_split: dict[str, set[int]] = {}
    split_stats: dict[str, dict[str, Any]] = {}
    numeric_features = [
        name for name in selected if name not in CATEGORICAL_FEATURES
    ]
    scan_columns = list(
        dict.fromkeys(
            ["row_id", "split", "step", "label", "rule_score", "amount", *selected]
        )
    )

    for split_name in SPLIT_NAMES:
        path = artifacts / f"{split_name}_features.parquet"
        parquet = pq.ParquetFile(path)
        arrow_schema = parquet.schema_arrow.remove_metadata()
        if common_schema is None:
            common_schema = arrow_schema
        else:
            _require(
                arrow_schema.equals(common_schema),
                f"{split_name} feature schema differs from train schema",
            )
        names = parquet.schema_arrow.names
        missing_columns = [name for name in scan_columns if name not in names]
        _require(not missing_columns, f"{path.name} lacks columns: {missing_columns}")
        selected_order = [name for name in names if name in selected]
        _require(
            selected_order == selected,
            f"{path.name} selected feature order differs from contract",
        )

        row_id_chunks: list[np.ndarray] = []
        step_values: set[int] = set()
        row_count = 0
        fraud_count = 0
        feature_missing_count = 0
        infinite_count = 0
        rule_score_min = np.inf
        rule_score_max = -np.inf
        split_values: set[str] = set()
        label_values: set[int] = set()
        step_min: int | None = None
        step_max: int | None = None

        for batch in parquet.iter_batches(batch_size=batch_size, columns=scan_columns):
            current = batch.to_pandas()
            row_count += len(current)
            ids = current["row_id"].to_numpy(dtype=np.int64, copy=False)
            row_id_chunks.append(ids.copy())
            current_steps = current["step"].to_numpy(dtype=np.int64, copy=False)
            step_values.update(int(value) for value in np.unique(current_steps))
            if len(current_steps):
                current_min = int(current_steps.min())
                current_max = int(current_steps.max())
                step_min = current_min if step_min is None else min(step_min, current_min)
                step_max = current_max if step_max is None else max(step_max, current_max)
            labels = current["label"].to_numpy(dtype=np.int8, copy=False)
            fraud_count += int(labels.sum())
            label_values.update(int(value) for value in np.unique(labels))
            split_values.update(str(value) for value in current["split"].unique())

            feature_missing_count += int(current[selected].isna().sum().sum())
            for name in numeric_features:
                values = pd.to_numeric(current[name], errors="coerce").to_numpy(
                    dtype=np.float64, copy=False
                )
                infinite_count += int(np.isinf(values).sum())
            rule_scores = current["rule_score"].to_numpy(dtype=np.float64, copy=False)
            _require(
                bool(np.isfinite(rule_scores).all()),
                f"{path.name} rule_score contains NaN or infinity",
            )
            rule_score_min = min(rule_score_min, float(rule_scores.min()))
            rule_score_max = max(rule_score_max, float(rule_scores.max()))

        row_ids = np.concatenate(row_id_chunks)
        _require(
            len(np.unique(row_ids)) == len(row_ids),
            f"{path.name} contains duplicate row_id values",
        )
        expected_ids = manifest.loc[
            manifest["split"] == split_name, "row_id"
        ].to_numpy(dtype=np.int64)
        _require(
            np.array_equal(np.sort(row_ids), np.sort(expected_ids)),
            f"{path.name} row_id values do not exactly match the manifest",
        )
        _require(split_values == {split_name}, f"{path.name} contains another split")
        _require(label_values == {0, 1}, f"{path.name} does not contain both labels")
        _require(feature_missing_count == 0, f"{path.name} contains missing features")
        _require(infinite_count == 0, f"{path.name} contains infinite features")
        _require(
            0.0 <= rule_score_min <= rule_score_max <= 1.0,
            f"{path.name} rule_score is outside [0, 1]",
        )
        _require(
            row_count == int(metadata["split_counts"][split_name]),
            f"{path.name} row count differs from dataset metadata",
        )
        row_ids_by_split[split_name] = row_ids
        steps_by_split[split_name] = step_values
        split_stats[split_name] = {
            "row_count": row_count,
            "row_ratio": float(row_count / expected_total),
            "fraud_count": fraud_count,
            "fraud_rate": float(fraud_count / row_count),
            "step_min": step_min,
            "step_max": step_max,
            "rule_score_min": rule_score_min,
            "rule_score_max": rule_score_max,
            "missing_feature_values": feature_missing_count,
            "infinite_feature_values": infinite_count,
        }

    all_ids = np.concatenate([row_ids_by_split[name] for name in SPLIT_NAMES])
    _require(len(all_ids) == expected_total, "Feature split rows do not sum to dataset")
    _require(
        len(np.unique(all_ids)) == expected_total,
        "Feature split row_id values overlap",
    )
    _require(
        not (steps_by_split["train"] & steps_by_split["validation"])
        and not (steps_by_split["train"] & steps_by_split["test"])
        and not (steps_by_split["validation"] & steps_by_split["test"]),
        "A step appears in multiple feature split artifacts",
    )
    _require(
        split_stats["train"]["step_max"] < split_stats["validation"]["step_min"]
        and split_stats["validation"]["step_max"]
        < split_stats["test"]["step_min"],
        "Feature split artifacts are not chronological",
    )
    ratios = [split_stats[name]["row_ratio"] for name in SPLIT_NAMES]
    _require(
        all(abs(actual - target) <= 0.03 for actual, target in zip(ratios, (0.6, 0.2, 0.2))),
        f"Split ratios are not near 60/20/20: {ratios}",
    )
    total_fraud = sum(split_stats[name]["fraud_count"] for name in SPLIT_NAMES)
    _require(
        total_fraud == int(metadata["fraud_count"]),
        "Feature split fraud counts do not match dataset metadata",
    )

    return {
        "status": "passed",
        "artifacts_dir": str(artifacts.resolve()),
        "dataset_version": metadata["dataset_version"],
        "split_manifest_version": metadata["split_manifest_version"],
        "feature_configuration": selected_payload["feature_configuration"],
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "forbidden_selected_features": forbidden,
        "post_transaction_selected_features": post_transaction,
        "metadata_columns_present": ["row_id", "split", "step", "label", "rule_score", "amount"],
        "schema_and_feature_order_match": True,
        "row_id_unique_and_non_overlapping": True,
        "step_atomic_and_non_overlapping": True,
        "strictly_chronological": True,
        "total_row_count": len(all_ids),
        "total_fraud_count": total_fraud,
        "splits": split_stats,
    }


def _assert_metrics_match(
    system: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for name, expected_value in expected.items():
        _require(name in actual, f"Recomputed {system} metrics lack {name}")
        actual_value = actual[name]
        if expected_value is None or actual_value is None:
            _require(
                expected_value is actual_value,
                f"{system} metric {name} differs: {actual_value} != {expected_value}",
            )
        elif isinstance(expected_value, (int, np.integer)):
            _require(
                int(actual_value) == int(expected_value),
                f"{system} metric {name} differs: {actual_value} != {expected_value}",
            )
        else:
            _require(
                bool(
                    np.isclose(
                        float(actual_value),
                        float(expected_value),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ),
                f"{system} metric {name} differs: {actual_value} != {expected_value}",
            )


def verify_final_artifacts(
    artifacts_dir: str | Path,
    *,
    score_batch_size: int = 10_000,
) -> dict[str, Any]:
    from model.model_utils import load_model_artifact, predict_frame, reset_artifact_cache
    from model.train_models import evaluate_scores, validate_test_prediction_consistency

    artifacts = Path(artifacts_dir)
    missing_artifacts = [
        name for name in FINAL_REQUIRED_ARTIFACTS if not (artifacts / name).exists()
    ]
    _require(not missing_artifacts, f"Missing final artifacts: {missing_artifacts}")

    selected_payload = _read_json(artifacts / "selected_features.json")
    metadata = _read_json(artifacts / "dataset_metadata.json")
    comparison = _read_json(artifacts / "model_comparison.json")
    evaluation = _read_json(artifacts / "evaluation_results.json")
    selected_features = [str(name) for name in selected_payload["selected_features"]]
    selected_tag = str(comparison["selected_model_tag"])
    model_tags = [str(row["model_tag"]) for row in comparison["models"]]
    _require(
        set(model_tags) == {"logreg", "rf", "xgb", "lgbm"} and len(model_tags) == 4,
        f"Model comparison does not contain exactly four model types: {model_tags}",
    )
    selected_rows = [
        row for row in comparison["models"] if str(row["model_tag"]) == selected_tag
    ]
    _require(len(selected_rows) == 1, "Selected model is absent or duplicated in comparison")
    minimum_cost = min(
        float(row["validation_hybrid"]["business_cost"])
        for row in comparison["models"]
    )
    _require(
        np.isclose(
            float(selected_rows[0]["validation_hybrid"]["business_cost"]),
            minimum_cost,
            rtol=0.0,
            atol=0.0,
        ),
        "Selected model does not have minimum validation hybrid business cost",
    )

    artifact_path = artifacts / "fraud_pipeline_selected.joblib"
    raw_artifact = joblib.load(artifact_path)
    reset_artifact_cache()
    strict_artifact = load_model_artifact(
        artifact_path,
        strict=True,
        force_reload=True,
    )
    _require(strict_artifact is not None, "Strict artifact loader returned no artifact")
    _require(strict_artifact["model_tag"] == selected_tag, "Artifact model tag mismatch")
    _require(
        strict_artifact["feature_configuration"] == "deployment_safe",
        "Atomic artifact is not deployment_safe",
    )
    _require(
        list(strict_artifact["feature_columns"]) == selected_features,
        "Atomic artifact feature order differs from selected_features.json",
    )
    _require(
        list(evaluation["feature_columns"]) == selected_features,
        "Evaluation feature order differs from selected_features.json",
    )
    _require(
        hasattr(strict_artifact["preprocessor"], "transform")
        and hasattr(strict_artifact["model"], "predict_proba"),
        "Atomic artifact lacks fitted preprocessing or predict_proba model",
    )
    _require(
        strict_artifact["rule_config"] == metadata["rule_config"],
        "Atomic artifact rule configuration differs from dataset metadata",
    )
    rule_weight = float(strict_artifact["rule_weight"])
    ml_weight = float(strict_artifact["ml_weight"])
    _require(
        np.isclose(rule_weight, 0.6) and np.isclose(ml_weight, 0.4),
        f"Unexpected hybrid weights: rule={rule_weight}, ml={ml_weight}",
    )
    _require(
        np.isclose(rule_weight + ml_weight, 1.0),
        "Hybrid weights do not sum to one",
    )
    threshold_names = ("rule_threshold", "ml_threshold", "hybrid_threshold")
    for name in threshold_names:
        _require(name in strict_artifact, f"Atomic artifact lacks {name}")
        _require(
            np.isfinite(float(strict_artifact[name])),
            f"Atomic artifact {name} is not finite",
        )

    pointer_path = artifacts.parent / "selected_model.json"
    _require(pointer_path.exists(), f"Runtime pointer is missing: {pointer_path}")
    pointer = _read_json(pointer_path)
    pointer_artifact = Path(str(pointer["artifact_path"]))
    if not pointer_artifact.is_absolute():
        pointer_artifact = pointer_path.parent / pointer_artifact
    _require(
        pointer_artifact.resolve() == artifact_path.resolve(),
        "Runtime pointer does not resolve to the selected atomic artifact",
    )
    _require(pointer["model_tag"] == selected_tag, "Runtime pointer model tag mismatch")

    predictions = pd.read_parquet(artifacts / "test_predictions.parquet")
    expected_test_rows = int(metadata["split_counts"]["test"])
    _require(
        len(predictions) == expected_test_rows,
        f"test_predictions has {len(predictions)} rows, expected {expected_test_rows}",
    )
    _require(
        not predictions["row_id"].isna().any()
        and not predictions["row_id"].duplicated().any(),
        "test_predictions row_id is null or duplicated",
    )
    test_ids = pd.read_parquet(
        artifacts / "test_features.parquet", columns=["row_id"]
    )["row_id"].to_numpy(dtype=np.int64)
    prediction_ids = predictions["row_id"].to_numpy(dtype=np.int64)
    _require(
        np.array_equal(np.sort(test_ids), np.sort(prediction_ids)),
        "test_predictions row_id values do not exactly cover the test split",
    )

    rule_scores = predictions["rule_score"].to_numpy(dtype=np.float64)
    ml_scores = predictions["ml_score"].to_numpy(dtype=np.float64)
    hybrid_scores = predictions["hybrid_score"].to_numpy(dtype=np.float64)
    reconstructed_hybrid = rule_weight * rule_scores + ml_weight * ml_scores
    _require(
        bool(np.allclose(hybrid_scores, reconstructed_hybrid, rtol=1e-12, atol=1e-12)),
        "Persisted hybrid scores do not match frozen rule/ML weights",
    )
    persisted_thresholds = predictions["threshold"].astype(float).unique()
    _require(len(persisted_thresholds) == 1, "Predictions contain multiple thresholds")
    _require(
        np.isclose(
            float(persisted_thresholds[0]),
            float(strict_artifact["hybrid_threshold"]),
            rtol=0.0,
            atol=0.0,
        ),
        "Persisted threshold differs from atomic artifact hybrid threshold",
    )
    reconstructed_predictions = (
        hybrid_scores >= float(strict_artifact["hybrid_threshold"])
    ).astype(np.int8)
    _require(
        np.array_equal(
            reconstructed_predictions,
            predictions["prediction"].to_numpy(dtype=np.int8),
        ),
        "Persisted decisions do not match hybrid score and threshold",
    )

    labels = predictions["label"].to_numpy(dtype=np.int8)
    amounts = predictions["amount"].to_numpy(dtype=np.float64)
    false_alarm_cost = float(evaluation["false_alarm_unit_cost"])
    system_scores = {
        "rule_only": (rule_scores, float(strict_artifact["rule_threshold"])),
        "ml_only": (ml_scores, float(strict_artifact["ml_threshold"])),
        "hybrid": (hybrid_scores, float(strict_artifact["hybrid_threshold"])),
    }
    recomputed_metrics: dict[str, dict[str, Any]] = {}
    for system, (scores, threshold) in system_scores.items():
        expected_metrics = evaluation["evaluation"][system]
        actual_metrics = evaluate_scores(
            labels,
            scores,
            amounts,
            threshold,
            false_alarm_unit_cost=false_alarm_cost,
            training_time=float(expected_metrics["training_time"]),
        )
        _assert_metrics_match(system, actual_metrics, expected_metrics)
        recomputed_metrics[system] = actual_metrics
    validate_test_prediction_consistency(
        predictions,
        evaluation["evaluation"]["hybrid"],
        false_alarm_cost,
    )

    batch_columns = ["row_id", *selected_features]
    first_batch = next(
        pq.ParquetFile(artifacts / "test_features.parquet").iter_batches(
            batch_size=score_batch_size,
            columns=batch_columns,
        )
    ).to_pandas()
    raw_scores = predict_frame(raw_artifact, first_batch)
    reloaded_scores = predict_frame(strict_artifact, first_batch)
    _require(
        bool(np.allclose(raw_scores, reloaded_scores, rtol=1e-10, atol=1e-12)),
        "Batch ML scores changed after strict reload",
    )
    persisted_batch = predictions.iloc[: len(first_batch)]
    _require(
        np.array_equal(
            first_batch["row_id"].to_numpy(dtype=np.int64),
            persisted_batch["row_id"].to_numpy(dtype=np.int64),
        ),
        "First persisted prediction batch is not aligned with test features",
    )
    _require(
        bool(
            np.allclose(
                reloaded_scores,
                persisted_batch["ml_score"].to_numpy(dtype=np.float64),
                rtol=1e-10,
                atol=1e-12,
            )
        ),
        "Reloaded batch ML scores differ from persisted test predictions",
    )

    return {
        "status": "passed",
        "verified_at": datetime.now().astimezone().isoformat(),
        "artifacts_dir": str(artifacts.resolve()),
        "required_artifacts": [str(artifacts / name) for name in FINAL_REQUIRED_ARTIFACTS],
        "model_tag": selected_tag,
        "model_version": strict_artifact["model_version"],
        "feature_configuration": strict_artifact["feature_configuration"],
        "ordered_feature_count": len(selected_features),
        "ordered_features": selected_features,
        "preprocessor_type": type(strict_artifact["preprocessor"]).__name__,
        "model_type": type(strict_artifact["model"]).__name__,
        "thresholds": {
            name: float(strict_artifact[name]) for name in threshold_names
        },
        "hybrid_weights": {"rule_weight": rule_weight, "ml_weight": ml_weight},
        "rule_config_match": True,
        "four_model_validation_comparison_complete": True,
        "selected_by_minimum_validation_hybrid_business_cost": True,
        "runtime_pointer_match": True,
        "strict_artifact_reload_passed": True,
        "reload_batch_size": len(first_batch),
        "reload_prediction_match": True,
        "persisted_batch_prediction_match": True,
        "test_prediction_row_count": len(predictions),
        "test_prediction_row_ids_unique_and_complete": True,
        "hybrid_formula_match": True,
        "frozen_threshold_decision_match": True,
        "confusion_and_business_metrics_match": True,
        "recomputed_test_metrics": recomputed_metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate prepared fraud feature artifacts")
    parser.add_argument("--artifacts-dir", default="model/artifacts")
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--score-batch-size", type=int, default=10_000)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.score_batch_size <= 0:
        raise ValueError("--score-batch-size must be positive")
    if args.final:
        result = verify_final_artifacts(
            args.artifacts_dir,
            score_batch_size=args.score_batch_size,
        )
        output_path = Path(args.artifacts_dir) / "final_artifact_verification.json"
        _write_json(output_path, result)
        print(
            f"Verified atomic {result['model_tag']} artifact and "
            f"{result['test_prediction_row_count']:,} test predictions: {output_path}"
        )
    else:
        result = validate_feature_artifacts(
            args.artifacts_dir,
            batch_size=args.batch_size,
        )
        output_path = Path(args.artifacts_dir) / "feature_artifact_validation.json"
        _write_json(output_path, result)
        print(
            f"Validated {result['total_row_count']:,} rows, "
            f"{result['selected_feature_count']} selected features: {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
