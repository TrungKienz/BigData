from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fraud_pipeline.features import (
    CATEGORICAL_FEATURES,
    FEATURE_GROUPS,
    FORBIDDEN_FEATURE_COLUMNS,
    FULL_PAYSIM_FEATURES,
    assert_no_forbidden_features,
)
from fraud_pipeline.rules import (
    DEFAULT_ML_WEIGHT,
    DEFAULT_RULE_WEIGHT,
    combine_risk_score_arrays,
)
from model.model_utils import (
    load_model_artifact,
    predict_frame,
    reset_artifact_cache,
)


DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_ORDER = ("logreg", "rf", "xgb", "lgbm")
FALSE_ALARM_SENSITIVITY_COSTS = (1.0, 5.0, 10.0)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def parse_model_types(raw: str) -> list[str]:
    values = MODEL_ORDER if raw.strip().lower() == "all" else tuple(
        value.strip().lower() for value in raw.split(",") if value.strip()
    )
    unknown = sorted(set(values).difference(MODEL_ORDER))
    if unknown:
        raise ValueError(
            f"Unknown model types {unknown}; valid values are all, {', '.join(MODEL_ORDER)}"
        )
    return list(dict.fromkeys(values))


def calculate_scale_pos_weight(labels: Sequence[int]) -> float:
    values = np.asarray(labels, dtype=np.int8)
    positives = int(values.sum())
    negatives = int(len(values) - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("Training labels must contain both fraud and non-fraud rows")
    return float(negatives / positives)


def build_preprocessor(
    feature_columns: Sequence[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    features = [str(name) for name in feature_columns]
    assert_no_forbidden_features(features)
    categorical = [name for name in features if name in CATEGORICAL_FEATURES]
    numeric = [name for name in features if name not in CATEGORICAL_FEATURES]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    if not transformers:
        raise ValueError("Cannot build a preprocessor without feature columns")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def build_estimator(
    model_type: str,
    train_labels: Sequence[int],
    *,
    random_state: int = 42,
    quick: bool = False,
) -> tuple[Any, dict[str, Any]]:
    scale_pos_weight = calculate_scale_pos_weight(train_labels)
    if model_type == "logreg":
        estimator = LogisticRegression(
            max_iter=200 if quick else 1_000,
            class_weight="balanced",
            solver="saga",
            random_state=random_state,
        )
        strategy = {
            "method": "class_weight",
            "class_weight": "balanced",
            "smote": False,
        }
    elif model_type == "rf":
        estimator = RandomForestClassifier(
            n_estimators=30 if quick else 200,
            max_depth=12 if quick else 20,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        strategy = {
            "method": "class_weight",
            "class_weight": "balanced_subsample",
            "smote": False,
        }
    elif model_type == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost is requested but xgboost is not installed") from exc
        estimator = XGBClassifier(
            n_estimators=50 if quick else 300,
            max_depth=6,
            learning_rate=0.1 if quick else 0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
        )
        strategy = {
            "method": "scale_pos_weight",
            "scale_pos_weight": scale_pos_weight,
            "source_distribution": "original_training_split",
            "smote": False,
        }
    elif model_type == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM is requested but lightgbm is not installed") from exc
        estimator = LGBMClassifier(
            n_estimators=75 if quick else 300,
            num_leaves=31,
            learning_rate=0.1 if quick else 0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        strategy = {
            "method": "scale_pos_weight",
            "scale_pos_weight": scale_pos_weight,
            "source_distribution": "original_training_split",
            "smote": False,
        }
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")
    return estimator, strategy


def business_cost_breakdown(
    labels: Sequence[int],
    predictions: Sequence[int],
    amounts: Sequence[float],
    false_alarm_unit_cost: float = 5.0,
) -> dict[str, float]:
    if false_alarm_unit_cost < 0:
        raise ValueError("false_alarm_unit_cost must be non-negative")
    y_true = np.asarray(labels, dtype=np.int8)
    y_pred = np.asarray(predictions, dtype=np.int8)
    amount_values = np.asarray(amounts, dtype=np.float64)
    if not (len(y_true) == len(y_pred) == len(amount_values)):
        raise ValueError("labels, predictions, and amounts must have equal length")
    false_negative_mask = (y_true == 1) & (y_pred == 0)
    false_positive_mask = (y_true == 0) & (y_pred == 1)
    missed_fraud_cost = float(amount_values[false_negative_mask].sum())
    false_alarm_cost = float(false_alarm_unit_cost * false_positive_mask.sum())
    business_cost = float(missed_fraud_cost + false_alarm_cost)
    no_model_baseline_cost = float(amount_values[y_true == 1].sum())
    net_cost_savings = float(no_model_baseline_cost - business_cost)
    savings_rate = float(
        net_cost_savings / no_model_baseline_cost
        if no_model_baseline_cost > 0
        else 0.0
    )
    return {
        "business_cost": business_cost,
        "missed_fraud_cost": missed_fraud_cost,
        "false_alarm_cost": false_alarm_cost,
        "no_model_baseline_cost": no_model_baseline_cost,
        "net_cost_savings": net_cost_savings,
        "savings_rate": savings_rate,
    }


def tune_threshold_by_business_cost(
    validation_labels: Sequence[int],
    validation_scores: Sequence[float],
    validation_amounts: Sequence[float],
    false_alarm_unit_cost: float = 5.0,
) -> dict[str, Any]:
    """Select a threshold using validation data only."""

    labels = np.asarray(validation_labels, dtype=np.int8)
    scores = np.asarray(validation_scores, dtype=np.float64)
    amounts = np.asarray(validation_amounts, dtype=np.float64)
    if not (len(labels) == len(scores) == len(amounts)):
        raise ValueError("Validation labels, scores, and amounts must have equal length")
    if len(labels) == 0:
        raise ValueError("Validation split is empty")
    if not np.isfinite(scores).all():
        raise ValueError("Validation scores contain NaN or infinite values")

    order = np.argsort(scores, kind="stable")[::-1]
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ordered_amounts = amounts[order]
    baseline = float(ordered_amounts[ordered_labels == 1].sum())
    detected = np.cumsum(
        np.where(ordered_labels == 1, ordered_amounts, 0.0), dtype=np.float64
    )
    false_positives = np.cumsum((ordered_labels == 0).astype(np.int64))
    change_indices = np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:])
    end_indices = np.r_[change_indices, len(ordered_scores) - 1]
    thresholds = ordered_scores[end_indices]
    costs = (
        baseline
        - detected[end_indices]
        + false_alarm_unit_cost * false_positives[end_indices]
    )
    no_alert_threshold = float(np.nextafter(float(ordered_scores[0]), np.inf))
    all_thresholds = np.r_[no_alert_threshold, thresholds]
    all_costs = np.r_[baseline, costs]
    best_index = int(np.argmin(all_costs))
    threshold = float(all_thresholds[best_index])
    predictions = (scores >= threshold).astype(np.int8)
    return {
        "threshold": threshold,
        "evaluated_thresholds": int(len(all_thresholds)),
        **business_cost_breakdown(
            labels, predictions, amounts, false_alarm_unit_cost
        ),
    }


def evaluate_scores(
    labels: Sequence[int],
    scores: Sequence[float],
    amounts: Sequence[float],
    threshold: float,
    *,
    false_alarm_unit_cost: float = 5.0,
    training_time: float = 0.0,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int8)
    score_values = np.asarray(scores, dtype=np.float64)
    predictions = (score_values >= threshold).astype(np.int8)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    negative_count = tn + fp
    positive_count = tp + fn
    metrics: dict[str, Any] = {
        "average_precision": float(
            average_precision_score(y_true, score_values)
            if positive_count > 0
            else 0.0
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, score_values))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "false_positive_rate": float(fp / negative_count if negative_count else 0.0),
        "false_negative_rate": float(fn / positive_count if positive_count else 0.0),
        "alert_rate": float(predictions.mean() if len(predictions) else 0.0),
        "manual_review_rate": float(
            predictions.mean() if len(predictions) else 0.0
        ),
        "training_time": float(training_time),
        "selected_threshold": float(threshold),
    }
    metrics.update(
        business_cost_breakdown(
            y_true, predictions, amounts, false_alarm_unit_cost
        )
    )
    return metrics


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required feature artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_artifacts(
    artifacts_dir: str | Path,
    splits: Sequence[str] = ("train", "validation", "test"),
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(artifacts_dir)
    selected = _load_json(root / "selected_features.json")
    schema = _load_json(root / "feature_schema.json")
    metadata = _load_json(root / "dataset_metadata.json")
    features = [str(name) for name in selected["selected_features"]]
    assert_no_forbidden_features(features)
    columns = list(
        dict.fromkeys(
            ["row_id", "split", "step", "label", "rule_score", "amount", *features]
        )
    )
    frames = {}
    unknown_splits = sorted(set(splits).difference({"train", "validation", "test"}))
    if unknown_splits:
        raise ValueError(f"Unknown feature artifact splits requested: {unknown_splits}")
    for split in splits:
        path = root / f"{split}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Required feature artifact not found: {path}. Run model.prepare_features first."
            )
        try:
            frame = pd.read_parquet(path, columns=columns)
        except Exception as exc:
            raise ValueError(
                f"Could not read required selected feature columns from {path}: {exc}"
            ) from exc
        if set(frame["split"].unique()) != {split}:
            raise ValueError(f"{path.name} contains rows outside split={split!r}")
        if frame["row_id"].duplicated().any():
            raise ValueError(f"{path.name} contains duplicate row_id values")
        frames[split] = frame

    row_sets = [set(frame["row_id"]) for frame in frames.values()]
    if any(
        row_sets[left] & row_sets[right]
        for left in range(len(row_sets))
        for right in range(left + 1, len(row_sets))
    ):
        raise ValueError("Feature split artifacts contain overlapping row_id values")

    missing_by_split = {
        split: [name for name in features if name not in frame.columns]
        for split, frame in frames.items()
    }
    missing_by_split = {
        split: names for split, names in missing_by_split.items() if names
    }
    if missing_by_split:
        raise ValueError(f"Selected features missing from split artifacts: {missing_by_split}")
    return frames, selected, schema, metadata


def sanity_check_feature_artifacts(
    frames: Mapping[str, pd.DataFrame], feature_columns: Sequence[str]
) -> dict[str, Any]:
    report: dict[str, Any] = {"feature_count": len(feature_columns), "splits": {}}
    for split, frame in frames.items():
        numeric_features = [
            name for name in feature_columns if name not in CATEGORICAL_FEATURES
        ]
        infinite_count = 0
        for name in numeric_features:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(
                dtype=np.float64, copy=False
            )
            infinite_count += int(np.isinf(values).sum())
        if infinite_count:
            raise ValueError(f"{split} features contain {infinite_count} infinite values")
        report["splits"][split] = {
            "row_count": len(frame),
            "fraud_count": int(frame["label"].sum()),
            "fraud_rate": float(frame["label"].mean()),
            "missing_values": int(
                sum(int(frame[name].isna().sum()) for name in feature_columns)
            ),
            "infinite_values": infinite_count,
        }
    if any(
        set(frame["label"].astype(int).unique()) != {0, 1}
        for frame in frames.values()
    ):
        raise ValueError("Every chronological split must contain both fraud classes")
    return report


def _candidate_bundle(
    model_type: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    false_alarm_unit_cost: float,
    random_state: int,
    quick: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels_train = train["label"].to_numpy(dtype=np.int8)
    labels_validation = validation["label"].to_numpy(dtype=np.int8)
    amounts_validation = validation["amount"].to_numpy(dtype=np.float64)
    rule_scores_validation = validation["rule_score"].to_numpy(dtype=np.float64)
    scale_numeric = model_type == "logreg"
    preprocessor = build_preprocessor(
        feature_columns, scale_numeric=scale_numeric
    )
    estimator, imbalance_strategy = build_estimator(
        model_type, labels_train, random_state=random_state, quick=quick
    )

    started = time.perf_counter()
    transformed_train = preprocessor.fit_transform(train[list(feature_columns)])
    estimator.fit(transformed_train, labels_train)
    del transformed_train
    training_time = float(time.perf_counter() - started)
    transformed_validation = preprocessor.transform(validation[list(feature_columns)])
    ml_scores = np.asarray(estimator.predict_proba(transformed_validation))[:, 1]
    hybrid_scores = combine_risk_score_arrays(
        rule_scores_validation,
        ml_scores,
        DEFAULT_RULE_WEIGHT,
        DEFAULT_ML_WEIGHT,
    )

    ml_tuning = tune_threshold_by_business_cost(
        labels_validation,
        ml_scores,
        amounts_validation,
        false_alarm_unit_cost,
    )
    hybrid_tuning = tune_threshold_by_business_cost(
        labels_validation,
        hybrid_scores,
        amounts_validation,
        false_alarm_unit_cost,
    )
    rule_tuning = tune_threshold_by_business_cost(
        labels_validation,
        rule_scores_validation,
        amounts_validation,
        false_alarm_unit_cost,
    )
    validation_metrics = {
        "rule_only": evaluate_scores(
            labels_validation,
            rule_scores_validation,
            amounts_validation,
            float(rule_tuning["threshold"]),
            false_alarm_unit_cost=false_alarm_unit_cost,
        ),
        "ml_only": evaluate_scores(
            labels_validation,
            ml_scores,
            amounts_validation,
            float(ml_tuning["threshold"]),
            false_alarm_unit_cost=false_alarm_unit_cost,
            training_time=training_time,
        ),
        "hybrid": evaluate_scores(
            labels_validation,
            hybrid_scores,
            amounts_validation,
            float(hybrid_tuning["threshold"]),
            false_alarm_unit_cost=false_alarm_unit_cost,
            training_time=training_time,
        ),
    }
    bundle = {
        "preprocessor": preprocessor,
        "model": estimator,
        "feature_columns": list(feature_columns),
        "model_tag": model_type,
        "ml_threshold": float(ml_tuning["threshold"]),
        "hybrid_threshold": float(hybrid_tuning["threshold"]),
        "rule_threshold": float(rule_tuning["threshold"]),
        "training_time": training_time,
        "imbalance_strategy": imbalance_strategy,
    }
    comparison = {
        "model_tag": model_type,
        "training_time": training_time,
        "imbalance_strategy": imbalance_strategy,
        "ml_threshold": bundle["ml_threshold"],
        "hybrid_threshold": bundle["hybrid_threshold"],
        "validation_rule_only": validation_metrics["rule_only"],
        "validation_ml_only": validation_metrics["ml_only"],
        "validation_hybrid": validation_metrics["hybrid"],
    }
    return bundle, comparison


def _selection_key(comparison: Mapping[str, Any]) -> tuple[float, float, str]:
    hybrid = comparison["validation_hybrid"]
    return (
        float(hybrid["business_cost"]),
        -float(hybrid["average_precision"]),
        str(comparison["model_tag"]),
    )


def _sample_training_rows(
    frame: pd.DataFrame, max_rows: int | None, random_state: int
) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    positives = frame[frame["label"] == 1]
    negatives = frame[frame["label"] == 0]
    positive_limit = min(len(positives), max(2, max_rows // 5))
    negative_limit = max_rows - positive_limit
    sampled = pd.concat(
        [
            positives.sample(
                n=positive_limit, random_state=random_state, replace=False
            ),
            negatives.sample(
                n=min(negative_limit, len(negatives)),
                random_state=random_state,
                replace=False,
            ),
        ]
    )
    return sampled.sort_values("row_id", kind="stable")


def run_ablation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    false_alarm_unit_cost: float,
    random_state: int,
    max_train_rows: int | None,
    model_type: str = "logreg",
) -> pd.DataFrame:
    sampled_train = _sample_training_rows(train, max_train_rows, random_state)
    rows: list[dict[str, Any]] = []
    for feature_set, configured in FEATURE_GROUPS.items():
        features = [
            name
            for name in configured
            if name in train.columns
            and name not in FORBIDDEN_FEATURE_COLUMNS
            and train[name].nunique(dropna=False) > 1
        ]
        if not features:
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_count": 0,
                    "status": "skipped_no_nonconstant_features",
                }
            )
            continue
        rows.append(
            _evaluate_ablation_set(
                feature_set,
                features,
                sampled_train,
                validation,
                false_alarm_unit_cost=false_alarm_unit_cost,
                random_state=random_state,
                model_type=model_type,
            )
        )
    return pd.DataFrame(rows)


def _evaluate_ablation_set(
    feature_set: str,
    features: Sequence[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    false_alarm_unit_cost: float,
    random_state: int,
    model_type: str = "logreg",
) -> dict[str, Any]:
    preprocessor = build_preprocessor(
        features,
        scale_numeric=model_type == "logreg",
    )
    if model_type == "logreg":
        estimator = LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            solver="saga",
            random_state=random_state,
        )
    else:
        estimator, _ = build_estimator(
            model_type,
            train["label"].to_numpy(dtype=np.int8),
            random_state=random_state,
            quick=False,
        )
    started = time.perf_counter()
    transformed_train = preprocessor.fit_transform(train[list(features)])
    estimator.fit(transformed_train, train["label"].to_numpy(dtype=np.int8))
    elapsed = time.perf_counter() - started
    transformed_validation = preprocessor.transform(validation[list(features)])
    ml_scores = np.asarray(estimator.predict_proba(transformed_validation))[:, 1]
    hybrid_scores = combine_risk_score_arrays(
        validation["rule_score"].to_numpy(dtype=np.float64), ml_scores
    )
    tuned = tune_threshold_by_business_cost(
        validation["label"],
        hybrid_scores,
        validation["amount"],
        false_alarm_unit_cost,
    )
    metrics = evaluate_scores(
        validation["label"],
        hybrid_scores,
        validation["amount"],
        float(tuned["threshold"]),
        false_alarm_unit_cost=false_alarm_unit_cost,
        training_time=elapsed,
    )
    return {
        "feature_set": feature_set,
        "model_tag": model_type,
        "feature_count": len(features),
        "status": "completed",
        "validation_average_precision": metrics["average_precision"],
        "validation_roc_auc": metrics["roc_auc"],
        "validation_business_cost": metrics["business_cost"],
        "validation_f1": metrics["f1"],
        "selected_hybrid_threshold": metrics["selected_threshold"],
        "training_time": elapsed,
    }


def _deterministic_training_sample(
    parquet_path: Path,
    columns: Sequence[str],
    max_rows: int,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    if parquet.metadata.num_rows <= max_rows:
        return pd.read_parquet(parquet_path, columns=list(columns))
    modulus = 1_000_003
    threshold = max(
        1,
        int(modulus * max_rows * 1.1 / parquet.metadata.num_rows),
    )
    chunks: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(batch_size=100_000, columns=list(columns)):
        current = batch.to_pandas()
        hashes = pd.util.hash_pandas_object(
            current["row_id"], index=False
        ).to_numpy(dtype=np.uint64)
        selected = current.loc[(hashes % modulus) < threshold]
        if not selected.empty:
            chunks.append(selected)
    sample = pd.concat(chunks, ignore_index=True)
    if len(sample) > max_rows:
        sample = sample.sort_values("row_id", kind="stable").iloc[:max_rows]
    if set(sample["label"].astype(int).unique()) != {0, 1}:
        raise ValueError(
            "Deterministic ablation sample must contain both fraud classes; "
            "increase --ablation-max-rows"
        )
    return sample.reset_index(drop=True)


def run_ablation_from_artifacts(
    artifacts: Path,
    *,
    false_alarm_unit_cost: float,
    random_state: int,
    max_train_rows: int,
    model_type: str = "logreg",
) -> pd.DataFrame:
    metadata_columns = ["row_id", "label", "amount", "rule_score"]
    train_columns = list(dict.fromkeys([*metadata_columns, *FULL_PAYSIM_FEATURES]))
    sampled_train = _deterministic_training_sample(
        artifacts / "train_features.parquet",
        train_columns,
        max_train_rows,
    )
    rows: list[dict[str, Any]] = []
    for feature_set, configured in FEATURE_GROUPS.items():
        features = [
            name
            for name in configured
            if name in sampled_train.columns
            and name not in FORBIDDEN_FEATURE_COLUMNS
            and sampled_train[name].nunique(dropna=False) > 1
        ]
        if not features:
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_count": 0,
                    "status": "skipped_no_nonconstant_features",
                }
            )
            continue
        validation_columns = list(
            dict.fromkeys(["label", "amount", "rule_score", *features])
        )
        validation = pd.read_parquet(
            artifacts / "validation_features.parquet",
            columns=validation_columns,
        )
        rows.append(
            _evaluate_ablation_set(
                feature_set,
                features,
                sampled_train,
                validation,
                false_alarm_unit_cost=false_alarm_unit_cost,
                random_state=random_state,
                model_type=model_type,
            )
        )
        del validation
        gc.collect()
    return pd.DataFrame(rows)


def validate_test_prediction_consistency(
    predictions: pd.DataFrame,
    expected_metrics: Mapping[str, Any],
    false_alarm_unit_cost: float = 5.0,
) -> None:
    required = {
        "label",
        "amount",
        "hybrid_score",
        "threshold",
        "prediction",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Test prediction artifact is missing columns: {missing}")
    thresholds = predictions["threshold"].astype(float).unique()
    if len(thresholds) != 1:
        raise ValueError("Test prediction artifact must contain one frozen threshold")
    threshold = float(thresholds[0])
    reconstructed_predictions = (
        predictions["hybrid_score"].to_numpy(dtype=np.float64) >= threshold
    ).astype(np.int8)
    persisted_predictions = predictions["prediction"].to_numpy(dtype=np.int8)
    if not np.array_equal(reconstructed_predictions, persisted_predictions):
        raise ValueError("Persisted predictions do not match hybrid_score >= threshold")
    reconstructed = evaluate_scores(
        predictions["label"].to_numpy(dtype=np.int8),
        predictions["hybrid_score"].to_numpy(dtype=np.float64),
        predictions["amount"].to_numpy(dtype=np.float64),
        threshold,
        false_alarm_unit_cost=false_alarm_unit_cost,
    )
    integer_metrics = ("TP", "FP", "TN", "FN")
    for name in integer_metrics:
        if int(reconstructed[name]) != int(expected_metrics[name]):
            raise ValueError(
                f"test_predictions {name}={reconstructed[name]} does not match "
                f"evaluation {expected_metrics[name]}"
            )
    float_metrics = (
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "alert_rate",
        "manual_review_rate",
        "business_cost",
        "missed_fraud_cost",
        "false_alarm_cost",
        "no_model_baseline_cost",
        "net_cost_savings",
        "savings_rate",
        "selected_threshold",
    )
    for name in float_metrics:
        actual = reconstructed[name]
        expected = expected_metrics[name]
        if actual is None or expected is None:
            if actual != expected:
                raise ValueError(f"test_predictions metric {name} does not match evaluation")
        elif not np.isclose(float(actual), float(expected), rtol=1e-12, atol=1e-12):
            raise ValueError(
                f"test_predictions {name}={actual} does not match evaluation {expected}"
            )


def _dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "xgboost", "lightgbm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def train_and_export_models(
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    *,
    model_types: Sequence[str] = MODEL_ORDER,
    false_alarm_unit_cost: float = 5.0,
    random_state: int = 42,
    quick: bool = False,
    run_sensitivity_analysis: bool = True,
    run_feature_ablation: bool = True,
    ablation_max_rows: int | None = 500_000,
    update_runtime_pointer: bool = True,
) -> dict[str, Any]:
    if false_alarm_unit_cost < 0:
        raise ValueError("false_alarm_unit_cost must be non-negative")
    requested = parse_model_types(",".join(model_types))
    artifacts = Path(artifacts_dir)
    frames, selected_config, feature_schema, dataset_metadata = load_feature_artifacts(
        artifacts, splits=("train", "validation")
    )
    if selected_config["feature_configuration"] != "deployment_safe":
        raise ValueError(
            "Production artifact export requires feature_configuration='deployment_safe'. "
            "Use full_paysim only for analysis/ablation."
        )
    features = [str(name) for name in selected_config["selected_features"]]
    sanity = sanity_check_feature_artifacts(frames, features)
    train = frames["train"]
    validation = frames["validation"]

    candidate_dir = artifacts / ".training_candidates"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True)
    comparisons: list[dict[str, Any]] = []
    candidate_paths: dict[str, Path] = {}
    try:
        for model_type in requested:
            print(f"Training candidate model: {model_type}", flush=True)
            bundle, comparison = _candidate_bundle(
                model_type,
                train,
                validation,
                features,
                false_alarm_unit_cost=false_alarm_unit_cost,
                random_state=random_state,
                quick=quick,
            )
            candidate_path = candidate_dir / f"{model_type}.joblib"
            joblib.dump(bundle, candidate_path)
            candidate_paths[model_type] = candidate_path
            comparisons.append(comparison)
            print(
                f"Completed candidate model: {model_type} "
                f"in {comparison['training_time']:.2f}s",
                flush=True,
            )
            del bundle
            gc.collect()

        selected_comparison = min(comparisons, key=_selection_key)
        selected_tag = str(selected_comparison["model_tag"])
        selected_bundle = joblib.load(candidate_paths[selected_tag])
    finally:
        if not comparisons and candidate_dir.exists():
            shutil.rmtree(candidate_dir)

    validation_labels = validation["label"].to_numpy(dtype=np.int8)
    validation_amounts = validation["amount"].to_numpy(dtype=np.float64)
    validation_rule_scores = validation["rule_score"].to_numpy(dtype=np.float64)
    validation_ml_scores = predict_frame(selected_bundle, validation)
    validation_hybrid_scores = combine_risk_score_arrays(
        validation_rule_scores,
        validation_ml_scores,
        DEFAULT_RULE_WEIGHT,
        DEFAULT_ML_WEIGHT,
    )

    if run_sensitivity_analysis:
        sensitivity_rows: list[dict[str, Any]] = []
        for unit_cost in FALSE_ALARM_SENSITIVITY_COSTS:
            for system, scores in (
                ("rule_only", validation_rule_scores),
                ("ml_only", validation_ml_scores),
                ("hybrid", validation_hybrid_scores),
            ):
                tuned = tune_threshold_by_business_cost(
                    validation_labels,
                    scores,
                    validation_amounts,
                    unit_cost,
                )
                sensitivity_rows.append(
                    {
                        "system": system,
                        "false_alarm_unit_cost": unit_cost,
                        "validation_threshold": tuned["threshold"],
                        "validation_business_cost": tuned["business_cost"],
                        "validation_missed_fraud_cost": tuned["missed_fraud_cost"],
                        "validation_false_alarm_cost": tuned["false_alarm_cost"],
                        "validation_savings_rate": tuned["savings_rate"],
                    }
                )
        pd.DataFrame(sensitivity_rows).to_csv(
            artifacts / "sensitivity_analysis.csv", index=False
        )

    del (
        frames,
        train,
        validation,
        validation_labels,
        validation_amounts,
        validation_rule_scores,
        validation_ml_scores,
        validation_hybrid_scores,
    )
    gc.collect()

    # The held-out test artifact is not opened until model selection, production
    # thresholds, and validation-only sensitivity analysis have been frozen.
    test_path = artifacts / "test_features.parquet"
    test_columns = list(
        dict.fromkeys(
            ["row_id", "split", "step", "label", "rule_score", "amount", *features]
        )
    )
    test = pd.read_parquet(test_path, columns=test_columns)
    if set(test["split"].unique()) != {"test"}:
        raise ValueError("test_features.parquet contains rows outside split='test'")
    if test["row_id"].duplicated().any():
        raise ValueError("test_features.parquet contains duplicate row_id values")
    missing_test_features = [name for name in features if name not in test.columns]
    if missing_test_features:
        raise ValueError(
            f"Selected features missing from test_features.parquet: {missing_test_features}"
        )
    test_sanity = sanity_check_feature_artifacts({"test": test}, features)
    sanity["splits"].update(test_sanity["splits"])
    labels_test = test["label"].to_numpy(dtype=np.int8)
    amounts_test = test["amount"].to_numpy(dtype=np.float64)
    rule_scores_test = test["rule_score"].to_numpy(dtype=np.float64)
    ml_scores_test = predict_frame(selected_bundle, test)
    hybrid_scores_test = combine_risk_score_arrays(
        rule_scores_test,
        ml_scores_test,
        DEFAULT_RULE_WEIGHT,
        DEFAULT_ML_WEIGHT,
    )
    evaluation = {
        "metric_note": "Average Precision is used as the summary AUC-PR metric.",
        "selection_source": "validation only",
        "test_usage": "single final evaluation after model and thresholds were frozen",
        "rule_only": evaluate_scores(
            labels_test,
            rule_scores_test,
            amounts_test,
            float(selected_bundle["rule_threshold"]),
            false_alarm_unit_cost=false_alarm_unit_cost,
        ),
        "ml_only": evaluate_scores(
            labels_test,
            ml_scores_test,
            amounts_test,
            float(selected_bundle["ml_threshold"]),
            false_alarm_unit_cost=false_alarm_unit_cost,
            training_time=float(selected_bundle["training_time"]),
        ),
        "hybrid": evaluate_scores(
            labels_test,
            hybrid_scores_test,
            amounts_test,
            float(selected_bundle["hybrid_threshold"]),
            false_alarm_unit_cost=false_alarm_unit_cost,
            training_time=float(selected_bundle["training_time"]),
        ),
    }

    model_version = (
        f"hybrid-{selected_tag}-"
        f"{dataset_metadata['dataset_version'][:12]}-"
        f"{dataset_metadata['split_manifest_version'][:12]}"
    )
    production_artifact = {
        "artifact_format_version": 2,
        "feature_configuration": "deployment_safe",
        "preprocessor": selected_bundle["preprocessor"],
        "model": selected_bundle["model"],
        "model_tag": selected_tag,
        "feature_columns": features,
        "ml_threshold": float(selected_bundle["ml_threshold"]),
        "hybrid_threshold": float(selected_bundle["hybrid_threshold"]),
        "rule_threshold": float(selected_bundle["rule_threshold"]),
        "rule_weight": DEFAULT_RULE_WEIGHT,
        "ml_weight": DEFAULT_ML_WEIGHT,
        "model_version": model_version,
        "dataset_version": dataset_metadata["dataset_version"],
        "split_manifest_version": dataset_metadata["split_manifest_version"],
        "cost_config": {"false_alarm_unit_cost": float(false_alarm_unit_cost)},
        "rule_config": dataset_metadata["rule_config"],
        "imbalance_strategy": selected_bundle["imbalance_strategy"],
        "dependency_versions": _dependency_versions(),
        "feature_schema_version": feature_schema["schema_version"],
        "random_state": random_state,
    }
    artifact_path = artifacts / "fraud_pipeline_selected.joblib"
    temporary_path = artifact_path.with_suffix(".joblib.tmp")
    joblib.dump(production_artifact, temporary_path)

    reloaded = load_model_artifact(temporary_path, strict=True, force_reload=True)
    assert reloaded is not None
    reloaded_scores = predict_frame(reloaded, test)
    if not np.allclose(ml_scores_test, reloaded_scores, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Predictions changed after exporting/reloading selected artifact")

    predictions = pd.DataFrame(
        {
            "row_id": test["row_id"].astype(np.int64),
            "step": test["step"].astype(np.int64),
            "amount": amounts_test,
            "label": labels_test,
            "feature_artifact": "test_features.parquet",
            "rule_score": rule_scores_test,
            "ml_score": ml_scores_test,
            "hybrid_score": hybrid_scores_test,
            "threshold": float(selected_bundle["hybrid_threshold"]),
            "prediction": (
                hybrid_scores_test >= float(selected_bundle["hybrid_threshold"])
            ).astype(np.int8),
        }
    )
    validate_test_prediction_consistency(
        predictions, evaluation["hybrid"], false_alarm_unit_cost
    )
    predictions_path = artifacts / "test_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    persisted_predictions = pd.read_parquet(predictions_path)
    validate_test_prediction_consistency(
        persisted_predictions, evaluation["hybrid"], false_alarm_unit_cost
    )

    comparison_rows = []
    for comparison in comparisons:
        hybrid = comparison["validation_hybrid"]
        ml = comparison["validation_ml_only"]
        comparison_rows.append(
            {
                "model_tag": comparison["model_tag"],
                "selected": comparison["model_tag"] == selected_tag,
                "validation_hybrid_business_cost": hybrid["business_cost"],
                "validation_hybrid_average_precision": hybrid["average_precision"],
                "validation_hybrid_f1": hybrid["f1"],
                "validation_ml_average_precision": ml["average_precision"],
                "ml_threshold": comparison["ml_threshold"],
                "hybrid_threshold": comparison["hybrid_threshold"],
                "training_time": comparison["training_time"],
                "imbalance_strategy": json.dumps(
                    comparison["imbalance_strategy"], sort_keys=True
                ),
            }
        )
    comparison_frame = pd.DataFrame(comparison_rows).sort_values(
        ["validation_hybrid_business_cost", "model_tag"], kind="stable"
    )
    comparison_frame.to_csv(artifacts / "model_comparison.csv", index=False)
    comparison_payload = {
        "selection_metric": "minimum_validation_hybrid_business_cost",
        "tie_breaker": "higher_validation_hybrid_average_precision",
        "selected_model_tag": selected_tag,
        "selected_model_version": model_version,
        "models": comparisons,
    }
    _write_json(artifacts / "model_comparison.json", comparison_payload)

    evaluation_payload = {
        "model_tag": selected_tag,
        "model_version": model_version,
        "feature_configuration": "deployment_safe",
        "feature_columns": features,
        "ml_threshold": float(selected_bundle["ml_threshold"]),
        "hybrid_threshold": float(selected_bundle["hybrid_threshold"]),
        "rule_weight": DEFAULT_RULE_WEIGHT,
        "ml_weight": DEFAULT_ML_WEIGHT,
        "false_alarm_unit_cost": float(false_alarm_unit_cost),
        "sanity_checks": sanity,
        "evaluation": evaluation,
        "artifact_reload_prediction_match": True,
        "test_prediction_confusion_match": True,
    }
    _write_json(artifacts / "evaluation_results.json", evaluation_payload)

    if run_feature_ablation:
        del (
            test,
            predictions,
            persisted_predictions,
            labels_test,
            amounts_test,
            rule_scores_test,
            ml_scores_test,
            hybrid_scores_test,
        )
        gc.collect()
        ablation = run_ablation_from_artifacts(
            artifacts,
            false_alarm_unit_cost=false_alarm_unit_cost,
            random_state=random_state,
            max_train_rows=int(
                ablation_max_rows
                if ablation_max_rows is not None
                else dataset_metadata["split_counts"]["train"]
            ),
        )
        ablation.to_csv(artifacts / "ablation_results.csv", index=False)

    # Publish only after the temporary bundle and every derived audit artifact have
    # passed consistency checks.
    temporary_path.replace(artifact_path)
    final_artifact = load_model_artifact(
        artifact_path, strict=True, force_reload=True
    )
    assert final_artifact is not None

    pointer_path = Path(__file__).resolve().parent / "selected_model.json"
    if update_runtime_pointer:
        try:
            relative_artifact = artifact_path.resolve().relative_to(pointer_path.parent)
            pointer_artifact = relative_artifact.as_posix()
        except ValueError:
            pointer_artifact = str(artifact_path.resolve())
        _write_json_atomic(
            pointer_path,
            {
                "artifact_path": pointer_artifact,
                "artifact_format_version": 2,
                "model_tag": selected_tag,
                "model_version": model_version,
                "feature_configuration": "deployment_safe",
            },
        )

    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    reset_artifact_cache()
    return {
        "selected_model_tag": selected_tag,
        "model_version": model_version,
        "artifact_path": str(artifact_path),
        "test_predictions_path": str(predictions_path),
        "evaluation": evaluation_payload,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate fraud models from prepared feature artifacts"
    )
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument(
        "--model-types",
        default="all",
        help=f"Comma-separated model types or all: {', '.join(MODEL_ORDER)}",
    )
    parser.add_argument("--model-type", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--false-alarm-cost", type=float, default=5.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--ablation-max-rows", type=int, default=500_000)
    parser.add_argument("--no-update-runtime-pointer", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    model_types = parse_model_types(args.model_type or args.model_types)
    result = train_and_export_models(
        args.artifacts_dir,
        model_types=model_types,
        false_alarm_unit_cost=args.false_alarm_cost,
        random_state=args.random_state,
        quick=args.quick,
        run_sensitivity_analysis=not args.skip_sensitivity,
        run_feature_ablation=not args.skip_ablation,
        ablation_max_rows=args.ablation_max_rows,
        update_runtime_pointer=not args.no_update_runtime_pointer,
    )
    hybrid = result["evaluation"]["evaluation"]["hybrid"]
    print(
        f"Selected {result['selected_model_tag']} with test hybrid "
        f"business_cost={hybrid['business_cost']:.2f}, "
        f"average_precision={hybrid['average_precision']:.6f}"
    )
    print(f"Artifact: {result['artifact_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
