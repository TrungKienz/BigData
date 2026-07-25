from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fraud_pipeline.config import PipelineConfig, rule_scoring_config
from fraud_pipeline.features import (
    CATEGORICAL_FEATURES,
    DYNAMIC_FEATURES,
    FEATURE_CONFIGURATIONS,
    FEATURE_GROUPS,
    FORBIDDEN_FEATURE_COLUMNS,
    FULL_PAYSIM_FEATURES,
    INFERENCE_AVAILABLE_FEATURES,
    POST_TRANSACTION_FEATURES,
    REQUIRED_CLEANED_COLUMNS,
    DynamicFeatureState,
    assert_no_forbidden_features,
    build_static_features_frame,
    validate_cleaned_schema,
)
from fraud_pipeline.models import TransactionEvent
from fraud_pipeline.rules import RuleEngine


DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "transactions_cleaned.parquet"
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = (0.60, 0.20, 0.20)


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


def read_cleaned_dataset(
    data_path: str | Path,
    *,
    limit: int | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned transaction dataset not found: {path}. Expected "
            "transactions_cleaned.parquet from the upstream cleaning pipeline."
        )
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer")
    if limit is not None and sample_size is not None:
        raise ValueError("Use either --limit or --sample-size, not both")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            if limit is None:
                frame = pd.read_parquet(path)
            else:
                import pyarrow.parquet as pq

                batches = []
                remaining = limit
                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(batch_size=min(65_536, remaining)):
                    current = batch.to_pandas()
                    batches.append(current.iloc[:remaining])
                    remaining -= len(current)
                    if remaining <= 0:
                        break
                frame = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
        except ImportError as exc:
            raise RuntimeError(
                "Parquet support requires pyarrow. Install model/requirements-ml.txt."
            ) from exc
    elif suffix == ".csv":
        frame = pd.read_csv(path, nrows=limit)
    else:
        raise ValueError(
            f"Unsupported cleaned dataset format {suffix!r}; use Parquet (default) or CSV"
        )

    validate_cleaned_schema(frame)
    frame = frame.copy()
    if "row_id" not in frame.columns:
        frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    else:
        if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
            raise ValueError("Existing row_id column must be non-null and unique")
        frame["row_id"] = pd.to_numeric(frame["row_id"], errors="raise").astype(
            np.int64
        )

    if sample_size is not None and sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=random_state, replace=False)
    frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype(np.int64)
    frame["isFraud"] = pd.to_numeric(
        frame["isFraud"], errors="raise"
    ).astype(np.int8)
    return frame.sort_values(["step", "row_id"], kind="stable").reset_index(drop=True)


def create_split_manifest(
    frame: pd.DataFrame,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> pd.DataFrame:
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must contain train/validation/test values summing to 1")
    if any(value <= 0 for value in ratios):
        raise ValueError("Every split ratio must be positive")
    if frame["row_id"].duplicated().any():
        raise ValueError("row_id must be unique before creating a split manifest")

    counts = frame.groupby("step", sort=True).size()
    if len(counts) < 3:
        raise ValueError(
            "Chronological 60/20/20 splitting requires at least three distinct step values"
        )
    cumulative = counts.cumsum().to_numpy(dtype=np.int64)
    total = int(cumulative[-1])
    first_target = ratios[0] * total
    second_target = (ratios[0] + ratios[1]) * total

    first_candidates = np.arange(0, len(counts) - 2)
    first_index = int(
        first_candidates[
            np.argmin(np.abs(cumulative[first_candidates] - first_target))
        ]
    )
    second_candidates = np.arange(first_index + 1, len(counts) - 1)
    second_index = int(
        second_candidates[
            np.argmin(np.abs(cumulative[second_candidates] - second_target))
        ]
    )

    steps = counts.index.to_numpy()
    split_by_step = {
        step: (
            "train"
            if index <= first_index
            else "validation"
            if index <= second_index
            else "test"
        )
        for index, step in enumerate(steps)
    }
    manifest = frame.loc[:, ["row_id", "step"]].copy()
    manifest["split"] = manifest["step"].map(split_by_step)
    validate_split_manifest(manifest, frame)
    return manifest


def validate_split_manifest(manifest: pd.DataFrame, frame: pd.DataFrame) -> None:
    required = {"row_id", "step", "split"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Split manifest is missing columns: {missing}")
    if manifest["row_id"].isna().any() or manifest["row_id"].duplicated().any():
        raise ValueError("Split manifest row_id values must be non-null and unique")
    if set(manifest["split"].unique()) != set(SPLIT_NAMES):
        raise ValueError(
            f"Split manifest must contain exactly {SPLIT_NAMES}; "
            f"found {sorted(manifest['split'].unique())}"
        )

    expected = frame.loc[:, ["row_id", "step"]].sort_values("row_id").reset_index(drop=True)
    actual = manifest.loc[:, ["row_id", "step"]].sort_values("row_id").reset_index(drop=True)
    if len(expected) != len(actual) or not expected.equals(actual):
        raise ValueError(
            "Split manifest does not exactly cover the current dataset row_id/step pairs"
        )
    step_split_counts = manifest.groupby("step")["split"].nunique()
    if int(step_split_counts.max()) != 1:
        bad_steps = step_split_counts[step_split_counts > 1].index.tolist()
        raise ValueError(f"Transactions from the same step cross splits: {bad_steps[:10]}")

    boundaries = manifest.groupby("split")["step"].agg(["min", "max"])
    if not (
        boundaries.loc["train", "max"] < boundaries.loc["validation", "min"]
        and boundaries.loc["validation", "max"] < boundaries.loc["test", "min"]
    ):
        raise ValueError("Split manifest is not strictly chronological")


def load_or_create_split_manifest(
    frame: pd.DataFrame,
    manifest_path: str | Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    path = Path(manifest_path)
    if path.exists() and not force:
        manifest = pd.read_parquet(path)
        manifest["row_id"] = manifest["row_id"].astype(np.int64)
        manifest["step"] = manifest["step"].astype(np.int64)
        validate_split_manifest(manifest, frame)
        return manifest
    manifest = create_split_manifest(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)
    return manifest


def _event_from_row(row: Any, config: PipelineConfig) -> TransactionEvent:
    event_time = config.base_event_time + timedelta(
        seconds=int(row.step) * config.step_seconds
    )
    return TransactionEvent(
        event_id=f"row:{int(row.row_id)}",
        event_time=event_time,
        producer_ts=event_time,
        step=int(row.step),
        txn_type=str(row.type),
        amount=float(row.amount),
        name_orig=str(row.nameOrig),
        oldbalance_org=float(row.oldbalanceOrg),
        newbalance_orig=float(row.newbalanceOrig),
        name_dest=str(row.nameDest),
        oldbalance_dest=float(row.oldbalanceDest),
        newbalance_dest=float(row.newbalanceDest),
        is_fraud=int(row.isFraud),
        schema_version=config.schema_version,
        hour_of_day=int(row.hour_of_day),
        is_night_transaction=int(row.is_night_transaction),
        customer_account_age_days=float(row.customer_account_age_days),
        browser=str(row.browser),
        device_type=str(row.device_type),
        new_device_flag=int(row.new_device_flag),
        billing_country=str(row.billing_country),
        ip_country=str(row.ip_country),
        ip_billing_distance_km=float(row.ip_billing_distance_km),
        ip_billing_country_mismatch=int(row.ip_billing_country_mismatch),
        shipping_billing_mismatch=int(row.shipping_billing_mismatch),
        failed_payment_attempts_24h=float(row.failed_payment_attempts_24h),
    )


def build_engineered_features(
    frame: pd.DataFrame,
    config: PipelineConfig | None = None,
    *,
    state: DynamicFeatureState | None = None,
    engine: RuleEngine | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    config = config or PipelineConfig()
    ordered = frame.sort_values(["step", "row_id"], kind="stable").reset_index(drop=True)
    static = build_static_features_frame(ordered, config=config).reset_index(drop=True)
    dynamic_values = np.zeros((len(ordered), len(DYNAMIC_FEATURES)), dtype=np.float32)
    rule_scores = np.zeros(len(ordered), dtype=np.float64)
    state = state or DynamicFeatureState(config)
    engine = engine or RuleEngine(config)

    steps = ordered["step"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(steps[1:] != steps[:-1]) + 1, len(ordered)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rows = ordered.iloc[int(start) : int(end)]
        events = [_event_from_row(row, config) for row in rows.itertuples(index=False)]
        contexts, step_features = state.calculate_step(events)
        for offset, (event, context, values) in enumerate(
            zip(events, contexts, step_features)
        ):
            dynamic_values[int(start) + offset] = [
                values[name] for name in DYNAMIC_FEATURES
            ]
            rule_scores[int(start) + offset] = engine.evaluate_rules(
                event, context
            ).score
        state.update_after_step(events)

    dynamic = pd.DataFrame(dynamic_values, columns=DYNAMIC_FEATURES)
    engineered = pd.concat([static, dynamic], axis=1)
    return engineered, rule_scores


def _series_signature(series: pd.Series) -> bytes:
    hashed = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).digest()


def select_features(
    engineered: pd.DataFrame,
    manifest: pd.DataFrame,
    feature_configuration: str,
    *,
    near_constant_threshold: float = 0.9999,
) -> tuple[list[str], list[dict[str, Any]]]:
    if feature_configuration not in FEATURE_CONFIGURATIONS:
        raise ValueError(
            f"Unknown feature configuration {feature_configuration!r}; "
            f"choose from {sorted(FEATURE_CONFIGURATIONS)}"
        )
    if not 0.5 < near_constant_threshold <= 1.0:
        raise ValueError("near_constant_threshold must be in (0.5, 1.0]")

    candidates = list(FEATURE_CONFIGURATIONS[feature_configuration])
    missing = [name for name in candidates if name not in engineered.columns]
    if missing:
        raise ValueError(f"Engineered feature frame is missing configured columns: {missing}")
    assert_no_forbidden_features(candidates)

    train_positions = manifest.index[manifest["split"] == "train"]
    train = engineered.loc[train_positions, candidates]
    reasons: dict[str, str] = {}

    if feature_configuration == "deployment_safe":
        for name in POST_TRANSACTION_FEATURES:
            reasons[name] = (
                "post_transaction_value_not_guaranteed_before_approve_review_block"
            )

    for name in candidates:
        if name not in INFERENCE_AVAILABLE_FEATURES and feature_configuration == "deployment_safe":
            reasons[name] = "unavailable_at_runtime_inference"
            continue
        counts = train[name].value_counts(dropna=False, normalize=True)
        unique_count = int(train[name].nunique(dropna=False))
        if unique_count <= 1:
            reasons[name] = "constant_on_training_split"
        elif not counts.empty and float(counts.iloc[0]) >= near_constant_threshold:
            reasons[name] = (
                f"near_constant_on_training_split>={near_constant_threshold}"
            )

    signatures: dict[bytes, str] = {}
    for name in candidates:
        if name in reasons:
            continue
        signature = _series_signature(train[name])
        previous = signatures.get(signature)
        if previous is not None and train[name].equals(train[previous]):
            reasons[name] = f"duplicate_of:{previous}"
        else:
            signatures[signature] = name

    remaining_numeric = [
        name
        for name in candidates
        if name not in reasons
        and name not in CATEGORICAL_FEATURES
        and pd.api.types.is_numeric_dtype(train[name])
    ]
    if len(remaining_numeric) > 1:
        correlation_sample = train[remaining_numeric].iloc[:200_000]
        correlations = correlation_sample.corr().abs()
        for right_index, right in enumerate(remaining_numeric):
            if right in reasons:
                continue
            for left in remaining_numeric[:right_index]:
                if left in reasons:
                    continue
                value = correlations.loc[left, right]
                if pd.notna(value) and float(value) >= 0.999999:
                    reasons[right] = f"redundant_derived_feature_of:{left}"
                    break

    selected = [name for name in candidates if name not in reasons]
    assert_no_forbidden_features(selected)
    if not selected:
        raise ValueError("Feature selection removed every candidate feature")

    explicit_forbidden_reasons = {
        "isFraud": "target_label",
        "label_is_fraud": "target_label",
        "label": "target_label",
        "isFlaggedFraud": "upstream_rule_label_leakage",
        "row_id": "artifact_row_identifier",
        "split": "dataset_partition_marker",
        "step": "raw_time_index_not_deployable",
        "nameOrig": "high_cardinality_sender_identifier",
        "nameDest": "high_cardinality_receiver_identifier",
        "device_id": "high_cardinality_device_identifier",
        "event_id": "event_identifier",
    }
    report: list[dict[str, Any]] = []
    all_report_features = list(dict.fromkeys(FULL_PAYSIM_FEATURES + tuple(explicit_forbidden_reasons)))
    for name in all_report_features:
        selected_flag = name in selected
        reason = (
            "selected"
            if selected_flag
            else reasons.get(name)
            or explicit_forbidden_reasons.get(name)
            or "not_in_requested_feature_configuration"
        )
        report.append(
            {
                "feature": name,
                "selected": selected_flag,
                "reason": reason,
                "feature_configuration": feature_configuration,
                "categorical": name in CATEGORICAL_FEATURES,
                "inference_available": name in INFERENCE_AVAILABLE_FEATURES,
            }
        )
    return selected, report


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["row_id", *REQUIRED_CLEANED_COLUMNS]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(
        dtype=np.uint64
    )
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _manifest_fingerprint(manifest: pd.DataFrame) -> str:
    ordered = manifest.sort_values("row_id")
    hashed = pd.util.hash_pandas_object(
        ordered[["row_id", "step", "split"]], index=False
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _write_feature_contract_files(
    artifacts: Path,
    selected: list[str],
    selection_report: list[dict[str, Any]],
    feature_configuration: str,
    metadata: dict[str, Any],
) -> dict[str, str]:
    categorical_selected = [name for name in selected if name in CATEGORICAL_FEATURES]
    numeric_selected = [name for name in selected if name not in CATEGORICAL_FEATURES]
    _write_json(
        artifacts / "selected_features.json",
        {
            "feature_configuration": feature_configuration,
            "selected_features": selected,
            "numeric_features": numeric_selected,
            "categorical_features": categorical_selected,
        },
    )
    excluded = [row for row in selection_report if not row["selected"]]
    _write_json(artifacts / "excluded_features.json", excluded)
    pd.DataFrame(selection_report).to_csv(
        artifacts / "feature_selection_report.csv", index=False
    )
    _write_json(
        artifacts / "feature_schema.json",
        {
            "schema_version": 2,
            "feature_configuration": feature_configuration,
            "selected_features": selected,
            "numeric_features": numeric_selected,
            "categorical_features": categorical_selected,
            "feature_configurations": {
                name: list(values)
                for name, values in FEATURE_CONFIGURATIONS.items()
            },
            "feature_groups": {
                name: list(values) for name, values in FEATURE_GROUPS.items()
            },
            "forbidden_features": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "dynamic_feature_semantics": {
                "ordering": "chronological_by_step",
                "same_step_visibility": "none; state updates after every row in the step is scored",
                "history": "strictly earlier step and timestamp values only",
                "time_since_last_transaction_default_seconds": 86_400.0,
                "validation_history": "train plus prior validation steps",
                "test_history": "train, validation, plus prior test steps",
            },
        },
    )
    _write_json(artifacts / "dataset_metadata.json", metadata)
    ablation_rows = [
        {
            "feature_set": name,
            "feature_count": len(values),
            "status": "prepared; metrics populated by model.train_models",
        }
        for name, values in FEATURE_GROUPS.items()
    ]
    pd.DataFrame(ablation_rows).to_csv(
        artifacts / "ablation_results.csv", index=False
    )
    return {
        "selected_features": str(artifacts / "selected_features.json"),
        "excluded_features": str(artifacts / "excluded_features.json"),
        "feature_selection_report": str(
            artifacts / "feature_selection_report.csv"
        ),
        "ablation_results": str(artifacts / "ablation_results.csv"),
        "feature_schema": str(artifacts / "feature_schema.json"),
        "dataset_metadata": str(artifacts / "dataset_metadata.json"),
    }


def _scan_parquet_manifest(
    data_path: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, bool]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support requires pyarrow. Install model/requirements-ml.txt."
        ) from exc

    parquet = pq.ParquetFile(data_path)
    schema_names = set(parquet.schema_arrow.names)
    missing = sorted(set(REQUIRED_CLEANED_COLUMNS).difference(schema_names))
    if missing:
        raise ValueError(
            "Cleaned transaction dataset is missing required columns: "
            + ", ".join(missing)
        )
    has_row_id = "row_id" in schema_names
    columns = ["step"] + (["row_id"] if has_row_id else [])
    chunks: list[pd.DataFrame] = []
    offset = 0
    monotonic = True
    previous_step: int | None = None
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        current = batch.to_pandas()
        current["step"] = pd.to_numeric(current["step"], errors="raise").astype(
            np.int64
        )
        if has_row_id:
            current["row_id"] = pd.to_numeric(
                current["row_id"], errors="raise"
            ).astype(np.int64)
        else:
            current["row_id"] = np.arange(
                offset, offset + len(current), dtype=np.int64
            )
        if len(current):
            values = current["step"].to_numpy(dtype=np.int64, copy=False)
            monotonic = monotonic and bool(np.all(values[1:] >= values[:-1]))
            if previous_step is not None and int(values[0]) < previous_step:
                monotonic = False
            previous_step = int(values[-1])
        chunks.append(current[["row_id", "step"]])
        offset += len(current)
    if not chunks:
        raise ValueError("Cleaned transaction dataset contains no rows")
    frame = pd.concat(chunks, ignore_index=True)
    if frame["row_id"].duplicated().any():
        raise ValueError("row_id must be unique in transactions_cleaned.parquet")
    return frame, monotonic


def _scan_csv_manifest(
    data_path: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, bool]:
    schema_names = set(pd.read_csv(data_path, nrows=0).columns)
    missing = sorted(set(REQUIRED_CLEANED_COLUMNS).difference(schema_names))
    if missing:
        raise ValueError(
            "Cleaned transaction dataset is missing required columns: "
            + ", ".join(missing)
        )
    has_row_id = "row_id" in schema_names
    columns = ["step"] + (["row_id"] if has_row_id else [])
    chunks: list[pd.DataFrame] = []
    offset = 0
    monotonic = True
    previous_step: int | None = None
    for current in pd.read_csv(
        data_path,
        usecols=columns,
        chunksize=batch_size,
        low_memory=False,
    ):
        current["step"] = pd.to_numeric(current["step"], errors="raise").astype(
            np.int64
        )
        if has_row_id:
            current["row_id"] = pd.to_numeric(
                current["row_id"], errors="raise"
            ).astype(np.int64)
        else:
            current["row_id"] = np.arange(
                offset, offset + len(current), dtype=np.int64
            )
        if len(current):
            values = current["step"].to_numpy(dtype=np.int64, copy=False)
            monotonic = monotonic and bool(np.all(values[1:] >= values[:-1]))
            if previous_step is not None and int(values[0]) < previous_step:
                monotonic = False
            previous_step = int(values[-1])
        chunks.append(current[["row_id", "step"]])
        offset += len(current)
    if not chunks:
        raise ValueError("Cleaned transaction dataset contains no rows")
    frame = pd.concat(chunks, ignore_index=True)
    if frame["row_id"].duplicated().any():
        raise ValueError("row_id must be unique in transactions_cleaned.csv")
    return frame, monotonic


def _deterministic_parquet_sample(
    parquet_path: Path,
    columns: list[str],
    max_rows: int,
    batch_size: int,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    total_rows = parquet.metadata.num_rows
    if total_rows <= max_rows:
        return pd.read_parquet(parquet_path, columns=columns)
    modulus = 1_000_003
    threshold = max(1, int(modulus * max_rows * 1.1 / total_rows))
    samples: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        current = batch.to_pandas()
        row_hash = pd.util.hash_pandas_object(
            current["row_id"], index=False
        ).to_numpy(dtype=np.uint64)
        selected = current.loc[(row_hash % modulus) < threshold]
        if not selected.empty:
            samples.append(selected)
    if not samples:
        raise ValueError("Could not create a deterministic training feature sample")
    sample = pd.concat(samples, ignore_index=True)
    if len(sample) > max_rows:
        sample = sample.sort_values("row_id", kind="stable").iloc[:max_rows]
    return sample.reset_index(drop=True)


def _verify_streaming_feature_exclusions(
    train_path: Path,
    selected: list[str],
    report: list[dict[str, Any]],
    feature_configuration: str,
    near_constant_threshold: float,
    batch_size: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Verify every sample-derived removal against the complete training split."""

    import pyarrow.parquet as pq

    rows_by_feature = {row["feature"]: row for row in report}
    sampled_reasons = {
        name: str(row["reason"])
        for name, row in rows_by_feature.items()
        if not row["selected"]
        and str(row["reason"]).startswith(
            (
                "constant_on_training_split",
                "near_constant_on_training_split",
                "duplicate_of:",
                "redundant_derived_feature_of:",
            )
        )
    }
    if not sampled_reasons:
        return selected, report

    columns = set(sampled_reasons)
    for reason in sampled_reasons.values():
        if ":" in reason:
            columns.add(reason.split(":", 1)[1])
    value_counts: dict[str, dict[object, int] | None] = {
        name: {} for name, reason in sampled_reasons.items()
        if reason.startswith(("constant_", "near_constant_"))
    }
    duplicate_matches = {
        name: True
        for name, reason in sampled_reasons.items()
        if reason.startswith("duplicate_of:")
    }
    correlation_stats = {
        name: np.zeros(6, dtype=np.float64)
        for name, reason in sampled_reasons.items()
        if reason.startswith("redundant_derived_feature_of:")
    }
    total_rows = 0
    parquet = pq.ParquetFile(train_path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=sorted(columns)):
        current = batch.to_pandas()
        total_rows += len(current)
        for name, counts in list(value_counts.items()):
            if counts is None:
                continue
            for value, count in current[name].value_counts(dropna=False).items():
                key: object = "__nan__" if pd.isna(value) else value
                counts[key] = counts.get(key, 0) + int(count)
            if len(counts) > 100_000:
                value_counts[name] = None
        for name in duplicate_matches:
            reference = sampled_reasons[name].split(":", 1)[1]
            if duplicate_matches[name] and not current[name].equals(current[reference]):
                duplicate_matches[name] = False
        for name, stats in correlation_stats.items():
            reference = sampled_reasons[name].split(":", 1)[1]
            x = pd.to_numeric(current[name], errors="coerce").to_numpy(
                dtype=np.float64
            )
            y = pd.to_numeric(current[reference], errors="coerce").to_numpy(
                dtype=np.float64
            )
            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            stats += np.array(
                [
                    len(x),
                    x.sum(),
                    y.sum(),
                    np.dot(x, x),
                    np.dot(y, y),
                    np.dot(x, y),
                ],
                dtype=np.float64,
            )

    verified_excluded: set[str] = set()
    for name, reason in sampled_reasons.items():
        if reason.startswith("constant_"):
            counts = value_counts[name]
            if counts is not None and len(counts) <= 1:
                verified_excluded.add(name)
        elif reason.startswith("near_constant_"):
            counts = value_counts[name]
            if counts and max(counts.values()) / total_rows >= near_constant_threshold:
                verified_excluded.add(name)
        elif reason.startswith("duplicate_of:"):
            if duplicate_matches[name]:
                verified_excluded.add(name)
        elif reason.startswith("redundant_derived_feature_of:"):
            n, sx, sy, sxx, syy, sxy = correlation_stats[name]
            covariance = sxy - sx * sy / n if n else 0.0
            variance_x = sxx - sx * sx / n if n else 0.0
            variance_y = syy - sy * sy / n if n else 0.0
            denominator = np.sqrt(max(variance_x * variance_y, 0.0))
            correlation = abs(covariance / denominator) if denominator > 0 else 0.0
            if correlation >= 0.999999:
                verified_excluded.add(name)

    for name in sampled_reasons:
        if name not in verified_excluded:
            rows_by_feature[name]["selected"] = True
            rows_by_feature[name]["reason"] = "selected_after_full_train_verification"
    verified_selected = [
        name
        for name in FEATURE_CONFIGURATIONS[feature_configuration]
        if rows_by_feature[name]["selected"]
    ]
    assert_no_forbidden_features(verified_selected)
    return verified_selected, report


def _prepare_streaming(
    data_path: Path,
    artifacts: Path,
    *,
    feature_configuration: str,
    force_split: bool,
    near_constant_threshold: float,
    chunk_size: int,
    feature_selection_sample_rows: int,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_format = data_path.suffix.lower().lstrip(".")
    if source_format in {"parquet", "pq"}:
        manifest_source, monotonic = _scan_parquet_manifest(data_path, chunk_size)
        parquet = pq.ParquetFile(data_path)
        has_row_id = "row_id" in parquet.schema_arrow.names
        source_batches = (
            batch.to_pandas()
            for batch in parquet.iter_batches(batch_size=chunk_size)
        )
    elif source_format == "csv":
        manifest_source, monotonic = _scan_csv_manifest(data_path, chunk_size)
        has_row_id = "row_id" in pd.read_csv(data_path, nrows=0).columns
        source_batches = pd.read_csv(
            data_path,
            chunksize=chunk_size,
            low_memory=False,
        )
    else:
        raise ValueError(
            f"Unsupported cleaned dataset format {data_path.suffix!r}; "
            "use Parquet or CSV"
        )
    if not monotonic:
        raise ValueError(
            "Streaming feature preparation requires the cleaned dataset "
            "to be sorted by step. Use --in-memory for an unsorted compatibility input."
        )
    manifest_path = artifacts / "split_manifest.parquet"
    manifest = load_or_create_split_manifest(
        manifest_source, manifest_path, force=force_split
    )
    manifest = manifest.set_index("row_id").loc[
        manifest_source["row_id"]
    ].reset_index()
    validate_split_manifest(manifest, manifest_source)
    split_by_step = (
        manifest.drop_duplicates("step").set_index("step")["split"].to_dict()
    )
    split_counts = manifest["split"].value_counts().to_dict()
    split_version = _manifest_fingerprint(manifest)

    state = DynamicFeatureState(PipelineConfig())
    engine = RuleEngine(state.config)
    writers: dict[str, pq.ParquetWriter] = {}
    output_paths = {
        split: artifacts / f"{split}_features.parquet" for split in SPLIT_NAMES
    }
    fingerprint = hashlib.sha256()
    row_count = 0
    fraud_count = 0
    step_min: int | None = None
    step_max: int | None = None
    carry = pd.DataFrame()

    def write_complete_steps(complete: pd.DataFrame) -> None:
        if complete.empty:
            return
        complete = complete.sort_values(["step", "row_id"], kind="stable").reset_index(
            drop=True
        )
        engineered, rule_scores = build_engineered_features(
            complete, state=state, engine=engine
        )
        split_values = complete["step"].map(split_by_step)
        artifact_frame = pd.DataFrame(
            {
                "row_id": complete["row_id"].astype(np.int64),
                "split": split_values.astype(str),
                "step": complete["step"].astype(np.int64),
                "label": complete["isFraud"].astype(np.int8),
                "rule_score": rule_scores,
            }
        )
        artifact_frame = pd.concat(
            [artifact_frame, engineered.loc[:, list(FULL_PAYSIM_FEATURES)]],
            axis=1,
        )
        for split_name in SPLIT_NAMES:
            split_frame = artifact_frame.loc[
                artifact_frame["split"] == split_name
            ]
            if split_frame.empty:
                continue
            table = pa.Table.from_pandas(split_frame, preserve_index=False)
            writer = writers.get(split_name)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_paths[split_name], table.schema, compression="snappy"
                )
                writers[split_name] = writer
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)

    offset = 0
    try:
        for current in source_batches:
            if has_row_id:
                current["row_id"] = pd.to_numeric(
                    current["row_id"], errors="raise"
                ).astype(np.int64)
            else:
                current.insert(
                    0,
                    "row_id",
                    np.arange(offset, offset + len(current), dtype=np.int64),
                )
            offset += len(current)
            validate_cleaned_schema(current)
            current["step"] = pd.to_numeric(
                current["step"], errors="raise"
            ).astype(np.int64)
            current["isFraud"] = pd.to_numeric(
                current["isFraud"], errors="raise"
            ).astype(np.int8)
            hashed = pd.util.hash_pandas_object(
                current[["row_id", *REQUIRED_CLEANED_COLUMNS]], index=False
            ).to_numpy(dtype=np.uint64)
            fingerprint.update(hashed.tobytes())
            row_count += len(current)
            fraud_count += int(current["isFraud"].sum())
            if len(current):
                batch_min = int(current["step"].min())
                batch_max = int(current["step"].max())
                step_min = batch_min if step_min is None else min(step_min, batch_min)
                step_max = batch_max if step_max is None else max(step_max, batch_max)

            pending = (
                pd.concat([carry, current], ignore_index=True)
                if not carry.empty
                else current
            )
            last_step = int(pending["step"].iloc[-1])
            complete = pending.loc[pending["step"] < last_step]
            carry = pending.loc[pending["step"] == last_step].copy()
            write_complete_steps(complete)
        write_complete_steps(carry)
    finally:
        for writer in writers.values():
            writer.close()
    missing_outputs = [
        split for split in SPLIT_NAMES if not output_paths[split].exists()
    ]
    if missing_outputs:
        raise RuntimeError(f"No feature artifact was written for splits: {missing_outputs}")

    selection_columns = ["row_id", *FULL_PAYSIM_FEATURES]
    selection_sample = _deterministic_parquet_sample(
        output_paths["train"],
        selection_columns,
        feature_selection_sample_rows,
        chunk_size,
    )
    selection_manifest = pd.DataFrame(
        {"split": np.repeat("train", len(selection_sample))}
    )
    selected, selection_report = select_features(
        selection_sample[list(FULL_PAYSIM_FEATURES)],
        selection_manifest,
        feature_configuration,
        near_constant_threshold=near_constant_threshold,
    )
    selected, selection_report = _verify_streaming_feature_exclusions(
        output_paths["train"],
        selected,
        selection_report,
        feature_configuration,
        near_constant_threshold,
        chunk_size,
    )
    metadata = {
        "source_path": str(data_path.resolve()),
        "source_format": source_format,
        "dataset_version": fingerprint.hexdigest(),
        "split_manifest_version": split_version,
        "row_count": row_count,
        "fraud_count": fraud_count,
        "fraud_rate": float(fraud_count / row_count),
        "split_counts": {
            name: int(split_counts.get(name, 0)) for name in SPLIT_NAMES
        },
        "step_min": int(step_min),
        "step_max": int(step_max),
        "feature_configuration": feature_configuration,
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "rule_config": rule_scoring_config(),
        "preparation_mode": "streaming_complete_step_batches",
        "chunk_size": chunk_size,
        "feature_selection_sample_rows": len(selection_sample),
        "limit": None,
        "sample_size": None,
    }
    contract_paths = _write_feature_contract_files(
        artifacts,
        selected,
        selection_report,
        feature_configuration,
        metadata,
    )
    paths = {
        "split_manifest": str(manifest_path),
        **{
            f"{split}_features": str(path)
            for split, path in output_paths.items()
        },
        **contract_paths,
    }
    return {"paths": paths, "metadata": metadata, "selected_features": selected}


def prepare_feature_artifacts(
    data_path: str | Path,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    *,
    feature_configuration: str = "deployment_safe",
    limit: int | None = None,
    sample_size: int | None = None,
    force_split: bool = False,
    near_constant_threshold: float = 0.9999,
    chunk_size: int = 250_000,
    feature_selection_sample_rows: int = 500_000,
    in_memory: bool = False,
) -> dict[str, Any]:
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    path = Path(data_path)
    if chunk_size <= 0 or feature_selection_sample_rows <= 0:
        raise ValueError("chunk_size and feature_selection_sample_rows must be positive")
    if (
        path.suffix.lower() in {".parquet", ".pq", ".csv"}
        and limit is None
        and sample_size is None
        and not in_memory
    ):
        return _prepare_streaming(
            path,
            artifacts,
            feature_configuration=feature_configuration,
            force_split=force_split,
            near_constant_threshold=near_constant_threshold,
            chunk_size=chunk_size,
            feature_selection_sample_rows=feature_selection_sample_rows,
        )
    frame = read_cleaned_dataset(
        data_path, limit=limit, sample_size=sample_size
    )
    manifest_path = artifacts / "split_manifest.parquet"
    manifest = load_or_create_split_manifest(
        frame, manifest_path, force=force_split
    )
    manifest = manifest.set_index("row_id").loc[frame["row_id"]].reset_index()
    validate_split_manifest(manifest, frame)

    engineered, rule_scores = build_engineered_features(frame)
    selected, selection_report = select_features(
        engineered,
        manifest,
        feature_configuration,
        near_constant_threshold=near_constant_threshold,
    )

    artifact_frame = pd.DataFrame(
        {
            "row_id": frame["row_id"].astype(np.int64),
            "split": manifest["split"].astype(str),
            "step": frame["step"].astype(np.int64),
            "label": frame["isFraud"].astype(np.int8),
            "rule_score": rule_scores,
        }
    )
    artifact_frame = pd.concat(
        [artifact_frame, engineered.loc[:, list(FULL_PAYSIM_FEATURES)]], axis=1
    )

    output_paths: dict[str, str] = {"split_manifest": str(manifest_path)}
    for split_name in SPLIT_NAMES:
        output_path = artifacts / f"{split_name}_features.parquet"
        artifact_frame.loc[artifact_frame["split"] == split_name].to_parquet(
            output_path, index=False
        )
        output_paths[f"{split_name}_features"] = str(output_path)

    categorical_selected = [name for name in selected if name in CATEGORICAL_FEATURES]
    numeric_selected = [name for name in selected if name not in CATEGORICAL_FEATURES]
    selected_payload = {
        "feature_configuration": feature_configuration,
        "selected_features": selected,
        "numeric_features": numeric_selected,
        "categorical_features": categorical_selected,
    }
    _write_json(artifacts / "selected_features.json", selected_payload)

    excluded = [row for row in selection_report if not row["selected"]]
    _write_json(artifacts / "excluded_features.json", excluded)
    pd.DataFrame(selection_report).to_csv(
        artifacts / "feature_selection_report.csv", index=False
    )

    schema_payload = {
        "schema_version": 2,
        "feature_configuration": feature_configuration,
        "selected_features": selected,
        "numeric_features": numeric_selected,
        "categorical_features": categorical_selected,
        "feature_configurations": {
            name: list(values) for name, values in FEATURE_CONFIGURATIONS.items()
        },
        "feature_groups": {name: list(values) for name, values in FEATURE_GROUPS.items()},
        "forbidden_features": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "dynamic_feature_semantics": {
            "ordering": "chronological_by_step",
            "same_step_visibility": "none; state updates after every row in the step is scored",
            "history": "strictly earlier timestamps only",
            "time_since_last_transaction_default_seconds": 86_400.0,
            "validation_history": "train plus prior validation steps",
            "test_history": "train, validation, plus prior test steps",
        },
    }
    _write_json(artifacts / "feature_schema.json", schema_payload)

    split_counts = manifest["split"].value_counts().to_dict()
    dataset_version = _frame_fingerprint(frame)
    split_version = _manifest_fingerprint(manifest)
    metadata = {
        "source_path": str(Path(data_path).resolve()),
        "source_format": Path(data_path).suffix.lower().lstrip("."),
        "dataset_version": dataset_version,
        "split_manifest_version": split_version,
        "row_count": len(frame),
        "fraud_count": int(frame["isFraud"].sum()),
        "fraud_rate": float(frame["isFraud"].mean()),
        "split_counts": {name: int(split_counts.get(name, 0)) for name in SPLIT_NAMES},
        "step_min": int(frame["step"].min()),
        "step_max": int(frame["step"].max()),
        "feature_configuration": feature_configuration,
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "rule_config": rule_scoring_config(),
        "limit": limit,
        "sample_size": sample_size,
    }
    _write_json(artifacts / "dataset_metadata.json", metadata)

    ablation_rows = []
    for name, values in FEATURE_GROUPS.items():
        available = [value for value in values if value in engineered.columns]
        ablation_rows.append(
            {
                "feature_set": name,
                "feature_count": len(available),
                "status": "prepared; metrics populated by model.train_models",
            }
        )
    pd.DataFrame(ablation_rows).to_csv(
        artifacts / "ablation_results.csv", index=False
    )

    output_paths.update(
        {
            "selected_features": str(artifacts / "selected_features.json"),
            "excluded_features": str(artifacts / "excluded_features.json"),
            "feature_selection_report": str(
                artifacts / "feature_selection_report.csv"
            ),
            "ablation_results": str(artifacts / "ablation_results.csv"),
            "feature_schema": str(artifacts / "feature_schema.json"),
            "dataset_metadata": str(artifacts / "dataset_metadata.json"),
        }
    )
    return {"paths": output_paths, "metadata": metadata, "selected_features": selected}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe fraud model feature artifacts"
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path to transactions_cleaned Parquet or CSV",
    )
    parser.add_argument(
        "--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR), help="Artifact output directory"
    )
    parser.add_argument(
        "--feature-config",
        default="deployment_safe",
        choices=sorted(FEATURE_CONFIGURATIONS),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--near-constant-threshold", type=float, default=0.9999)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--feature-selection-sample-rows", type=int, default=500_000)
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Compatibility mode for unsorted/small input; full inputs default to streaming",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    result = prepare_feature_artifacts(
        args.data_path,
        args.artifacts_dir,
        feature_configuration=args.feature_config,
        limit=args.limit,
        sample_size=args.sample_size,
        force_split=args.force_split,
        near_constant_threshold=args.near_constant_threshold,
        chunk_size=args.chunk_size,
        feature_selection_sample_rows=args.feature_selection_sample_rows,
        in_memory=args.in_memory,
    )
    metadata = result["metadata"]
    print(
        f"Prepared {metadata['row_count']:,} rows with "
        f"{metadata['selected_feature_count']} selected features in {args.artifacts_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
