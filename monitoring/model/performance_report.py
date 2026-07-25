from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.model.metrics_store import (
    load_predictions_from_cassandra,
    load_reviews_from_cassandra,
    parse_day_bucket,
    prepare_labeled_predictions,
)


REPORT_DIR = ROOT / "monitoring" / "reports"
DEFAULT_OUTPUT_JSON = REPORT_DIR / "performance_report.json"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_binary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "label_available" not in frame.columns:
        return {
            "labeled_rows": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "false_positive_rate": None,
        }

    labeled = frame[frame["label_available"]].copy()
    tp = int((labeled["predicted_positive"] & labeled["actual_positive"]).sum())
    fp = int((labeled["predicted_positive"] & labeled["actual_negative"]).sum())
    tn = int((~labeled["predicted_positive"] & labeled["actual_negative"]).sum())
    fn = int((~labeled["predicted_positive"] & labeled["actual_positive"]).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision is not None and recall is not None and (precision + recall) > 0) else None
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else None

    return {
        "labeled_rows": int(len(labeled)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(float(precision), 6) if precision is not None else None,
        "recall": round(float(recall), 6) if recall is not None else None,
        "f1": round(float(f1), 6) if f1 is not None else None,
        "false_positive_rate": round(float(false_positive_rate), 6) if false_positive_rate is not None else None,
    }


def build_window_reports(frame: pd.DataFrame, windows: tuple[int, ...] = (1, 7, 30)) -> dict[str, dict[str, Any]]:
    if frame.empty:
        return {f"{days}d": compute_binary_metrics(frame) for days in windows}

    anchor = pd.to_datetime(frame["evaluation_ts"]).max()
    reports: dict[str, dict[str, Any]] = {}
    for days in windows:
        window_start = anchor - timedelta(days=days)
        window_frame = frame[pd.to_datetime(frame["evaluation_ts"]) >= window_start]
        window_metrics = compute_binary_metrics(window_frame)
        window_metrics["window_start"] = window_start.isoformat()
        window_metrics["window_end"] = anchor.isoformat()
        reports[f"{days}d"] = window_metrics
    return reports


def build_performance_summary(predictions_df: pd.DataFrame, reviews_df: pd.DataFrame) -> dict[str, Any]:
    labeled_predictions = prepare_labeled_predictions(predictions_df, reviews_df)
    overall = compute_binary_metrics(labeled_predictions)
    label_source_breakdown = (
        labeled_predictions["label_source"].value_counts(dropna=False).to_dict() if not labeled_predictions.empty else {}
    )

    labeled_rows = int(labeled_predictions["label_available"].sum()) if not labeled_predictions.empty else 0
    prediction_rows = int(len(labeled_predictions))
    unlabeled_rows = prediction_rows - labeled_rows
    coverage = (labeled_rows / prediction_rows) if prediction_rows > 0 else 0.0

    warnings: list[str] = []
    if labeled_rows == 0:
        warnings.append("No labeled predictions available yet. Precision/recall cannot represent production quality.")
    if labeled_rows > 0 and int((~labeled_predictions["predicted_positive"] & labeled_predictions["label_available"]).sum()) == 0:
        warnings.append("No labeled negative predictions found. Recall may be optimistic because false negatives are not yet observable from reviewed alerts alone.")

    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "prediction_rows": prediction_rows,
        "review_rows": int(len(reviews_df)),
        "labeled_rows": labeled_rows,
        "unlabeled_rows": unlabeled_rows,
        "label_coverage": round(float(coverage), 6),
        "label_source_breakdown": label_source_breakdown,
        "overall": overall,
        "rolling_windows": build_window_reports(labeled_predictions),
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rolling performance metrics from predictions and analyst reviews.")
    parser.add_argument("--predictions-csv", help="Optional predictions CSV path.")
    parser.add_argument("--reviews-csv", help="Optional reviews CSV path.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Performance JSON output path.")
    parser.add_argument("--cassandra-host", default="localhost", help="Cassandra host.")
    parser.add_argument("--cassandra-port", type=int, default=9042, help="Cassandra port.")
    parser.add_argument("--cassandra-keyspace", default="fraud_detection", help="Cassandra keyspace.")
    parser.add_argument("--day-bucket", action="append", default=[], help="Optional prediction day bucket YYYY-MM-DD. Can be repeated.")
    parser.add_argument("--per-day-limit", type=int, default=5000, help="Maximum prediction rows to fetch per day bucket.")
    return parser.parse_args()


def load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    if args.predictions_csv:
        df = pd.read_csv(args.predictions_csv)
        if not df.empty and "event_ts" in df.columns:
            df["event_ts"] = pd.to_datetime(df["event_ts"])
        return df

    day_buckets = [parse_day_bucket(value) for value in args.day_bucket]
    return load_predictions_from_cassandra(
        host=args.cassandra_host,
        port=args.cassandra_port,
        keyspace=args.cassandra_keyspace,
        day_buckets=day_buckets,
        per_day_limit=args.per_day_limit,
    )


def load_reviews(args: argparse.Namespace) -> pd.DataFrame:
    if args.reviews_csv:
        df = pd.read_csv(args.reviews_csv)
        if not df.empty and "reviewed_at" in df.columns:
            df["reviewed_at"] = pd.to_datetime(df["reviewed_at"])
        return df

    return load_reviews_from_cassandra(
        host=args.cassandra_host,
        port=args.cassandra_port,
        keyspace=args.cassandra_keyspace,
    )


def main() -> int:
    args = parse_args()
    predictions_df = load_predictions(args)
    reviews_df = load_reviews(args)
    summary = build_performance_summary(predictions_df, reviews_df)

    output_json = Path(args.output_json)
    ensure_parent(output_json)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote performance summary to {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
