from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from cassandra.cluster import Cluster
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REFERENCE_DIR = ROOT / "monitoring" / "reference"
REPORT_DIR = ROOT / "monitoring" / "reports"
DEFAULT_REFERENCE_CSV = REFERENCE_DIR / "reference_dataset.csv"
DEFAULT_REPORT_JSON = REPORT_DIR / "drift_report.json"
DEFAULT_REPORT_HTML = REPORT_DIR / "drift_report.html"

NUMERIC_FEATURES = ("amount", "risk_score", "ml_score")
CATEGORICAL_FEATURES = ("txn_type", "severity", "is_alert")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_day_bucket(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_reference_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def discover_prediction_days(session, lookback_days: int, limit: int) -> list[date]:
    discovered: list[date] = []
    today = datetime.utcnow().date()
    probe_dates = [today - timedelta(days=offset) for offset in range(lookback_days)]
    if date(2026, 1, 1) not in probe_dates:
        probe_dates.append(date(2026, 1, 1))

    for day_bucket in probe_dates:
        rows = session.execute(
            "SELECT event_id FROM model_predictions_by_day WHERE day_bucket = %s LIMIT %s",
            (day_bucket, 1),
        )
        if list(rows):
            discovered.append(day_bucket)
            if len(discovered) >= limit:
                break
    return discovered


def load_serving_dataset_from_cassandra(host: str, port: int, keyspace: str, day_buckets: list[date], per_day_limit: int) -> pd.DataFrame:
    cluster = Cluster([host], port=port)
    session = cluster.connect(keyspace)
    try:
        buckets = day_buckets or discover_prediction_days(session, lookback_days=400, limit=7)
        records: list[dict[str, Any]] = []
        for day_bucket in buckets:
            rows = session.execute(
                """
                SELECT event_id, txn_type, amount, risk_score, severity, ml_score, is_alert
                FROM model_predictions_by_day WHERE day_bucket = %s LIMIT %s
                """,
                (day_bucket, per_day_limit),
            )
            for row in rows:
                records.append({
                    "event_id": row.event_id,
                    "txn_type": row.txn_type,
                    "amount": row.amount,
                    "risk_score": row.risk_score,
                    "severity": row.severity,
                    "ml_score": row.ml_score,
                    "is_alert": row.is_alert,
                })
        return pd.DataFrame(records)
    finally:
        session.shutdown()
        cluster.shutdown()


def load_serving_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if args.serving_csv:
        return pd.read_csv(args.serving_csv)

    day_buckets = [parse_day_bucket(item) for item in args.day_bucket]
    return load_serving_dataset_from_cassandra(
        host=args.cassandra_host,
        port=args.cassandra_port,
        keyspace=args.cassandra_keyspace,
        day_buckets=day_buckets,
        per_day_limit=args.per_day_limit,
    )


def compute_numeric_drift(reference: pd.Series, serving: pd.Series) -> dict[str, Any]:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    serving = pd.to_numeric(serving, errors="coerce").dropna()
    if reference.empty or serving.empty:
        return {"drift_detected": False, "reason": "insufficient_data"}

    ks_stat, ks_pvalue = ks_2samp(reference, serving)
    mean_delta = float(serving.mean() - reference.mean())
    drift_detected = bool(ks_stat >= 0.2)
    return {
        "drift_detected": drift_detected,
        "metric": "ks_statistic",
        "ks_statistic": round(float(ks_stat), 6),
        "ks_pvalue": round(float(ks_pvalue), 6),
        "reference_mean": round(float(reference.mean()), 6),
        "serving_mean": round(float(serving.mean()), 6),
        "mean_delta": round(mean_delta, 6),
    }


def compute_categorical_drift(reference: pd.Series, serving: pd.Series) -> dict[str, Any]:
    reference_dist = reference.astype(str).value_counts(normalize=True)
    serving_dist = serving.astype(str).value_counts(normalize=True)
    all_keys = sorted(set(reference_dist.index).union(set(serving_dist.index)))
    total_variation = 0.0
    per_category: dict[str, dict[str, float]] = {}

    for key in all_keys:
        reference_value = float(reference_dist.get(key, 0.0))
        serving_value = float(serving_dist.get(key, 0.0))
        total_variation += abs(reference_value - serving_value)
        per_category[key] = {
            "reference": round(reference_value, 6),
            "serving": round(serving_value, 6),
        }

    total_variation *= 0.5
    return {
        "drift_detected": bool(total_variation >= 0.1),
        "metric": "total_variation_distance",
        "total_variation_distance": round(total_variation, 6),
        "distribution": per_category,
    }


def build_drift_summary(reference_df: pd.DataFrame, serving_df: pd.DataFrame) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        if feature in reference_df.columns and feature in serving_df.columns:
            features[feature] = compute_numeric_drift(reference_df[feature], serving_df[feature])
    for feature in CATEGORICAL_FEATURES:
        if feature in reference_df.columns and feature in serving_df.columns:
            features[feature] = compute_categorical_drift(reference_df[feature], serving_df[feature])

    drifted_features = [name for name, details in features.items() if details.get("drift_detected")]
    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "reference_rows": int(len(reference_df)),
        "serving_rows": int(len(serving_df)),
        "monitored_features": list(features.keys()),
        "drifted_feature_count": len(drifted_features),
        "drifted_features": drifted_features,
        "features": features,
    }


def build_html_report(summary: dict[str, Any]) -> str:
    rows = []
    for feature, details in summary["features"].items():
        rows.append(
            f"<tr><td>{feature}</td><td>{details.get('metric', 'n/a')}</td><td>{details.get('drift_detected')}</td><td><pre>{json.dumps(details, indent=2)}</pre></td></tr>"
        )
    joined_rows = "\n".join(rows)
    return f"""
<html>
  <head>
    <title>Drift Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; padding: 24px; background: #0b1220; color: #e5e7eb; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 24px; }}
      th, td {{ border: 1px solid #334155; padding: 12px; vertical-align: top; }}
      th {{ background: #111827; }}
      pre {{ white-space: pre-wrap; margin: 0; }}
    </style>
  </head>
  <body>
    <h1>Model Drift Report</h1>
    <p>Generated at: {summary['generated_at']}</p>
    <p>Reference rows: {summary['reference_rows']} | Serving rows: {summary['serving_rows']}</p>
    <p>Drifted features: {', '.join(summary['drifted_features']) if summary['drifted_features'] else 'None'}</p>
    <table>
      <thead>
        <tr><th>Feature</th><th>Metric</th><th>Drift Detected</th><th>Details</th></tr>
      </thead>
      <tbody>
        {joined_rows}
      </tbody>
    </table>
  </body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a drift report comparing reference and serving data.")
    parser.add_argument("--reference-csv", default=str(DEFAULT_REFERENCE_CSV), help="Reference dataset CSV path.")
    parser.add_argument("--serving-csv", help="Optional serving dataset CSV path for offline verification.")
    parser.add_argument("--output-json", default=str(DEFAULT_REPORT_JSON), help="JSON output path.")
    parser.add_argument("--output-html", default=str(DEFAULT_REPORT_HTML), help="HTML output path.")
    parser.add_argument("--cassandra-host", default="localhost", help="Cassandra host for serving data.")
    parser.add_argument("--cassandra-port", type=int, default=9042, help="Cassandra port for serving data.")
    parser.add_argument("--cassandra-keyspace", default="fraud_detection", help="Cassandra keyspace.")
    parser.add_argument("--day-bucket", action="append", default=[], help="Optional day bucket YYYY-MM-DD. Can be repeated.")
    parser.add_argument("--per-day-limit", type=int, default=5000, help="Maximum serving rows to fetch per day bucket.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_df = load_reference_dataset(Path(args.reference_csv))
    serving_df = load_serving_dataset(args)
    if serving_df.empty:
        raise SystemExit("No serving rows available for drift comparison.")

    summary = build_drift_summary(reference_df, serving_df)
    html_report = build_html_report(summary)

    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    ensure_parent(output_json)
    ensure_parent(output_html)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_html.write_text(html_report, encoding="utf-8")

    print(f"Wrote drift summary to {output_json}")
    print(f"Wrote drift report to {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
