from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("FRAUD_MODEL_TYPE", "v1")

from fraud_pipeline import PipelineConfig, RuleEngine, parse_csv_row

MODEL_DIR = ROOT / "model"
REFERENCE_DIR = ROOT / "monitoring" / "reference"
DEFAULT_INPUT_CSV = MODEL_DIR / "test_set.csv"
DEFAULT_OUTPUT_CSV = REFERENCE_DIR / "reference_dataset.csv"
DEFAULT_OUTPUT_JSON = REFERENCE_DIR / "reference_summary.json"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_reference_dataset(input_csv: Path, max_rows: int) -> pd.DataFrame:
    sample = pd.read_csv(input_csv, nrows=max_rows)
    engine = RuleEngine(PipelineConfig())
    records: list[dict[str, Any]] = []

    for row in sample.to_dict(orient="records"):
        csv_row = {
            "step": str(row["step"]),
            "type": str(row["type"]),
            "amount": str(row["amount"]),
            "nameOrig": str(row["nameOrig"]),
            "oldbalanceOrg": str(row["oldbalanceOrg"]),
            "newbalanceOrig": str(row["newbalanceOrig"]),
            "nameDest": str(row["nameDest"]),
            "oldbalanceDest": str(row["oldbalanceDest"]),
            "newbalanceDest": str(row["newbalanceDest"]),
            "isFraud": str(row["isFraud"]),
        }
        event = parse_csv_row(csv_row)
        decision = engine.evaluate(event)
        records.append(
            {
                "event_id": event.event_id,
                "step": event.step,
                "txn_type": event.txn_type,
                "amount": event.amount,
                "risk_score": decision.risk_score,
                "severity": decision.severity,
                "ml_score": decision.ml_score,
                "is_alert": decision.is_alert,
                "label_is_fraud": event.is_fraud,
            }
        )

    return pd.DataFrame(records)


def summarize_reference(df: pd.DataFrame, input_csv: Path) -> dict[str, Any]:
    txn_type_mix = (
        df["txn_type"].value_counts(normalize=True).sort_index().round(6).to_dict()
        if not df.empty
        else {}
    )
    severity_mix = (
        df["severity"].value_counts(normalize=True).sort_index().round(6).to_dict()
        if not df.empty
        else {}
    )
    return {
        "source_csv": str(input_csv),
        "sample_rows": int(len(df)),
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "columns": list(df.columns),
        "numeric_summary": {
            "amount": {
                "mean": round(float(df["amount"].mean()), 6),
                "std": round(float(df["amount"].std(ddof=0)), 6),
                "min": round(float(df["amount"].min()), 6),
                "max": round(float(df["amount"].max()), 6),
            },
            "risk_score": {
                "mean": round(float(df["risk_score"].mean()), 6),
                "std": round(float(df["risk_score"].std(ddof=0)), 6),
            },
            "ml_score": {
                "mean": round(float(df["ml_score"].mean()), 6),
                "std": round(float(df["ml_score"].std(ddof=0)), 6),
            },
        },
        "categorical_summary": {
            "txn_type": txn_type_mix,
            "severity": severity_mix,
        },
        "alert_rate": round(float(df["is_alert"].mean()), 6),
        "fraud_label_rate": round(float(df["label_is_fraud"].mean()), 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reference baseline dataset for model monitoring.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Path to the PaySim CSV used as reference.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Path to write the sampled reference dataset.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Path to write the reference summary JSON.")
    parser.add_argument("--max-rows", type=int, default=5000, help="Maximum number of rows to sample from the reference CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)

    reference_df = build_reference_dataset(input_csv=input_csv, max_rows=args.max_rows)
    summary = summarize_reference(reference_df, input_csv=input_csv)

    ensure_parent(output_csv)
    ensure_parent(output_json)
    reference_df.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote reference dataset to {output_csv}")
    print(f"Wrote reference summary to {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
