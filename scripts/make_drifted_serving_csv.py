#!/usr/bin/env python3
"""Build a synthetic 'serving_drifted.csv' from the monitoring reference dataset.

Demo helper for monitoring/model/drift_report.py: simulates a serving window
where transaction amounts have shifted (numeric drift, KS-test) and TRANSFER
transactions have become more common (categorical drift, total variation
distance), so the drift report shows real, non-trivial findings without
needing Kafka/Spark/Cassandra running.

Usage:
    python monitoring/model/reference_builder.py --max-rows 5000
    python scripts/make_drifted_serving_csv.py
    python monitoring/model/drift_report.py \
        --reference-csv monitoring/reference/reference_dataset.csv \
        --serving-csv monitoring/reference/serving_drifted.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_CSV = ROOT / "monitoring" / "reference" / "reference_dataset.csv"
DEFAULT_OUTPUT_CSV = ROOT / "monitoring" / "reference" / "serving_drifted.csv"


def inject_drift(df: pd.DataFrame, amount_multiplier: float, transfer_boost: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drifted = df.copy()

    # Numeric drift: scale amount with noise so its distribution shifts (KS-test on "amount").
    noise = rng.uniform(0.7, 1.3, size=len(drifted))
    drifted["amount"] = (drifted["amount"] * amount_multiplier * noise).round(2)

    # Categorical drift: push part of the mix to TRANSFER (total variation distance on "txn_type").
    flip_mask = rng.random(len(drifted)) < transfer_boost
    drifted.loc[flip_mask, "txn_type"] = "TRANSFER"

    # Knock-on effect so the shift also shows up in risk/severity/alert-rate, not just amount.
    bump = flip_mask.astype(float) * rng.uniform(0.2, 0.5, size=len(drifted))
    drifted["risk_score"] = np.minimum(drifted["risk_score"] + bump, 1.0)
    drifted["is_alert"] = drifted["is_alert"] | flip_mask
    drifted.loc[flip_mask, "severity"] = "high"

    return drifted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic drifted serving CSV for demoing drift_report.py.")
    parser.add_argument("--reference-csv", default=str(DEFAULT_REFERENCE_CSV), help="Path to reference_dataset.csv.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Path to write the drifted serving CSV.")
    parser.add_argument("--sample-rows", type=int, default=1500, help="Rows sampled from the reference set as the serving window.")
    parser.add_argument("--amount-multiplier", type=float, default=4.0, help="Multiplier applied to amount to force numeric drift.")
    parser.add_argument("--transfer-boost", type=float, default=0.45, help="Fraction of rows forced to txn_type=TRANSFER to force categorical drift.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible demo output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_path = Path(args.reference_csv)
    if not reference_path.exists():
        raise SystemExit(
            f"Reference CSV not found: {reference_path}. Run monitoring/model/reference_builder.py first."
        )

    reference_df = pd.read_csv(reference_path)
    sample_df = reference_df.sample(
        n=min(args.sample_rows, len(reference_df)), random_state=args.seed
    ).reset_index(drop=True)

    drifted_df = inject_drift(
        sample_df,
        amount_multiplier=args.amount_multiplier,
        transfer_boost=args.transfer_boost,
        seed=args.seed,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drifted_df.to_csv(output_path, index=False)

    transfer_before = (sample_df["txn_type"] == "TRANSFER").mean()
    transfer_after = (drifted_df["txn_type"] == "TRANSFER").mean()
    print(f"Wrote {len(drifted_df)} drifted rows to {output_path}")
    print(f"amount mean: reference={sample_df['amount'].mean():,.2f} -> drifted={drifted_df['amount'].mean():,.2f}")
    print(f"TRANSFER share: reference={transfer_before:.2%} -> drifted={transfer_after:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
