#!/usr/bin/env python3
"""Replay real PaySim transactions through the live Fraud Scoring API (/score/batch).

Reads data/archive/paysim_dataset.csv (step,type,amount,nameOrig,oldbalanceOrg,
newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud),
drops isFraud/isFlaggedFraud (a real caller would not know these in advance),
and POSTs a random sample of transactions to the running API. Each request is
scored through the real serving path and persisted to Cassandra's
model_predictions_by_day table (requires prediction_logging_enabled, see /health).

The source CSV has ~6.36M rows / ~490MB, so rows are chosen with reservoir
sampling while streaming the file once, instead of loading it fully into memory.

Usage:
    docker-compose up -d        # api + cassandra must be running
    python scripts/push_paysim_to_api.py --count 5000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "archive" / "paysim_dataset.csv"
DEFAULT_API_URL = "http://localhost:8000"

# Columns forwarded to the API. isFraud/isFlaggedFraud are intentionally excluded:
# a real caller scoring a transaction in flight would not have these labels yet.
PAYLOAD_COLUMNS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
)


def reservoir_sample(csv_path: Path, count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if i < count:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < count:
                    reservoir[j] = row
    return reservoir


def row_to_payload(row: dict[str, str]) -> dict[str, Any]:
    return {
        "step": int(row["step"]),
        "type": row["type"],
        "amount": float(row["amount"]),
        "nameOrig": row["nameOrig"],
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "nameDest": row["nameDest"],
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
    }


def call_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API tra loi loi {exc.code} tai {url}: {detail}") from exc


def send_transactions(
    api_url: str, transactions: list[dict[str, Any]], batch_size: int, pause: float, timeout: float
) -> tuple[int, int]:
    sent = 0
    alerts = 0
    for start in range(0, len(transactions), batch_size):
        chunk = transactions[start : start + batch_size]
        result = call_json(f"{api_url}/score/batch", {"transactions": chunk}, timeout)
        predictions = result["predictions"]
        alerts += sum(1 for prediction in predictions if prediction["is_alert"])
        sent += len(chunk)
        print(f"  sent {sent}/{len(transactions)} (alerts so far: {alerts})")
        if pause:
            time.sleep(pause)
    return sent, alerts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample real PaySim transactions and push them through the live Fraud Scoring API."
    )
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV), help="Path to paysim_dataset.csv.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the running fraud-api service.")
    parser.add_argument("--count", type=int, default=5000, help="Number of transactions to sample and send.")
    parser.add_argument("--batch-size", type=int, default=50, help="Transactions per /score/batch call.")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to sleep between batches.")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout per request, in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="Send the sample in original CSV row order (chronological by step) instead of sampled order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"Khong tim thay CSV: {csv_path}")

    health = call_json(f"{args.api_url}/health", None, args.timeout)
    print(f"API OK: model_version={health.get('model_version')} model_type={health.get('model_type')}")
    if not health.get("prediction_logging_enabled"):
        print(
            "CANH BAO: prediction_logging_enabled=false tren API nay - Cassandra se KHONG duoc ghi, "
            "drift/performance report se rong. Kiem tra bien moi truong API_PREDICTION_LOGGING_ENABLED."
        )

    print(f"\nDang lay mau {args.count} giao dich tu {csv_path} (reservoir sampling, 1 lan doc qua file)...")
    rows = reservoir_sample(csv_path, args.count, args.seed)
    if args.keep_order:
        rows.sort(key=lambda row: int(row["step"]))
    print(f"Da lay mau {len(rows)} giao dich. Bo cot isFraud/isFlaggedFraud truoc khi gui.")

    transactions = [row_to_payload(row) for row in rows]

    print(f"\n== Day {len(transactions)} giao dich qua {args.api_url}/score/batch ==")
    sent, alerts = send_transactions(args.api_url, transactions, args.batch_size, args.pause, args.timeout)
    print(f"\nHoan tat: {sent} giao dich da gui, {alerts} alert ({alerts / sent:.1%})")

    print("\n== Buoc tiep theo (doc du lieu that tu Cassandra) ==")
    print(
        "  python monitoring/model/drift_report.py --cassandra-host localhost "
        "--cassandra-port 9042 --cassandra-keyspace fraud_detection --day-bucket <YYYY-MM-DD-UTC>"
    )
    print(
        "  python monitoring/model/performance_report.py --cassandra-host localhost "
        "--cassandra-port 9042 --cassandra-keyspace fraud_detection --day-bucket <YYYY-MM-DD-UTC>"
    )
    print("  python monitoring/model/check_retraining_trigger.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
