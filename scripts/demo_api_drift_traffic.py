#!/usr/bin/env python3
"""Send real traffic through the live Fraud Scoring API (/score/batch).

Unlike scripts/make_drifted_serving_csv.py (which fakes a serving CSV offline),
this script goes through the real serving path: it POSTs synthetic PaySim-style
transactions to the running API, which scores them and persists each prediction
to Cassandra's model_predictions_by_day table (requires prediction_logging_enabled,
checked via /health). After this runs, monitoring/model/drift_report.py and
performance_report.py can read genuine live data with --cassandra-host.

Usage:
    docker-compose up -d        # api + cassandra must be running
    python scripts/demo_api_drift_traffic.py --drift-count 200

    # optional "before" traffic to also seed some normal-looking rows first
    python scripts/demo_api_drift_traffic.py --baseline-count 150 --drift-count 200
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

DEFAULT_API_URL = "http://localhost:8000"

TXN_TYPES = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
# Matches the real mix in monitoring/reference/reference_dataset.csv.
BASELINE_TXN_WEIGHTS = [0.357, 0.346, 0.211, 0.079, 0.007]
# Skewed hard toward TRANSFER with larger amounts to force drift.
DRIFTED_TXN_WEIGHTS = [0.15, 0.10, 0.10, 0.60, 0.05]


def make_transaction(rng: np.random.Generator, step: int, txn_type: str, amount: float) -> dict[str, Any]:
    old_balance_org = amount + rng.uniform(0, amount * 2)
    if txn_type == "CASH_IN":
        new_balance_orig = old_balance_org + amount
    else:
        new_balance_orig = max(old_balance_org - amount, 0.0)
    old_balance_dest = rng.uniform(0, amount)
    new_balance_dest = old_balance_dest + amount

    return {
        "step": step,
        "type": txn_type,
        "amount": round(float(amount), 2),
        "nameOrig": f"C{int(rng.integers(10 ** 8, 10 ** 9))}",
        "oldbalanceOrg": round(float(old_balance_org), 2),
        "newbalanceOrig": round(float(new_balance_orig), 2),
        "nameDest": f"C{int(rng.integers(10 ** 8, 10 ** 9))}",
        "oldbalanceDest": round(float(old_balance_dest), 2),
        "newbalanceDest": round(float(new_balance_dest), 2),
        "isFraud": 0,
    }


def make_phase_transactions(
    rng: np.random.Generator, count: int, txn_weights: list[float], amount_multiplier: float, step_start: int
) -> list[dict[str, Any]]:
    transactions = []
    for i in range(count):
        txn_type = rng.choice(TXN_TYPES, p=txn_weights)
        base_amount = rng.lognormal(mean=11.0, sigma=1.1)
        amount = base_amount * amount_multiplier
        transactions.append(make_transaction(rng, step_start + i, txn_type, amount))
    return transactions


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


def send_phase(
    api_url: str, label: str, transactions: list[dict[str, Any]], batch_size: int, pause: float, timeout: float
) -> tuple[int, int]:
    sent = 0
    alerts = 0
    for start in range(0, len(transactions), batch_size):
        chunk = transactions[start : start + batch_size]
        result = call_json(f"{api_url}/score/batch", {"transactions": chunk}, timeout)
        predictions = result["predictions"]
        alerts += sum(1 for prediction in predictions if prediction["is_alert"])
        sent += len(chunk)
        print(f"  [{label}] sent {sent}/{len(transactions)} (alerts so far: {alerts})")
        if pause:
            time.sleep(pause)
    return sent, alerts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push baseline + drifted synthetic traffic through the live Fraud Scoring API "
        "so monitoring/model/*.py can read real drift from Cassandra."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the running fraud-api service.")
    parser.add_argument("--baseline-count", type=int, default=0, help="Normal-mix transactions to send first (0 = skip).")
    parser.add_argument("--drift-count", type=int, default=200, help="Drifted transactions to send.")
    parser.add_argument("--amount-multiplier", type=float, default=5.0, help="Amount multiplier for the drifted phase.")
    parser.add_argument("--batch-size", type=int, default=25, help="Transactions per /score/batch call.")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to sleep between batches.")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout per request, in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    health = call_json(f"{args.api_url}/health", None, args.timeout)
    print(f"API OK: model_version={health.get('model_version')} model_type={health.get('model_type')}")
    if not health.get("prediction_logging_enabled"):
        print(
            "CANH BAO: prediction_logging_enabled=false tren API nay - Cassandra se KHONG duoc ghi, "
            "drift/performance report se rong. Kiem tra bien moi truong API_PREDICTION_LOGGING_ENABLED."
        )

    if args.baseline_count > 0:
        print(f"\n== Phase 1: baseline traffic ({args.baseline_count} giao dich, phan phoi giong reference) ==")
        baseline_txns = make_phase_transactions(
            rng, args.baseline_count, BASELINE_TXN_WEIGHTS, amount_multiplier=1.0, step_start=1
        )
        b_sent, b_alerts = send_phase(args.api_url, "baseline", baseline_txns, args.batch_size, args.pause, args.timeout)
        print(f"Baseline: {b_sent} giao dich, {b_alerts} alert ({b_alerts / b_sent:.1%})")

    print(f"\n== Phase 2: drifted traffic ({args.drift_count} giao dich, amount x{args.amount_multiplier:.1f}, TRANSFER chiem uu the) ==")
    drifted_txns = make_phase_transactions(
        rng, args.drift_count, DRIFTED_TXN_WEIGHTS, amount_multiplier=args.amount_multiplier,
        step_start=args.baseline_count + 1,
    )
    d_sent, d_alerts = send_phase(args.api_url, "drifted", drifted_txns, args.batch_size, args.pause, args.timeout)
    print(f"Drifted: {d_sent} giao dich, {d_alerts} alert ({d_alerts / d_sent:.1%})")

    print("\n== Buoc tiep theo (doc du lieu that tu Cassandra) ==")
    print("  python monitoring/model/reference_builder.py --max-rows 5000   # chi can chay 1 lan neu chua co")
    print(
        "  python monitoring/model/drift_report.py --cassandra-host localhost "
        "--cassandra-port 9042 --cassandra-keyspace fraud_detection"
    )
    print(
        "  python monitoring/model/performance_report.py --cassandra-host localhost "
        "--cassandra-port 9042 --cassandra-keyspace fraud_detection"
    )
    print("  python monitoring/model/check_retraining_trigger.py")
    print("Sau do reload http://localhost:8501 (tab Monitoring).")
    print(
        "Luu y: performance_report.json se co label_coverage thap vi predictions moi chua duoc "
        "analyst gan nhan trong Review Queue - drift_report la phan the hien ro nhat trong demo nay."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
