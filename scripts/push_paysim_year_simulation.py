#!/usr/bin/env python3
"""Replay PaySim transactions through the API spread across a simulated 1-year window.

data/archive/paysim_dataset.csv only covers ~31 days of simulated time (PaySim's
"step" column is 1 hour, max step ~743 = 31 days x 24h). To give model monitoring
(monitoring/model/*.py) a realistic year of history to compute drift/rolling
performance over, this script stretches that 31-day pattern across N days
(default 365): each sampled row's "step" is mapped to a fraction of the target
date range, with random jitter so same-step rows don't all land on one instant.

The API scores each request the moment it is sent, but the "day_bucket" it lands
in comes from the request's own event_time field (see api/service.py), so all
requests can be sent immediately in tight batches -- no need to actually wait
a year. isFraud/isFlaggedFraud are dropped from the payload, same as
push_paysim_to_api.py: a real caller would not know these labels in advance.

Usage:
    docker-compose up -d        # api + cassandra must be running
    python scripts/push_paysim_year_simulation.py --count 7300 --days 365
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "archive" / "paysim_dataset.csv"
DEFAULT_API_URL = "http://localhost:8000"


def reservoir_sample_with_max_step(csv_path: Path, count: int, seed: int) -> tuple[list[dict[str, str]], int]:
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    max_step = 0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            step = int(row["step"])
            if step > max_step:
                max_step = step
            if i < count:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < count:
                    reservoir[j] = row
    return reservoir, max_step


def assign_event_times(
    rows: list[dict[str, str]], max_step: int, start_date: datetime, end_date: datetime, rng: random.Random
) -> list[tuple[datetime, dict[str, str]]]:
    total_span = (end_date - start_date).total_seconds()
    slice_width = total_span / max_step if max_step > 0 else 0.0
    timed: list[tuple[datetime, dict[str, str]]] = []
    for row in rows:
        step = int(row["step"])
        fraction = step / max_step if max_step > 0 else 0.0
        jitter = rng.uniform(0.0, slice_width)
        event_time = start_date + timedelta(seconds=fraction * total_span + jitter)
        if event_time > end_date:
            event_time = end_date
        timed.append((event_time, row))
    timed.sort(key=lambda item: item[0])
    return timed


def row_to_payload(row: dict[str, str], event_time: datetime) -> dict[str, Any]:
    return {
        "event_time": event_time.isoformat(),
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
        if sent % (batch_size * 10) == 0 or sent == len(transactions):
            print(f"  sent {sent}/{len(transactions)} (alerts so far: {alerts})")
        if pause:
            time.sleep(pause)
    return sent, alerts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample real PaySim transactions and push them through the API with "
        "synthetic event_time spread across a simulated 1-year window."
    )
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV), help="Path to paysim_dataset.csv.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the running fraud-api service.")
    parser.add_argument("--count", type=int, default=7300, help="Total number of transactions to sample and send.")
    parser.add_argument("--days", type=int, default=365, help="Width of the simulated date range, in days.")
    parser.add_argument(
        "--end-date",
        default=None,
        help="ISO date (YYYY-MM-DD) the simulated range ends at, UTC. Defaults to now (UTC).",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Transactions per /score/batch call.")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to sleep between batches.")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout per request, in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling/jitter.")
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

    end_date = (
        datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
        if args.end_date
        else datetime.now(timezone.utc)
    )
    start_date = end_date - timedelta(days=args.days)
    print(f"\nKhoang thoi gian mo phong: {start_date.date()} -> {end_date.date()} ({args.days} ngay)")

    print(f"Dang lay mau {args.count} giao dich tu {csv_path} (reservoir sampling, 1 lan doc qua file)...")
    rows, max_step = reservoir_sample_with_max_step(csv_path, args.count, args.seed)
    print(f"Da lay mau {len(rows)} giao dich (step goc 0..{max_step} trong du lieu PaySim, ~31 ngay).")

    rng = random.Random(args.seed)
    timed_rows = assign_event_times(rows, max_step, start_date, end_date, rng)
    transactions = [row_to_payload(row, event_time) for event_time, row in timed_rows]
    print(
        f"Da rai deu {len(transactions)} giao dich tren {args.days} ngay "
        f"(trung binh ~{len(transactions) / args.days:.1f} giao dich/ngay). "
        "Bo cot isFraud/isFlaggedFraud truoc khi gui."
    )

    print(f"\n== Day {len(transactions)} giao dich qua {args.api_url}/score/batch (event_time gia lap) ==")
    sent, alerts = send_transactions(args.api_url, transactions, args.batch_size, args.pause, args.timeout)
    print(f"\nHoan tat: {sent} giao dich da gui, {alerts} alert ({alerts / sent:.1%})")

    print("\n== Buoc tiep theo ==")
    print(
        "  # kiem tra cac day_bucket da tao (trai dai ca nam):\n"
        "  docker exec cassandra cqlsh -e \"SELECT day_bucket, COUNT(*) FROM fraud_detection.model_predictions_by_day "
        "GROUP BY day_bucket;\""
    )
    print(
        "  python monitoring/model/performance_report.py --cassandra-host localhost "
        "--cassandra-port 9042 --cassandra-keyspace fraud_detection   "
        "# bo --day-bucket: tu dong dot nguoc toi 400 ngay, lay ngay GAN NHAT co du lieu"
    )
    print("  python monitoring/model/check_retraining_trigger.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
