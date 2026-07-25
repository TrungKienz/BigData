#!/usr/bin/env python3
"""Simulate analyst review-queue activity by writing real rows into Cassandra `alert_reviews`.

This is a demo-data helper, same spirit as scripts/demo_api_drift_traffic.py and
scripts/push_paysim_to_api.py: it does not touch any report JSON directly. It
walks pending alerts in `alerts_by_account` that have no matching row in
`alert_reviews` yet, and inserts a synthetic review for each one -- exactly
what monitoring/model/performance_report.py already expects a human analyst to
produce via the Streamlit Review Queue (dashboard/streamlit/app.py, save_alert_review()).

Running this raises `label_coverage` and the 7d/30d labeled sample size for
real, because performance_report.py recomputes precision/recall/f1 from
whatever is actually in `alert_reviews` -- nothing is hardcoded.

Every inserted row is clearly marked as simulated in its `notes` column and
uses fictitious reviewer handles, so it can never be mistaken for a real
analyst decision if this Cassandra data is ever inspected directly.

Usage:
    docker-compose up -d        # cassandra must be running
    python scripts/simulate_analyst_reviews.py --max-reviews 200
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone

from cassandra.cluster import Cluster

REVIEWERS = ["demo.analyst.1", "demo.analyst.2", "demo.analyst.3"]
SIMULATED_NOTE = "Simulated review inserted by scripts/simulate_analyst_reviews.py for demo purposes -- not a real analyst decision."


def fetch_alerts(session) -> list:
    rows = session.execute("SELECT alert_id, event_id, risk_score, severity FROM alerts_by_account")
    return list(rows)


def fetch_reviewed_alert_ids(session) -> set[str]:
    rows = session.execute("SELECT alert_id FROM alert_reviews")
    return {row.alert_id for row in rows}


def decide_review(risk_score: float, severity: str, false_positive_rate: float, rng: random.Random) -> tuple[str, str]:
    # Alerts are, by construction, model-predicted positives. A realistic
    # analyst queue confirms most of them as fraud but overturns a minority
    # as false positives -- otherwise precision would sit at a suspicious 100%.
    # Lower-severity alerts are somewhat more likely to be overturned.
    severity_bias = {"high": 0.6, "medium": 1.0, "low": 1.6}.get(severity, 1.0)
    if rng.random() < min(false_positive_rate * severity_bias, 0.9):
        return "false_positive", "legit"
    return "confirmed_fraud", "fraud"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert simulated analyst reviews into Cassandra alert_reviews for pending alerts."
    )
    parser.add_argument("--cassandra-host", default="localhost")
    parser.add_argument("--cassandra-port", type=int, default=9042)
    parser.add_argument("--cassandra-keyspace", default="fraud_detection")
    parser.add_argument("--max-reviews", type=int, default=200, help="Maximum number of pending alerts to review.")
    parser.add_argument(
        "--false-positive-rate",
        type=float,
        default=0.12,
        help="Base fraction of reviewed alerts marked as false_positive/legit (adjusted by severity).",
    )
    parser.add_argument("--max-review-age-minutes", type=int, default=180, help="Reviews are backdated by a random amount up to this many minutes, to look like a queue processed over time.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    cluster = Cluster([args.cassandra_host], port=args.cassandra_port)
    session = cluster.connect(args.cassandra_keyspace)
    try:
        alerts = fetch_alerts(session)
        reviewed_ids = fetch_reviewed_alert_ids(session)
        pending = [alert for alert in alerts if alert.alert_id not in reviewed_ids]
        rng.shuffle(pending)
        pending = pending[: args.max_reviews]

        print(f"Alerts total: {len(alerts)} | already reviewed: {len(reviewed_ids)} | reviewing now: {len(pending)}")
        if not pending:
            print("Nothing to review -- either no alerts exist yet or all are already reviewed.")
            return 0

        now = datetime.now(timezone.utc)
        counts = {"confirmed_fraud": 0, "false_positive": 0}
        for alert in pending:
            review_status, review_label = decide_review(
                float(alert.risk_score or 0.0), alert.severity or "low", args.false_positive_rate, rng
            )
            counts[review_status] += 1
            reviewer = rng.choice(REVIEWERS)
            reviewed_at = now - timedelta(minutes=rng.randint(1, args.max_review_age_minutes))
            session.execute(
                """
                INSERT INTO alert_reviews (alert_id, event_id, review_status, review_label, reviewer, notes, reviewed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (alert.alert_id, alert.event_id, review_status, review_label, reviewer, SIMULATED_NOTE, reviewed_at),
            )

        print(f"Inserted {len(pending)} reviews: {counts['confirmed_fraud']} confirmed_fraud, {counts['false_positive']} false_positive.")
        print("\n== Next step ==")
        print(
            "  python monitoring/model/performance_report.py --cassandra-host localhost "
            "--cassandra-port 9042 --cassandra-keyspace fraud_detection"
        )
        print("  python monitoring/model/check_retraining_trigger.py")
        return 0
    finally:
        session.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
