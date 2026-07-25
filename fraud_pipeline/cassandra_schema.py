from __future__ import annotations

from cassandra.cluster import Cluster


HYBRID_METADATA_COLUMNS = (
    "rule_score double",
    "hybrid_score double",
    "threshold double",
    "decision text",
    "model_tag text",
    "feature_configuration text",
    "rule_weight double",
    "ml_weight double",
)


def _ensure_columns(session, table: str) -> None:
    for column in ("ml_score double", "ml_model_version text", *HYBRID_METADATA_COLUMNS):
        try:
            session.execute(f"ALTER TABLE {table} ADD {column}")
        except Exception:
            pass


def ensure_schema(host: str = "localhost", port: int = 9042, keyspace: str = "fraud_detection") -> None:
    cluster = Cluster([host], port=port)
    session = cluster.connect()
    try:
        session.execute(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
            """
        )
        session.set_keyspace(keyspace)
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts_by_account (
              account_id text,
              alert_date date,
              alert_ts timestamp,
              alert_id text,
              event_id text,
              name_dest text,
              txn_type text,
              amount double,
              risk_score double,
              rule_score double,
              ml_score double,
              hybrid_score double,
              threshold double,
              decision text,
              severity text,
              ml_model_version text,
              model_tag text,
              feature_configuration text,
              rule_weight double,
              ml_weight double,
              triggered_rules list<text>,
              PRIMARY KEY ((account_id, alert_date), alert_ts, alert_id)
            ) WITH CLUSTERING ORDER BY (alert_ts DESC)
            """
        )
        _ensure_columns(session, "alerts_by_account")
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions_by_day (
              day_bucket date,
              event_ts timestamp,
              event_id text,
              account_id text,
              name_dest text,
              txn_type text,
              amount double,
              sender_balance_before double,
              sender_balance_after double,
              receiver_balance_before double,
              receiver_balance_after double,
              is_fraud int,
              risk_score double,
              PRIMARY KEY ((day_bucket), event_ts, event_id)
            ) WITH CLUSTERING ORDER BY (event_ts DESC)
            """
        )
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS model_predictions_by_day (
              day_bucket date,
              event_ts timestamp,
              event_id text,
              account_id text,
              name_dest text,
              txn_type text,
              amount double,
              risk_score double,
              rule_score double,
              ml_score double,
              hybrid_score double,
              threshold double,
              decision text,
              severity text,
              ml_model_version text,
              model_tag text,
              feature_configuration text,
              rule_weight double,
              ml_weight double,
              triggered_rules list<text>,
              is_alert boolean,
              alert_id text,
              actual_label text,
              PRIMARY KEY ((day_bucket), event_ts, event_id)
            ) WITH CLUSTERING ORDER BY (event_ts DESC)
            """
        )
        _ensure_columns(session, "model_predictions_by_day")
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_reviews (
              alert_id text PRIMARY KEY,
              event_id text,
              review_status text,
              review_label text,
              reviewer text,
              notes text,
              reviewed_at timestamp
            )
            """
        )
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics_by_window (
              window_type text,
              window_bucket date,
              window_start timestamp,
              window_end timestamp,
              event_count bigint,
              fraud_count bigint,
              total_amount double,
              fraud_rate double,
              PRIMARY KEY ((window_type, window_bucket), window_start)
            ) WITH CLUSTERING ORDER BY (window_start DESC)
            """
        )
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS account_state_by_account (
              account_id text,
              event_ts timestamp,
              event_id text,
              source_event_id text,
              role text,
              step int,
              balance_before double,
              balance_after double,
              PRIMARY KEY ((account_id), event_ts, event_id)
            ) WITH CLUSTERING ORDER BY (event_ts DESC)
            """
        )
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS rules_by_id (
              rule_id text PRIMARY KEY,
              rule_type text,
              txn_type text,
              threshold double,
              count_threshold int,
              severity text,
              account_id text,
              operation text,
              updated_at timestamp
            )
            """
        )
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_stream_batches (
              query_name text,
              batch_id bigint,
              processed_at timestamp,
              PRIMARY KEY ((query_name), batch_id)
            )
            """
        )
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_stage_timings_by_run (
              benchmark_run_id text,
              event_id text,
              spark_seen_at timestamp,
              cassandra_persisted_at timestamp,
              PRIMARY KEY ((benchmark_run_id), event_id)
            )
            """
        )
    finally:
        session.shutdown()
        cluster.shutdown()
