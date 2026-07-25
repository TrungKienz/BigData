import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from cassandra.cluster import Cluster


st.set_page_config(
    page_title="Fraud Detection Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(10, 20, 30) 0%, rgb(0, 0, 0) 90%);
        color: #E0E0E0;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #00D2FF !important;
    }

    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    h1, h2, h3 {
        color: #FFFFFF !important;
        letter-spacing: -1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


REVIEW_STATUS_OPTIONS = ["new", "in_review", "confirmed_fraud", "false_positive", "escalated", "needs_more_info"]
REVIEW_LABEL_OPTIONS = ["unlabeled", "fraud", "legit", "needs_more_info"]


def resolve_root() -> Path:
    file_path = Path(__file__).resolve()
    for candidate in (file_path.parent, *file_path.parents):
        if (candidate / "monitoring" / "reports").exists():
            return candidate
    return file_path.parent


ROOT = resolve_root()
REPORT_DIR = ROOT / "monitoring" / "reports"


@st.cache_resource
def get_cassandra_session():
    try:
        cluster = Cluster(["localhost"], port=9042)
        return cluster.connect("fraud_detection")
    except Exception:
        try:
            cluster = Cluster(["cassandra"], port=9042)
            return cluster.connect("fraud_detection")
        except Exception as exc:
            st.error(f"Failed to connect to Cassandra: {exc}")
            return None


def load_alerts() -> pd.DataFrame:
    session = get_cassandra_session()
    if not session:
        return pd.DataFrame()

    try:
        rows = session.execute("SELECT * FROM alerts_by_account LIMIT 1000")
        df = pd.DataFrame(list(rows))
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df["alert_ts"] = pd.to_datetime(df["alert_ts"])
    if "triggered_rules" in df.columns:
        df["triggered_rules"] = df["triggered_rules"].apply(
            lambda value: list(value) if isinstance(value, (list, tuple)) else ([] if pd.isna(value) else [str(value)])
        )
        df["primary_rule"] = df["triggered_rules"].apply(lambda rules: rules[0] if rules else "ml_only")
    else:
        df["triggered_rules"] = [[] for _ in range(len(df))]
        df["primary_rule"] = "ml_only"

    df["triggered_rules_display"] = df["triggered_rules"].apply(lambda rules: ", ".join(rules) if rules else "ml_only")
    return df


def ensure_review_state() -> None:
    if "selected_alert_id" not in st.session_state:
        st.session_state.selected_alert_id = None


def load_alert_reviews() -> pd.DataFrame:
    session = get_cassandra_session()
    if not session:
        return pd.DataFrame(columns=["alert_id", "event_id", "review_status", "review_label", "reviewer", "notes", "reviewed_at"])

    try:
        rows = session.execute("SELECT alert_id, event_id, review_status, review_label, reviewer, notes, reviewed_at FROM alert_reviews")
        df = pd.DataFrame(list(rows))
    except Exception:
        return pd.DataFrame(columns=["alert_id", "event_id", "review_status", "review_label", "reviewer", "notes", "reviewed_at"])

    if df.empty:
        return pd.DataFrame(columns=["alert_id", "event_id", "review_status", "review_label", "reviewer", "notes", "reviewed_at"])

    df["reviewed_at"] = pd.to_datetime(df["reviewed_at"])
    return df


def merge_alerts_with_reviews(alerts_df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
    queue_df = alerts_df.copy()
    if reviews_df.empty:
        queue_df["review_status"] = "new"
        queue_df["review_label"] = "unlabeled"
        queue_df["reviewer"] = ""
        queue_df["review_notes"] = ""
        queue_df["reviewed_at"] = ""
        return queue_df

    normalized_reviews = reviews_df.rename(columns={"notes": "review_notes"}).copy()
    normalized_reviews["review_status"] = normalized_reviews["review_status"].fillna("new")
    normalized_reviews["review_label"] = normalized_reviews["review_label"].fillna("unlabeled")
    normalized_reviews["reviewer"] = normalized_reviews["reviewer"].fillna("")
    normalized_reviews["review_notes"] = normalized_reviews["review_notes"].fillna("")
    normalized_reviews["reviewed_at"] = normalized_reviews["reviewed_at"].fillna("")

    queue_df = queue_df.merge(
        normalized_reviews[["alert_id", "event_id", "review_status", "review_label", "reviewer", "review_notes", "reviewed_at"]],
        on=["alert_id", "event_id"],
        how="left",
    )
    queue_df["review_status"] = queue_df["review_status"].fillna("new")
    queue_df["review_label"] = queue_df["review_label"].fillna("unlabeled")
    queue_df["reviewer"] = queue_df["reviewer"].fillna("")
    queue_df["review_notes"] = queue_df["review_notes"].fillna("")
    queue_df["reviewed_at"] = queue_df["reviewed_at"].fillna("")
    return queue_df


def load_unreviewed_alerts(alerts_df: pd.DataFrame) -> pd.DataFrame:
    reviews_df = load_alert_reviews()
    merged_df = merge_alerts_with_reviews(alerts_df, reviews_df)
    return merged_df[merged_df["review_status"] == "new"]


def build_review_queue(alerts_df: pd.DataFrame) -> pd.DataFrame:
    reviews_df = load_alert_reviews()
    return merge_alerts_with_reviews(alerts_df, reviews_df)


def save_alert_review(alert_id: str, event_id: str, review_status: str, review_label: str, reviewer: str, notes: str) -> bool:
    session = get_cassandra_session()
    if not session:
        return False

    reviewed_at = datetime.utcnow()
    try:
        session.execute(
            """
            INSERT INTO alert_reviews (alert_id, event_id, review_status, review_label, reviewer, notes, reviewed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (alert_id, event_id, review_status, review_label, reviewer.strip(), notes.strip(), reviewed_at),
        )
        return True
    except Exception as exc:
        st.error(f"Failed to save review: {exc}")
        return False


def highlight_fraud(row):
    if row["severity"] == "high":
        return ["background-color: rgba(255, 75, 75, 0.2)"] * len(row)
    if row["severity"] == "medium":
        return ["background-color: rgba(255, 165, 0, 0.1)"] * len(row)
    return [""] * len(row)


def load_report(report_name: str) -> Optional[dict]:
    path = REPORT_DIR / report_name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def metric_display(value, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def render_live_alerts(alerts_df: pd.DataFrame, severity_filter: list[str]) -> None:
    filtered_df = alerts_df[alerts_df["severity"].isin(severity_filter)] if not alerts_df.empty else alerts_df

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Alerts", len(alerts_df))
    with m2:
        high_risk = len(alerts_df[alerts_df["severity"] == "high"])
        st.metric("High Severity", high_risk, delta=f"{high_risk / len(alerts_df) * 100:.1f}%" if len(alerts_df) > 0 else "0%")
    with m3:
        st.metric("Avg Risk Score", f"{alerts_df['risk_score'].mean():.2f}")
    with m4:
        st.metric("At-Risk Volume", f"${alerts_df['amount'].sum():,.0f}")

    st.divider()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("📈 Risk Timeline")
        fig_timeline = px.scatter(
            filtered_df,
            x="alert_ts",
            y="risk_score",
            color="severity",
            size="amount",
            hover_data=["account_id", "primary_rule"],
            color_discrete_map={"high": "#FF4B4B", "medium": "#FFA500", "low": "#00D2FF"},
            template="plotly_dark",
        )
        fig_timeline.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_timeline, use_container_width=True)

    with c2:
        st.subheader("📊 Transaction Types")
        fig_pie = px.pie(filtered_df, names="txn_type", values="amount", hole=0.4, template="plotly_dark")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    c3, c4 = st.columns([2, 3])
    with c3:
        st.subheader("🧩 Alert Distribution By Rule")
        exploded_rules = filtered_df[["triggered_rules", "amount"]].explode("triggered_rules")
        exploded_rules["triggered_rules"] = exploded_rules["triggered_rules"].fillna("ml_only")
        rule_counts = (
            exploded_rules.groupby("triggered_rules", as_index=False)
            .agg(alert_count=("triggered_rules", "size"), total_amount=("amount", "sum"))
            .sort_values(["alert_count", "total_amount"], ascending=[False, False])
        )
        fig_rules = px.bar(
            rule_counts.head(10),
            x="triggered_rules",
            y="alert_count",
            color="total_amount",
            template="plotly_dark",
            color_continuous_scale="Bluered",
            labels={"triggered_rules": "Rule", "alert_count": "Alerts", "total_amount": "Amount"},
        )
        fig_rules.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_rules, use_container_width=True)

    with c4:
        st.subheader("🎯 Primary Rule Mix")
        primary_mix = (
            filtered_df.groupby("primary_rule", as_index=False)
            .agg(alert_count=("primary_rule", "size"))
            .sort_values("alert_count", ascending=False)
        )
        fig_primary = px.pie(primary_mix, names="primary_rule", values="alert_count", hole=0.45, template="plotly_dark")
        fig_primary.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_primary, use_container_width=True)

    st.subheader("🚨 Live Alert Feed")
    display_df = filtered_df.copy()
    display_df["status"] = display_df["severity"].apply(
        lambda value: "🔴 CRITICAL" if value == "high" else ("🟠 WARNING" if value == "medium" else "🔵 INFO")
    )
    styled_df = (
        display_df[["status", "severity", "alert_ts", "account_id", "txn_type", "amount", "risk_score", "ml_score", "triggered_rules_display"]]
        .sort_values("alert_ts", ascending=False)
        .style.apply(highlight_fraud, axis=1)
        .hide(axis="columns", subset=["severity"])
    )
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "status": "Status",
            "risk_score": st.column_config.ProgressColumn("Rule Score", min_value=0, max_value=1),
            "ml_score": st.column_config.ProgressColumn("ML Score", min_value=0, max_value=1),
            "amount": st.column_config.NumberColumn("Amount ($)", format="$ %d"),
            "alert_ts": "Time",
            "triggered_rules_display": "Triggered Rules",
        },
    )


def render_review_queue(queue_df: pd.DataFrame) -> None:
    st.subheader("Fraud Analyst Review Queue")
    col1, col2, col3 = st.columns(3)
    with col1:
        queue_status_filter = st.multiselect(
            "Queue Status",
            REVIEW_STATUS_OPTIONS,
            default=["new", "in_review", "escalated", "needs_more_info"],
        )
    with col2:
        severity_filter = st.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium"])
    with col3:
        primary_rule_filter = st.multiselect(
            "Primary Rule",
            sorted(queue_df["primary_rule"].dropna().unique().tolist()),
            default=[],
        )

    filtered_queue = queue_df[queue_df["review_status"].isin(queue_status_filter)]
    filtered_queue = filtered_queue[filtered_queue["severity"].isin(severity_filter)]
    if primary_rule_filter:
        filtered_queue = filtered_queue[filtered_queue["primary_rule"].isin(primary_rule_filter)]
    filtered_queue = filtered_queue.sort_values(["severity", "risk_score", "alert_ts"], ascending=[True, False, False])

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Open Cases", int(filtered_queue["review_status"].isin(["new", "in_review", "escalated", "needs_more_info"]).sum()))
    with q2:
        st.metric("Confirmed Fraud", int((queue_df["review_label"] == "fraud").sum()))
    with q3:
        st.metric("False Positives", int((queue_df["review_label"] == "legit").sum()))
    with q4:
        st.metric("Escalated", int((queue_df["review_status"] == "escalated").sum()))

    st.caption(f"New cases waiting for review: {int((queue_df['review_status'] == 'new').sum())}")

    queue_display = filtered_queue[[
        "event_id", "alert_ts", "account_id", "txn_type", "amount", "risk_score", "severity",
        "primary_rule", "review_status", "review_label", "reviewer"
    ]].copy()
    st.dataframe(
        queue_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=1),
            "amount": st.column_config.NumberColumn("Amount ($)", format="$ %d"),
            "alert_ts": "Alert Time",
            "primary_rule": "Primary Rule",
        },
    )

    event_options = filtered_queue["event_id"].tolist()
    if not event_options:
        st.info("No cases match the current queue filters.")
        return

    default_event_id = st.session_state.selected_alert_id if st.session_state.selected_alert_id in event_options else event_options[0]
    selected_event_id = st.selectbox("Select case", event_options, index=event_options.index(default_event_id))
    st.session_state.selected_alert_id = selected_event_id

    selected_case = filtered_queue.loc[filtered_queue["event_id"] == selected_event_id].iloc[0]
    review_defaults = {
        "review_status": selected_case.get("review_status", "new"),
        "review_label": selected_case.get("review_label", "unlabeled"),
        "reviewer": selected_case.get("reviewer", ""),
        "notes": selected_case.get("review_notes", ""),
    }

    a1, a2, a3, a4 = st.columns(4)
    if a1.button("Mark Fraud", use_container_width=True):
        if save_alert_review(selected_case["alert_id"], selected_event_id, "confirmed_fraud", "fraud", review_defaults.get("reviewer", ""), review_defaults.get("notes", "")):
            st.success("Case marked as confirmed fraud.")
            st.rerun()
    if a2.button("Mark Legit", use_container_width=True):
        if save_alert_review(selected_case["alert_id"], selected_event_id, "false_positive", "legit", review_defaults.get("reviewer", ""), review_defaults.get("notes", "")):
            st.success("Case marked as legitimate.")
            st.rerun()
    if a3.button("Escalate", use_container_width=True):
        if save_alert_review(selected_case["alert_id"], selected_event_id, "escalated", review_defaults.get("review_label", "needs_more_info"), review_defaults.get("reviewer", ""), review_defaults.get("notes", "")):
            st.warning("Case escalated for deeper investigation.")
            st.rerun()
    if a4.button("Needs More Info", use_container_width=True):
        if save_alert_review(selected_case["alert_id"], selected_event_id, "needs_more_info", "needs_more_info", review_defaults.get("reviewer", ""), review_defaults.get("notes", "")):
            st.info("Case moved to needs-more-info.")
            st.rerun()

    with st.form("review_form"):
        reviewer = st.text_input("Reviewer", value=review_defaults.get("reviewer", ""))
        review_status = st.selectbox(
            "Review Status",
            REVIEW_STATUS_OPTIONS,
            index=REVIEW_STATUS_OPTIONS.index(review_defaults.get("review_status", "new")),
        )
        review_label = st.selectbox(
            "Review Label",
            REVIEW_LABEL_OPTIONS,
            index=REVIEW_LABEL_OPTIONS.index(review_defaults.get("review_label", "unlabeled")),
        )
        notes = st.text_area("Analyst Notes", value=review_defaults.get("notes", ""), height=120)
        submitted = st.form_submit_button("Save Review")
        if submitted:
            if save_alert_review(selected_case["alert_id"], selected_event_id, review_status, review_label, reviewer, notes):
                st.success("Review saved to Cassandra.")
                st.rerun()

    st.caption("Review queue hien tai luu persistence that vao Cassandra table `alert_reviews`.")


def render_case_details(queue_df: pd.DataFrame) -> None:
    st.subheader("Case Details")
    event_options = queue_df["event_id"].tolist()
    if not event_options:
        st.info("No alert details available.")
        return

    selected_event_id = st.session_state.selected_alert_id if st.session_state.selected_alert_id in event_options else event_options[0]
    selected_case = queue_df.loc[queue_df["event_id"] == selected_event_id].iloc[0]
    review_info = {
        "review_status": selected_case.get("review_status", "new"),
        "review_label": selected_case.get("review_label", "unlabeled"),
        "reviewer": selected_case.get("reviewer", ""),
        "reviewed_at": selected_case.get("reviewed_at", ""),
        "notes": selected_case.get("review_notes", ""),
    }

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"### Case `{selected_event_id}`")
        detail_rows = pd.DataFrame(
            [
                ("Account", selected_case["account_id"]),
                ("Destination", selected_case["name_dest"] if "name_dest" in selected_case else selected_case.get("nameDest", "n/a")),
                ("Transaction Type", selected_case["txn_type"]),
                ("Amount", f"${selected_case['amount']:,.2f}"),
                ("Severity", selected_case["severity"]),
                ("Risk Score", f"{selected_case['risk_score']:.4f}"),
                ("ML Score", f"{selected_case['ml_score']:.4f}"),
                ("Model Version", selected_case.get("ml_model_version", "n/a")),
                ("Primary Rule", selected_case.get("primary_rule", "ml_only")),
                ("Triggered Rules", selected_case.get("triggered_rules_display", "ml_only")),
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(detail_rows, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Review Snapshot")
        st.metric("Queue Status", review_info.get("review_status", "new"))
        st.metric("Review Label", review_info.get("review_label", "unlabeled"))
        st.write(f"**Reviewer:** {review_info.get('reviewer', 'unassigned') or 'unassigned'}")
        st.write(f"**Reviewed At:** {review_info.get('reviewed_at', 'pending')}")
        st.write(f"**Notes:** {review_info.get('notes', 'No notes yet.') or 'No notes yet.'}")

    timeline_source = queue_df.sort_values("alert_ts", ascending=True).copy()
    timeline_source["selected"] = timeline_source["event_id"].apply(lambda event_id: "selected" if event_id == selected_event_id else "other")
    fig = px.scatter(
        timeline_source.tail(100),
        x="alert_ts",
        y="risk_score",
        color="selected",
        size="amount",
        hover_data=["event_id", "account_id", "primary_rule"],
        color_discrete_map={"selected": "#00D2FF", "other": "#555555"},
        template="plotly_dark",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_monitoring_tab(queue_df: pd.DataFrame) -> None:
    st.subheader("Model Monitoring Dashboard")
    drift_report = load_report("drift_report.json")
    performance_report = load_report("performance_report.json")
    retraining_report = load_report("retraining_decision.json")

    if not drift_report or not performance_report or not retraining_report:
        st.warning(
            "Model monitoring artifacts are missing. Run the monitoring scripts in `monitoring/model/` to populate drift, performance, and retraining reports."
        )
        return

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Drifted Features", drift_report.get("drifted_feature_count", 0))
    with top2:
        st.metric("Label Coverage", metric_display(performance_report.get("label_coverage")))
    with top3:
        precision_7d = performance_report.get("rolling_windows", {}).get("7d", {}).get("precision")
        st.metric("7D Precision", metric_display(precision_7d))
    with top4:
        retrain_required = retraining_report.get("retrain_required", False)
        st.metric("Retrain Required", "YES" if retrain_required else "NO")

    st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Drift Summary")
        drift_rows = []
        for feature_name, details in drift_report.get("features", {}).items():
            drift_rows.append(
                {
                    "feature": feature_name,
                    "metric": details.get("metric", "n/a"),
                    "drift_detected": details.get("drift_detected", False),
                    "observed": details.get("ks_statistic", details.get("total_variation_distance", details.get("mean_delta", "n/a"))),
                }
            )
        drift_df = pd.DataFrame(drift_rows)
        if not drift_df.empty:
            fig_drift = px.bar(
                drift_df,
                x="feature",
                y="observed",
                color="drift_detected",
                template="plotly_dark",
                color_discrete_map={True: "#FF4B4B", False: "#00D2FF"},
            )
            fig_drift.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_drift, use_container_width=True)
            st.dataframe(drift_df, use_container_width=True, hide_index=True)
        else:
            st.info("No drift features available in the current report.")

    with right:
        st.subheader("Retraining Decision")
        if retrain_required:
            st.error("Retraining is recommended based on current monitoring signals.")
        else:
            st.success("Current policy does not require retraining yet.")

        observations = retraining_report.get("observations", {})
        st.write(f"**Drift Ratio:** {metric_display(observations.get('drift_ratio'))}")
        st.write(f"**Alert Rate Delta:** {metric_display(observations.get('alert_rate_delta'))}")
        st.write(f"**7D Recall:** {metric_display(observations.get('rolling_7d_recall'))}")
        st.write(f"**7D F1:** {metric_display(observations.get('rolling_7d_f1'))}")

        reasons = retraining_report.get("reasons", [])
        warnings = retraining_report.get("warnings", [])
        if reasons:
            st.markdown("**Reasons**")
            for reason in reasons:
                st.write(f"- {reason.get('message', reason.get('type', 'unknown_reason'))}")
        if warnings:
            st.markdown("**Warnings**")
            for warning in warnings:
                st.write(f"- {warning}")

    mid1, mid2 = st.columns(2)
    with mid1:
        st.subheader("Rolling Performance")
        rolling_windows = performance_report.get("rolling_windows", {})
        performance_rows = []
        for window_name, details in rolling_windows.items():
            performance_rows.append(
                {
                    "window": window_name,
                    "precision": details.get("precision"),
                    "recall": details.get("recall"),
                    "f1": details.get("f1"),
                    "false_positive_rate": details.get("false_positive_rate"),
                    "labeled_rows": details.get("labeled_rows"),
                }
            )
        perf_df = pd.DataFrame(performance_rows)
        if not perf_df.empty:
            perf_long = perf_df.melt(
                id_vars=["window", "labeled_rows"],
                value_vars=["precision", "recall", "f1", "false_positive_rate"],
                var_name="metric",
                value_name="value",
            )
            fig_perf = px.bar(
                perf_long,
                x="window",
                y="value",
                color="metric",
                barmode="group",
                template="plotly_dark",
            )
            fig_perf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_perf, use_container_width=True)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
        else:
            st.info("No rolling performance metrics available yet.")

    with mid2:
        st.subheader("Score Distribution Snapshot")
        score_df = pd.DataFrame(
            [
                {
                    "metric": "risk_score",
                    "reference_mean": drift_report.get("features", {}).get("risk_score", {}).get("reference_mean"),
                    "serving_mean": drift_report.get("features", {}).get("risk_score", {}).get("serving_mean"),
                },
                {
                    "metric": "ml_score",
                    "reference_mean": drift_report.get("features", {}).get("ml_score", {}).get("reference_mean"),
                    "serving_mean": drift_report.get("features", {}).get("ml_score", {}).get("serving_mean"),
                },
                {
                    "metric": "amount",
                    "reference_mean": drift_report.get("features", {}).get("amount", {}).get("reference_mean"),
                    "serving_mean": drift_report.get("features", {}).get("amount", {}).get("serving_mean"),
                },
            ]
        )
        score_long = score_df.melt(id_vars="metric", var_name="source", value_name="mean_value")
        fig_score = px.bar(
            score_long,
            x="metric",
            y="mean_value",
            color="source",
            barmode="group",
            template="plotly_dark",
        )
        fig_score.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_score, use_container_width=True)
        st.dataframe(score_df, use_container_width=True, hide_index=True)

    bottom1, bottom2 = st.columns(2)
    with bottom1:
        st.subheader("Review Workflow Context")
        review_counts = (
            queue_df.groupby("review_status", as_index=False)
            .agg(case_count=("event_id", "size"))
            .sort_values("case_count", ascending=False)
        )
        fig_status = px.bar(review_counts, x="review_status", y="case_count", template="plotly_dark", color="case_count")
        fig_status.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_status, use_container_width=True)

    with bottom2:
        st.subheader("Label Source And Coverage")
        label_source_breakdown = performance_report.get("label_source_breakdown", {})
        source_df = pd.DataFrame(
            [{"source": key, "count": value} for key, value in label_source_breakdown.items()]
        )
        if not source_df.empty:
            fig_source = px.pie(source_df, names="source", values="count", hole=0.45, template="plotly_dark")
            fig_source.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_source, use_container_width=True)
        st.caption(
            f"Prediction rows: {performance_report.get('prediction_rows', 0)} | Labeled rows: {performance_report.get('labeled_rows', 0)} | Review rows: {performance_report.get('review_rows', 0)}"
        )
    label_counts = (
        queue_df.groupby("review_label", as_index=False)
        .agg(case_count=("event_id", "size"))
        .sort_values("case_count", ascending=False)
    )

    st.subheader("Review Label Mix")
    fig_label = px.pie(label_counts, names="review_label", values="case_count", hole=0.45, template="plotly_dark")
    fig_label.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_label, use_container_width=True)

    st.info(
        "Monitoring tab hien tai theo doi workflow analyst dua tren review persistence that. Drift, precision/recall rolling window, va retraining trigger se duoc bo sung o cac phase monitoring tiep theo."
    )


def main() -> None:
    ensure_review_state()

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
        st.title("Control Panel")
        auto_refresh = st.toggle("Auto-refresh", value=False)
        refresh_rate = st.slider("Refresh interval (seconds)", 5, 60, 10)
        severity_filter = st.multiselect("Live Severity Filter", ["high", "medium", "low"], default=["high", "medium"])
        st.divider()
        st.info("System Status: **ACTIVE** 🟢")

    col_header, col_status = st.columns([4, 1])
    with col_header:
        st.title("🛡️ Fraud Detection Command Center")
        st.markdown("*Real-time Hybrid Intelligence Pipeline Monitoring And Fraud Analyst Review Queue*")
    with col_status:
        st.write(f"**Last Sync:** {datetime.now().strftime('%H:%M:%S')}")

    alerts_df = load_alerts()
    if alerts_df.empty:
        st.warning("📡 Waiting for live stream data... Please ensure Spark Job and Ingestion are running.")
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
        return

    queue_df = build_review_queue(alerts_df)

    tabs = st.tabs(["Live Alerts", "Review Queue", "Case Details", "Monitoring"])
    with tabs[0]:
        render_live_alerts(alerts_df, severity_filter)
    with tabs[1]:
        render_review_queue(queue_df)
    with tabs[2]:
        render_case_details(queue_df)
    with tabs[3]:
        render_monitoring_tab(queue_df)

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
