from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fraud_pipeline.rules import combine_risk_score_arrays
from model.model_utils import load_model_artifact, predict_frame
from model.train_models import (
    FALSE_ALARM_SENSITIVITY_COSTS,
    evaluate_scores,
    run_ablation_from_artifacts,
    tune_threshold_by_business_cost,
)


def _load_validation_scores(
    artifacts: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    artifact = load_model_artifact(
        artifacts / "fraud_pipeline_selected.joblib",
        strict=True,
        force_reload=True,
    )
    if artifact is None:
        raise RuntimeError("Selected model artifact could not be loaded")
    features = [str(name) for name in artifact["feature_columns"]]
    columns = list(
        dict.fromkeys(["row_id", "label", "amount", "rule_score", *features])
    )
    validation = pd.read_parquet(
        artifacts / "validation_features.parquet",
        columns=columns,
    )
    ml_scores = predict_frame(artifact, validation)
    rule_scores = validation["rule_score"].to_numpy(dtype=np.float64)
    hybrid_scores = combine_risk_score_arrays(
        rule_scores,
        ml_scores,
        float(artifact["rule_weight"]),
        float(artifact["ml_weight"]),
    )
    return artifact, validation, {
        "rule_only": rule_scores,
        "ml_only": ml_scores,
        "hybrid": hybrid_scores,
    }


def run_sensitivity_analysis(artifacts: Path) -> Path:
    _, validation, scores_by_system = _load_validation_scores(artifacts)
    labels = validation["label"].to_numpy(dtype=np.int8)
    amounts = validation["amount"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for unit_cost in FALSE_ALARM_SENSITIVITY_COSTS:
        for system, scores in scores_by_system.items():
            tuning = tune_threshold_by_business_cost(
                labels,
                scores,
                amounts,
                unit_cost,
            )
            metrics = evaluate_scores(
                labels,
                scores,
                amounts,
                float(tuning["threshold"]),
                false_alarm_unit_cost=unit_cost,
            )
            rows.append(
                {
                    "system": system,
                    "false_alarm_unit_cost": unit_cost,
                    "selected_threshold": metrics["selected_threshold"],
                    "average_precision": metrics["average_precision"],
                    "roc_auc": metrics["roc_auc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "alert_rate": metrics["alert_rate"],
                    "business_cost": metrics["business_cost"],
                    "missed_fraud_cost": metrics["missed_fraud_cost"],
                    "false_alarm_cost": metrics["false_alarm_cost"],
                    "false_alarm_total_cost": metrics["false_alarm_cost"],
                    "no_model_baseline_cost": metrics["no_model_baseline_cost"],
                    "net_cost_savings": metrics["net_cost_savings"],
                    "savings_rate": metrics["savings_rate"],
                }
            )
    output = artifacts / "sensitivity_analysis.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def run_selected_model_ablation(artifacts: Path, max_train_rows: int) -> Path:
    artifact = load_model_artifact(
        artifacts / "fraud_pipeline_selected.joblib",
        strict=True,
        force_reload=True,
    )
    if artifact is None:
        raise RuntimeError("Selected model artifact could not be loaded")
    results = run_ablation_from_artifacts(
        artifacts,
        false_alarm_unit_cost=float(artifact["cost_config"]["false_alarm_unit_cost"]),
        random_state=int(artifact.get("random_state", 42)),
        max_train_rows=max_train_rows,
        model_type=str(artifact["model_tag"]),
    )
    output = artifacts / "ablation_results.csv"
    results.to_csv(output, index=False)
    return output


def _downsample_curve(*values: np.ndarray, max_points: int = 5_000) -> list[np.ndarray]:
    if not values or len(values[0]) <= max_points:
        return [np.asarray(value) for value in values]
    indices = np.linspace(0, len(values[0]) - 1, max_points, dtype=np.int64)
    return [np.asarray(value)[indices] for value in values]


def _business_cost_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    false_alarm_unit_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="stable")[::-1]
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ordered_amounts = amounts[order]
    baseline = float(ordered_amounts[ordered_labels == 1].sum())
    detected = np.cumsum(
        np.where(ordered_labels == 1, ordered_amounts, 0.0),
        dtype=np.float64,
    )
    false_positives = np.cumsum((ordered_labels == 0).astype(np.int64))
    changes = np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:])
    ends = np.r_[changes, len(ordered_scores) - 1]
    thresholds = ordered_scores[ends]
    costs = (
        baseline
        - detected[ends]
        + false_alarm_unit_cost * false_positives[ends]
    )
    order_by_threshold = np.argsort(thresholds)
    return thresholds[order_by_threshold], costs[order_by_threshold]


def create_modeling_plots(artifacts: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

    artifact, validation, scores_by_system = _load_validation_scores(artifacts)
    labels = validation["label"].to_numpy(dtype=np.int8)
    amounts = validation["amount"].to_numpy(dtype=np.float64)
    plot_dir = artifacts / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    fig, axis = plt.subplots(figsize=(8, 6))
    colors = {"rule_only": "#8c6d31", "ml_only": "#1f77b4", "hybrid": "#c44e52"}
    for system, scores in scores_by_system.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        recall, precision = _downsample_curve(recall, precision)
        ap = average_precision_score(labels, scores)
        axis.plot(recall, precision, label=f"{system} (AP={ap:.4f})", color=colors[system])
    axis.set(xlabel="Recall", ylabel="Precision", title="Validation Precision-Recall Curves")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    path = plot_dir / "precision_recall_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["precision_recall_curves"] = str(path)

    unit_cost = float(artifact["cost_config"]["false_alarm_unit_cost"])
    thresholds, costs = _business_cost_curve(
        labels,
        scores_by_system["hybrid"],
        amounts,
        unit_cost,
    )
    thresholds, costs = _downsample_curve(thresholds, costs)
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.plot(thresholds, costs, color="#c44e52")
    axis.axvline(float(artifact["hybrid_threshold"]), color="#222222", linestyle="--")
    axis.set(
        xlabel="Hybrid threshold",
        ylabel="Validation business cost",
        title=f"Validation Business Cost vs Threshold (false alarm cost={unit_cost:g})",
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path = plot_dir / "validation_business_cost_vs_threshold.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["validation_business_cost_vs_threshold"] = str(path)

    predictions = pd.read_parquet(artifacts / "test_predictions.parquet")
    matrix = confusion_matrix(
        predictions["label"].to_numpy(dtype=np.int8),
        predictions["prediction"].to_numpy(dtype=np.int8),
        labels=[0, 1],
    )
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    axis.set_xticks([0, 1], labels=["Legitimate", "Fraud alert"])
    axis.set_yticks([0, 1], labels=["Legitimate", "Fraud"])
    axis.set(xlabel="Predicted", ylabel="Actual", title="Selected Hybrid Test Confusion Matrix")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    path = plot_dir / "selected_hybrid_confusion_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["selected_hybrid_confusion_matrix"] = str(path)

    evaluation = json.loads(
        (artifacts / "evaluation_results.json").read_text(encoding="utf-8")
    )["evaluation"]
    systems = ["rule_only", "ml_only", "hybrid"]
    costs = [float(evaluation[name]["business_cost"]) for name in systems]
    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(systems, costs, color=[colors[name] for name in systems])
    axis.bar_label(bars, labels=[f"{value:,.0f}" for value in costs], padding=3)
    axis.set(ylabel="Test business cost", title="Rule-only vs ML-only vs Hybrid")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = plot_dir / "rule_ml_hybrid_business_cost.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["rule_ml_hybrid_business_cost"] = str(path)

    transformed_names = np.asarray(artifact["preprocessor"].get_feature_names_out())
    raw_importance = np.asarray(artifact["model"].feature_importances_, dtype=np.float64)
    if len(transformed_names) != len(raw_importance):
        raise ValueError("Model importance count differs from transformed feature names")
    importance_rows = []
    for feature in artifact["feature_columns"]:
        mask = (transformed_names == feature) | np.char.startswith(
            transformed_names.astype(str), f"{feature}_"
        )
        importance_rows.append(
            {"feature": feature, "importance": float(raw_importance[mask].sum())}
        )
    importance = pd.DataFrame(importance_rows).sort_values("importance", ascending=True)
    importance.to_csv(artifacts / "selected_model_feature_importance.csv", index=False)
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(importance["feature"], importance["importance"], color="#1f77b4")
    axis.set(xlabel="Aggregated model importance", title="Selected XGBoost Feature Importance")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = plot_dir / "selected_model_feature_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["selected_model_feature_importance"] = str(path)

    manifest = plot_dir / "plot_manifest.json"
    manifest.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run post-training fraud model analysis")
    parser.add_argument("--artifacts-dir", default="model/artifacts")
    parser.add_argument(
        "--analysis",
        choices=("sensitivity", "ablation", "plots"),
        required=True,
    )
    parser.add_argument("--ablation-max-rows", type=int, default=500_000)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    artifacts = Path(args.artifacts_dir)
    if args.analysis == "sensitivity":
        output = run_sensitivity_analysis(artifacts)
    elif args.analysis == "ablation":
        if args.ablation_max_rows <= 0:
            raise ValueError("--ablation-max-rows must be positive")
        output = run_selected_model_ablation(artifacts, args.ablation_max_rows)
    else:
        output = create_modeling_plots(artifacts)
    print(f"Completed {args.analysis}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
