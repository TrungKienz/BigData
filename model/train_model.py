"""Compatibility entry point for the two-stage feature and model pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.prepare_features import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_PATH,
    prepare_feature_artifacts,
)
from model.train_models import parse_model_types, train_and_export_models


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare cleaned features, then train fraud models"
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Deprecated alias for --data-path; input must still use the cleaned schema",
    )
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument(
        "--feature-config",
        default="deployment_safe",
        choices=("deployment_safe", "full_paysim"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--model-types", default="all")
    parser.add_argument("--model-type", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--false-alarm-cost", type=float, default=5.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    data_path = args.csv_path or args.data_path
    prepare_feature_artifacts(
        data_path,
        args.artifacts_dir,
        feature_configuration=args.feature_config,
        limit=args.limit,
        sample_size=args.sample_size,
        force_split=args.force_split,
        chunk_size=args.chunk_size,
        in_memory=args.in_memory,
    )
    result = train_and_export_models(
        args.artifacts_dir,
        model_types=parse_model_types(args.model_type or args.model_types),
        false_alarm_unit_cost=args.false_alarm_cost,
        quick=args.quick,
        run_feature_ablation=not args.skip_ablation,
    )
    print(f"Pipeline complete: {Path(result['artifact_path']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
