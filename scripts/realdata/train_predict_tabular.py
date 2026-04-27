from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realdata_experts.predict_tabular.artifact import train_artifact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train deployable tabular predict artifact.")
    parser.add_argument("--dataset-file", default="data/realdata_inputs/prediction_dataset.jsonl")
    parser.add_argument("--output-dir", default="runs/realdata_experts/predict_tabular_elastic_net_flat_residual")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--max-peers", type=int, default=32)
    parser.add_argument("--feature-set", choices=("summary", "flat"), default="flat")
    parser.add_argument("--target-mode", choices=("direct", "residual"), default="residual")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = train_artifact(
        dataset_file=args.dataset_file,
        output_dir=args.output_dir,
        seed=args.seed,
        history_len=args.history_len,
        max_peers=args.max_peers,
        feature_set=args.feature_set,
        target_mode=args.target_mode,
    )
    print(
        json.dumps(
            {
                "artifact_path": payload["artifact_path"],
                "model_name": payload["model_name"],
                "feature_set": payload["feature_set"],
                "target_mode": payload["target_mode"],
                "test_revenue_mape": payload["test_metrics"]["revenue_mape"],
                "test_growth_mse": payload["test_metrics"]["growth_mse"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
