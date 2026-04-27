from __future__ import annotations

import argparse
import json
from pathlib import Path

from .realdata_query_bank import build_realdata_query_bank
from .router import MoERouter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a retrieval-based MoE router from the real-data dataset queries."
    )
    parser.add_argument("--output-dir", type=str, default="data/moe_router_realdata_v1")
    return parser


def build_router_assets(output_dir: str) -> dict[str, object]:
    records = build_realdata_query_bank()
    router = MoERouter().fit(records)
    output_path = router.save(output_dir)

    expert_counts: dict[str, int] = {}
    for record in records:
        expert_counts[record.expert] = expert_counts.get(record.expert, 0) + 1

    return {
        "output_dir": str(Path(output_path).resolve()),
        "num_queries": len(records),
        "expert_counts": expert_counts,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = build_router_assets(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

