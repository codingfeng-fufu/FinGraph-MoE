from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_router.realdata_demo import RealdataDemoService


DEFAULT_DOC = ROOT / "data" / "demo" / "query_conditioned_small_report.md"
DEFAULT_QUERIES = (
    "LIFE INSURANCE行业的收入总和是多少",
    "请统计 revenue >= 1000 的公司数量",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end small-document MoE demo using the uploaded-document path."
    )
    parser.add_argument(
        "--document",
        type=str,
        default=str(DEFAULT_DOC),
        help="Path to the small demo document.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Query to run. Can be passed multiple times. Defaults to two demo queries.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    document_path = Path(args.document).resolve()
    content = document_path.read_bytes()

    service = RealdataDemoService(ROOT / "data" / "moe_router_realdata_v1")
    session = service.create_document_session(
        file_name=document_path.name,
        media_type="text/markdown",
        content=content,
    )

    queries = args.query or list(DEFAULT_QUERIES)
    summary: dict[str, object] = {
        "document": {
            "path": str(document_path),
            "session_id": session["session_id"],
            "preview_text": session["preview_text"],
            "parse_summary": session["parse_summary"],
            "graph_summary": session["graph_summary"],
            "query_examples": session["query_examples"],
        },
        "runs": [],
    }

    for query in queries:
        result = service.run(query, session_id=session["session_id"])
        summary["runs"].append(
            {
                "query": query,
                "route": result["route"],
                "execute": result["execute"],
                "result": result["result"],
            }
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
