from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_router.document_graph import FinancialRecord, build_parsed_document_from_records
from moe_router.graph_extraction import GraphExtractionResult
from moe_router.realdata_demo import RealdataDemoService


DEFAULT_BENCHMARK = ROOT / "data" / "demo" / "qa_benchmark_v1" / "qa_dataset.json"
DEFAULT_OUTPUT = ROOT / "runs" / "demo_benchmarks" / "qa_benchmark_v1_results.json"
DEFAULT_CACHE_DIR = ROOT / "runs" / "demo_benchmarks" / "qa_benchmark_v1_graph_cache"
CACHE_VERSION = "qa_benchmark_graph_extraction_v1"


@dataclass(frozen=True)
class QARecord:
    qa_id: str
    domain_id: str
    document_file: str
    expected_expert: str
    query: str
    answer: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 50-QA small-document product demo benchmark."
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default=str(DEFAULT_BENCHMARK),
        help="Path to the QA benchmark dataset JSON file.",
    )
    parser.add_argument(
        "--router-dir",
        type=str,
        default=str(ROOT / "data" / "moe_router_realdata_v1"),
        help="Path to the router assets directory.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to write the JSON benchmark results.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Directory for cached LLM graph extraction results.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable graph extraction caching.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing cached graph extractions and rewrite them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N QA records for smoke testing.",
    )
    return parser


def _load_benchmark(path: Path) -> tuple[dict[str, Any], list[QARecord]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [
        QARecord(
            qa_id=item["qa_id"],
            domain_id=item["domain_id"],
            document_file=item["document_file"],
            expected_expert=item["expected_expert"],
            query=item["query"],
            answer=float(item["answer"]),
        )
        for item in payload["qas"]
    ]
    return payload, records


def _guess_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _record_to_dict(record: FinancialRecord) -> dict[str, Any]:
    return {
        "company_name": record.company_name,
        "industry": record.industry,
        "revenue": record.revenue,
        "operating_profit": record.operating_profit,
        "net_profit": record.net_profit,
        "employees": record.employees,
        "year": record.year,
        "source_index": record.source_index,
        "raw": record.raw,
    }


def _record_from_dict(row: dict[str, Any], index: int) -> FinancialRecord:
    return FinancialRecord(
        company_name=str(row["company_name"]),
        industry=row.get("industry"),
        revenue=_optional_float(row.get("revenue")),
        operating_profit=_optional_float(row.get("operating_profit")),
        net_profit=_optional_float(row.get("net_profit")),
        employees=_optional_float(row.get("employees")),
        year=int(row["year"]) if row.get("year") is not None else None,
        source_index=int(row.get("source_index", index)),
        raw=dict(row.get("raw") or {}),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _install_graph_extraction_cache(
    service: RealdataDemoService,
    cache_dir: Path,
    refresh_cache: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    uncached_extract = service.llm_extractor.extract_with_metadata

    def cached_extract(uploaded_document, query: str, expert_id: str) -> GraphExtractionResult:
        cache_path = cache_dir / f"{_cache_key(service, uploaded_document, query, expert_id)}.json"
        if cache_path.exists() and not refresh_cache:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            records = [
                _record_from_dict(row, index)
                for index, row in enumerate(payload["records"])
            ]
            document = build_parsed_document_from_records(
                uploaded_document.file_name,
                uploaded_document.media_type,
                records,
                source_format=payload.get("source_format", "llm_query_conditioned_cached"),
            )
            metadata = dict(payload["metadata"])
            metadata["cache"] = {
                "hit": True,
                "path": str(cache_path),
                "version": payload.get("cache_version"),
            }
            return GraphExtractionResult(document=document, metadata=metadata)

        result = uncached_extract(uploaded_document, query, expert_id)
        payload = {
            "cache_version": CACHE_VERSION,
            "source_format": result.document.source_format,
            "file_name": uploaded_document.file_name,
            "media_type": uploaded_document.media_type,
            "expert_id": expert_id,
            "query": query,
            "model": service.llm_extractor.config.model,
            "records": [_record_to_dict(record) for record in result.document.records],
            "metadata": result.metadata,
        }
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        metadata = dict(result.metadata)
        metadata["cache"] = {
            "hit": False,
            "path": str(cache_path),
            "version": CACHE_VERSION,
        }
        return GraphExtractionResult(document=result.document, metadata=metadata)

    service.llm_extractor.extract_with_metadata = cached_extract


def _cache_key(
    service: RealdataDemoService,
    uploaded_document,
    query: str,
    expert_id: str,
) -> str:
    material = {
        "version": CACHE_VERSION,
        "model": service.llm_extractor.config.model,
        "file_name": uploaded_document.file_name,
        "media_type": uploaded_document.media_type,
        "document_sha256": hashlib.sha256(uploaded_document.text.encode("utf-8")).hexdigest(),
        "expert_id": expert_id,
        "query": query,
    }
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _safe_mape(predictions: list[float], answers: list[float]) -> float:
    if not predictions:
        return 0.0
    ratios = []
    for prediction, answer in zip(predictions, answers, strict=True):
        denom = max(abs(answer), 1e-6)
        ratios.append(abs(prediction - answer) / denom)
    return float(sum(ratios) / len(ratios))


def _aggregate_numeric(predictions: list[float], answers: list[float]) -> dict[str, float]:
    if not predictions:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "median_abs_error": 0.0,
        }

    abs_errors = [abs(pred - ans) for pred, ans in zip(predictions, answers, strict=True)]
    squared_errors = [(pred - ans) ** 2 for pred, ans in zip(predictions, answers, strict=True)]
    return {
        "mae": float(sum(abs_errors) / len(abs_errors)),
        "rmse": float(math.sqrt(sum(squared_errors) / len(squared_errors))),
        "mape": _safe_mape(predictions, answers),
        "median_abs_error": float(statistics.median(abs_errors)),
    }


def _aggregate_bucket(entries: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [float(entry["prediction"]) for entry in entries]
    answers = [float(entry["answer"]) for entry in entries]
    route_correct = sum(1 for entry in entries if entry["route_correct"])
    graph_quality_scores = [float(entry["graph_quality_score"]) for entry in entries]
    expert_ready = sum(1 for entry in entries if entry["graph_expert_ready"])
    bucket = {
        "num_examples": len(entries),
        "route_accuracy": route_correct / max(len(entries), 1),
        "graph_quality_score": sum(graph_quality_scores) / max(len(graph_quality_scores), 1),
        "graph_expert_ready_rate": expert_ready / max(len(entries), 1),
        "graph_validation_warnings": sum(int(entry["graph_validation_warning_count"]) for entry in entries),
        "graph_validation_errors": sum(int(entry["graph_validation_error_count"]) for entry in entries),
        **_aggregate_numeric(predictions, answers),
    }

    count_entries = [entry for entry in entries if entry["expected_expert"] == "count"]
    if count_entries:
        rounded_exact = sum(
            1
            for entry in count_entries
            if round(float(entry["prediction"])) == round(float(entry["answer"]))
        )
        bucket["count_rounded_exact_match"] = rounded_exact / len(count_entries)
    return bucket


def run_benchmark(
    benchmark_file: Path,
    router_dir: Path,
    output_file: Path,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    benchmark_payload, qa_records = _load_benchmark(benchmark_file)
    if limit is not None:
        qa_records = qa_records[: max(limit, 0)]
    benchmark_root = benchmark_file.parent
    service = RealdataDemoService(router_dir)
    if cache_dir is not None:
        _install_graph_extraction_cache(service, cache_dir, refresh_cache)

    sessions_by_document: dict[str, str] = {}
    document_cache: dict[str, dict[str, Any]] = {}
    detailed_runs: list[dict[str, Any]] = []

    for index, qa in enumerate(qa_records, start=1):
        print(f"[{index}/{len(qa_records)}] {qa.qa_id} -> {qa.expected_expert}", flush=True)
        document_path = (benchmark_root / qa.document_file).resolve()
        document_key = str(document_path)

        if document_key not in sessions_by_document:
            session = service.create_document_session(
                file_name=document_path.name,
                media_type=_guess_media_type(document_path),
                content=document_path.read_bytes(),
            )
            sessions_by_document[document_key] = session["session_id"]
            document_cache[document_key] = {
                "document_file": qa.document_file,
                "session_id": session["session_id"],
                "query_examples": session["query_examples"],
                "parse_summary": session["parse_summary"],
                "graph_summary": session["graph_summary"],
            }

        session_id = sessions_by_document[document_key]
        result = service.run(qa.query, session_id=session_id)
        prediction = float(result["result"]["prediction"])
        answer = float(qa.answer)
        route_correct = result["route"]["selected_expert"] == qa.expected_expert
        graph_validation = result.get("graph_extraction", {}).get("validation", {})
        graph_cache = result.get("graph_extraction", {}).get("cache", {})

        detailed_runs.append(
            {
                "qa_id": qa.qa_id,
                "domain_id": qa.domain_id,
                "document_file": qa.document_file,
                "expected_expert": qa.expected_expert,
                "selected_expert": result["route"]["selected_expert"],
                "route_correct": route_correct,
                "query": qa.query,
                "answer": answer,
                "prediction": prediction,
                "abs_error": abs(prediction - answer),
                "confidence": float(result["route"]["confidence"]),
                "graph_mode": result["execute"]["graph_build"]["extraction_mode"],
                "graph_quality_score": float(graph_validation.get("quality_score", 0.0)),
                "graph_expert_ready": bool(graph_validation.get("expert_ready", False)),
                "graph_validation_warning_count": len(graph_validation.get("warnings", [])),
                "graph_validation_error_count": len(graph_validation.get("errors", [])),
                "graph_validation_warnings": graph_validation.get("warnings", []),
                "graph_validation_errors": graph_validation.get("errors", []),
                "graph_cache_hit": bool(graph_cache.get("hit", False)),
                "graph_cache_path": graph_cache.get("path"),
                "graph_evidence_span_count": int(
                    result.get("graph_extraction", {})
                    .get("draft_summary", {})
                    .get("evidence_span_count", 0)
                ),
                "num_nodes": int(result["execute"]["num_nodes"]),
                "num_edges": int(result["execute"]["num_edges"]),
                "count_rounded_prediction": int(round(prediction))
                if qa.expected_expert == "count"
                else None,
            }
        )

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_expert: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in detailed_runs:
        by_domain[run["domain_id"]].append(run)
        by_expert[run["expected_expert"]].append(run)

    summary = {
        "benchmark_id": benchmark_payload["benchmark_id"],
        "description": benchmark_payload["description"],
        "num_documents": len(document_cache),
        "num_qas": len(detailed_runs),
        "cache": {
            "enabled": cache_dir is not None,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "refresh_cache": refresh_cache,
            "hits": sum(1 for run in detailed_runs if run.get("graph_cache_hit")),
            "misses": sum(1 for run in detailed_runs if not run.get("graph_cache_hit")),
        },
        "documents": list(document_cache.values()),
        "overall": _aggregate_bucket(detailed_runs),
        "by_domain": {domain_id: _aggregate_bucket(entries) for domain_id, entries in by_domain.items()},
        "by_expected_expert": {
            expert_id: _aggregate_bucket(entries) for expert_id, entries in by_expert.items()
        },
        "runs": detailed_runs,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_benchmark(
        benchmark_file=Path(args.benchmark_file).resolve(),
        router_dir=Path(args.router_dir).resolve(),
        output_file=Path(args.output_file).resolve(),
        cache_dir=None if args.no_cache else Path(args.cache_dir).resolve(),
        refresh_cache=bool(args.refresh_cache),
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
