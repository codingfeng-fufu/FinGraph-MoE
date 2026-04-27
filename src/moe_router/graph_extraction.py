from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .document_graph import ParsedFinancialDocument


YEAR_PATTERN = re.compile(r"(19\d{2}|20\d{2})")
METRIC_ALIASES = {
    "revenue": ("revenue", "sales", "turnover", "收入", "营收", "营业收入"),
    "operating_profit": ("operating profit", "operating_profit", "营业利润", "经营利润"),
    "net_profit": ("net profit", "net_profit", "净利润", "利润"),
    "employees": ("employees", "headcount", "员工", "人数", "员工数"),
}


@dataclass(frozen=True)
class GraphExtractionResult:
    document: ParsedFinancialDocument
    metadata: dict[str, Any]


def build_extraction_plan(query: str, expert_id: str) -> dict[str, Any]:
    years = sorted({int(match) for match in YEAR_PATTERN.findall(query)})
    required_fields = ["company_name"]
    graph_scope = "query_slice"
    needs_candidate_universe = False
    needs_history = False
    needs_peer_previous_year = False

    if expert_id == "sum":
        required_fields.extend(["industry", "year", _infer_metric(query, default="revenue")])
        graph_scope = "aggregation_slice"
    elif expert_id == "count":
        required_fields.extend(["industry", "year", *_infer_count_fields(query)])
        graph_scope = "candidate_universe"
        needs_candidate_universe = True
    elif expert_id == "predict":
        required_fields.extend(["year", "revenue", "employees"])
        graph_scope = "target_history_and_peer_growth"
        needs_history = True
        needs_peer_previous_year = True
    else:
        required_fields.extend(["year", "revenue"])

    return {
        "expert_id": expert_id,
        "query": query,
        "graph_scope": graph_scope,
        "required_fields": sorted(set(required_fields)),
        "query_years": years,
        "needs_candidate_universe": needs_candidate_universe,
        "needs_history": needs_history,
        "needs_peer_previous_year": needs_peer_previous_year,
    }


def normalize_llm_extraction_metadata(
    payload: dict[str, Any],
    plan: dict[str, Any],
    parsed_document: ParsedFinancialDocument,
) -> dict[str, Any]:
    evidence_spans = _normalize_evidence_spans(payload.get("evidence_spans"))
    edges = _normalize_edges(payload.get("edges"))
    units = payload.get("units") if isinstance(payload.get("units"), dict) else {}
    assumptions = payload.get("assumptions") if isinstance(payload.get("assumptions"), list) else []
    confidence = _coerce_confidence(payload.get("confidence"))

    return {
        "source": "llm",
        "plan": plan,
        "llm": {
            "confidence": confidence,
            "units": units,
            "edges": edges,
            "evidence_spans": evidence_spans,
            "assumptions": [str(item) for item in assumptions if str(item).strip()][:8],
        },
        "draft_summary": {
            "record_count": len(parsed_document.records),
            "edge_count": len(edges),
            "evidence_span_count": len(evidence_spans),
        },
    }


def build_structured_extraction_metadata(
    document: ParsedFinancialDocument,
    query: str,
    expert_id: str,
) -> dict[str, Any]:
    plan = build_extraction_plan(query, expert_id)
    return {
        "source": "structured_fallback",
        "plan": plan,
        "llm": {
            "confidence": None,
            "units": {},
            "edges": [],
            "evidence_spans": [],
            "assumptions": [],
        },
        "draft_summary": {
            "record_count": len(document.records),
            "edge_count": 0,
            "evidence_span_count": 0,
        },
    }


def validate_extracted_graph(
    document: ParsedFinancialDocument,
    plan: dict[str, Any],
    graph_build: dict[str, Any],
    example: dict[str, Any],
    extraction_metadata: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    records = document.records

    if not records:
        errors.append("no_records")

    required_fields = [field for field in plan.get("required_fields", []) if field != "company_name"]
    field_coverage = _field_coverage(records, required_fields)
    for field, ratio in field_coverage.items():
        if ratio <= 0:
            errors.append(f"missing_required_field:{field}")
        elif ratio < 0.75:
            warnings.append(f"low_field_coverage:{field}")

    evidence_spans = extraction_metadata.get("llm", {}).get("evidence_spans", [])
    evidence_coverage = _record_evidence_coverage(records, evidence_spans)
    if extraction_metadata.get("source") == "llm" and evidence_coverage <= 0:
        warnings.append("missing_llm_evidence")

    expert_id = plan.get("expert_id")
    expert_ready = not errors
    expert_checks: dict[str, Any] = {}
    if expert_id == "count":
        candidate_count = int(graph_build.get("graph_record_count") or len(example["graph"]["nodes"]))
        needs_universe = bool(plan.get("needs_candidate_universe"))
        expert_checks = {
            "candidate_count": candidate_count,
            "condition_count": len(graph_build.get("conditions", [])),
            "needs_candidate_universe": needs_universe,
        }
        if needs_universe and candidate_count < 2:
            warnings.append("count_candidate_universe_too_small")
    elif expert_id == "predict":
        target_company = graph_build.get("target_company")
        selected_year = graph_build.get("selected_year")
        target_history_years = [
            record.year
            for record in records
            if record.company_name == target_company
            and record.year is not None
            and selected_year is not None
            and record.year < selected_year
        ]
        peer_previous_year_count = _peer_previous_year_count(records, target_company, selected_year)
        expert_checks = {
            "target_company": target_company,
            "selected_year": selected_year,
            "target_history_count": len(target_history_years),
            "peer_previous_year_count": peer_previous_year_count,
        }
        if not target_history_years:
            errors.append("predict_missing_target_history")
            expert_ready = False
        if peer_previous_year_count == 0:
            warnings.append("predict_missing_peer_previous_year")

    quality_score = _quality_score(
        schema_valid=not errors,
        field_coverage=field_coverage,
        evidence_coverage=evidence_coverage,
        expert_ready=expert_ready,
    )
    return {
        "schema_valid": not errors,
        "expert_ready": expert_ready,
        "quality_score": quality_score,
        "field_coverage": field_coverage,
        "evidence_coverage": evidence_coverage,
        "errors": errors,
        "warnings": warnings,
        "expert_checks": expert_checks,
    }


def _infer_metric(query: str, default: str) -> str:
    lowered = query.casefold()
    for metric, aliases in METRIC_ALIASES.items():
        if any(alias.casefold() in lowered for alias in aliases):
            return metric
    return default


def _infer_count_fields(query: str) -> list[str]:
    lowered = query.casefold()
    fields = [
        metric
        for metric, aliases in METRIC_ALIASES.items()
        if any(alias.casefold() in lowered for alias in aliases)
    ]
    return fields or ["revenue"]


def _normalize_evidence_spans(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    spans: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("quote") or "").strip()
        line_indices = item.get("line_indices") or item.get("lines") or item.get("source_lines") or []
        if isinstance(line_indices, (int, float)):
            line_indices = [line_indices]
        normalized_lines = []
        if isinstance(line_indices, list):
            for raw_line in line_indices:
                try:
                    normalized_lines.append(int(raw_line))
                except (TypeError, ValueError):
                    continue
        spans.append(
            {
                "id": str(item.get("id") or f"evidence_{index + 1}"),
                "text": text,
                "line_indices": sorted(set(normalized_lines)),
                "confidence": _coerce_confidence(item.get("confidence")),
            }
        )
    return spans


def _normalize_edges(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    edges: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = item.get("source") or item.get("source_company")
        target = item.get("target") or item.get("target_company")
        if not source or not target:
            continue
        edges.append(
            {
                "source": str(source),
                "target": str(target),
                "relation": str(item.get("relation") or "related"),
                "year": item.get("year"),
                "confidence": _coerce_confidence(item.get("confidence")),
            }
        )
    return edges


def _coerce_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(number, 1.0))


def _field_coverage(records, fields: list[str]) -> dict[str, float]:
    if not fields:
        return {}
    denominator = max(len(records), 1)
    return {
        field: sum(getattr(record, field, None) is not None for record in records) / denominator
        for field in fields
    }


def _record_evidence_coverage(records, evidence_spans: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    if evidence_spans:
        return 1.0
    supported = 0
    for record in records:
        raw = record.raw or {}
        if any(raw.get(key) for key in ("evidence", "evidence_span_id", "source_line", "source_lines")):
            supported += 1
    return supported / len(records)


def _peer_previous_year_count(records, target_company: str | None, selected_year: int | None) -> int:
    if target_company is None or selected_year is None:
        return 0
    companies_in_year = {
        record.company_name
        for record in records
        if record.year == selected_year and record.company_name != target_company
    }
    companies_previous_year = {
        record.company_name
        for record in records
        if record.year == selected_year - 1 and record.company_name != target_company
    }
    return len(companies_in_year & companies_previous_year)


def _quality_score(
    schema_valid: bool,
    field_coverage: dict[str, float],
    evidence_coverage: float,
    expert_ready: bool,
) -> float:
    field_score = sum(field_coverage.values()) / max(len(field_coverage), 1)
    score = (
        (0.25 if schema_valid else 0.0)
        + (0.35 * field_score)
        + (0.15 * evidence_coverage)
        + (0.25 if expert_ready else 0.0)
    )
    return round(max(0.0, min(score, 1.0)), 4)
