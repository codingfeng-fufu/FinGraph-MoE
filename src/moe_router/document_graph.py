from __future__ import annotations

import csv
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


NULL_TOKENS = {"", "-", "--", "na", "n/a", "none", "null", "nil"}
NUMERIC_VALUE_PATTERN = re.compile(
    r"\(?[-+]?\s*[$€£¥₹]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:k|m|b|bn|mn|cr|crore|lac|lakh|million|billion|thousand))?\)?",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"(19\d{2}|20\d{2})")
SUPPORTED_TOP_LEVEL_KEYS = ("records", "rows", "data", "items")
QUERY_ATTR_ALIASES = {
    "revenue": ("revenue", "sales", "turnover", "收入", "营收", "营业收入"),
    "operating_profit": (
        "operating profit",
        "operating_profit",
        "operating income",
        "营业利润",
        "经营利润",
    ),
    "net_profit": (
        "net profit",
        "net_profit",
        "profit after tax",
        "pat",
        "净利润",
        "净收益",
    ),
}
COUNT_QUERY_ALIASES = {
    "revenue": ("revenue", "sales", "turnover", "收入", "营收", "营业收入"),
    "operating_profit": (
        "operating profit",
        "operating_profit",
        "operating income",
        "营业利润",
        "经营利润",
    ),
    "net_profit": (
        "net profit",
        "net_profit",
        "profit after tax",
        "pat",
        "净利润",
        "净收益",
    ),
}
HEADER_ALIASES = {
    "company_name": (
        "company",
        "companyname",
        "company_name",
        "company name",
        "name",
        "entity",
        "entityname",
        "organization",
        "organisation",
        "issuer",
    ),
    "industry": (
        "industry",
        "sector",
        "industryname",
        "industry name",
        "businesssegment",
        "segment",
    ),
    "revenue": (
        "revenue",
        "sales",
        "turnover",
        "营业收入",
        "营收",
        "收入",
        "totalrevenue",
        "operatingrevenue",
    ),
    "operating_profit": (
        "operatingprofit",
        "operating profit",
        "operating_income",
        "operating income",
        "营业利润",
        "经营利润",
        "profitfromoperations",
    ),
    "net_profit": (
        "netprofit",
        "net profit",
        "profitaftertax",
        "profit after tax",
        "pat",
        "netincome",
        "净利润",
    ),
    "employees": (
        "employees",
        "employee",
        "headcount",
        "staff",
        "workforce",
        "员工数",
        "人数",
    ),
    "year": (
        "year",
        "fiscalyear",
        "fiscal year",
        "reportyear",
        "periodyear",
        "fy",
        "年份",
    ),
}


@dataclass(frozen=True)
class FinancialRecord:
    company_name: str
    industry: str | None
    revenue: float | None
    operating_profit: float | None
    net_profit: float | None
    employees: float | None
    year: int | None
    source_index: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class ParsedFinancialDocument:
    file_name: str
    media_type: str
    source_format: str
    records: list[FinancialRecord]
    detected_columns: list[str]
    preview_text: str
    parse_summary: dict[str, Any]
    graph_summary: dict[str, Any]


@dataclass(frozen=True)
class UploadedDocument:
    file_name: str
    media_type: str
    text: str
    preview_text: str
    parse_summary: dict[str, Any]
    graph_summary: dict[str, Any]
    structured_document: ParsedFinancialDocument | None
    query_examples: list[str]


@dataclass(frozen=True)
class PredictDocumentResources:
    company_vocab: dict[str, int]
    year_vocab: dict[int, int]
    history_map: dict[tuple[str, int], tuple[float, float]]


@dataclass(frozen=True)
class BuiltQueryExample:
    expert_id: str
    example: dict[str, Any]
    answer: float
    metadata: dict[str, Any]
    predict_resources: PredictDocumentResources | None = None


def _normalize_header(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", value.casefold())
    return re.sub(r"\s+", " ", cleaned).strip()


def _canonical_header(header: str) -> str | None:
    normalized = _normalize_header(header)
    for canonical, aliases in HEADER_ALIASES.items():
        if normalized in {_normalize_header(alias) for alias in aliases}:
            return canonical
    return None


def _parse_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text.casefold() in NULL_TOKENS:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()

    lowered = text.casefold()
    multiplier = 1.0
    if re.search(r"\b(crore|cr)\b", lowered):
        multiplier = 1e7
    elif re.search(r"\b(lakh|lac)\b", lowered):
        multiplier = 1e5
    elif re.search(r"\b(billion|bn|b)\b", lowered):
        multiplier = 1e9
    elif re.search(r"\b(million|mn|m)\b", lowered):
        multiplier = 1e6
    elif re.search(r"\b(thousand|k)\b", lowered):
        multiplier = 1e3

    cleaned = re.sub(r"[$€£¥₹,\s]", "", text)
    cleaned = re.sub(
        r"(?i)(crore|cr|lakh|lac|billion|bn|million|mn|thousand|k|m|b)",
        "",
        cleaned,
    )
    cleaned = re.sub(r"[^0-9eE+\-\.]", "", cleaned)
    if not cleaned:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if negative:
        number *= -1.0
    return number * multiplier


def _parse_year(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1900 <= value <= 2100 else None
    if isinstance(value, float):
        integer = int(value)
        return integer if 1900 <= integer <= 2100 else None

    match = YEAR_PATTERN.search(str(value))
    if match is None:
        return None
    return int(match.group(1))


def _decode_document_content(file_name: str, media_type: str, content: bytes) -> tuple[str, str]:
    suffix = Path(file_name).suffix.casefold()
    if suffix == ".pdf" or media_type == "application/pdf":
        if PdfReader is None:
            raise ValueError(
                "PDF parsing is not available in this environment. Upload CSV, TSV, JSON, JSONL, TXT, or a markdown table."
            )
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError("The uploaded PDF does not contain extractable text.")
        return text, "pdf"

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding), "text"
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode the uploaded document as text.")


def _extract_rows_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        for key in SUPPORTED_TOP_LEVEL_KEYS:
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return [dict(item) for item in value]
    raise ValueError("JSON document must be an array of objects or contain a top-level records/data/rows/items list.")


def _parse_json_rows(text: str) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    try:
        return _extract_rows_from_json(payload)
    except ValueError:
        return None


def _parse_jsonl_rows(text: str) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("{"):
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        rows.append(payload)
    return rows or None


def _parse_markdown_table_rows(text: str) -> list[dict[str, str]] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.count("|") >= 2]
    if len(table_lines) < 2:
        return None

    header_line = table_lines[0]
    separator_line = table_lines[1]
    if not re.fullmatch(r"\|?[\s:\-|\t]+\|?", separator_line):
        return None

    headers = [cell.strip() for cell in header_line.strip("|").split("|")]
    if not headers:
        return None

    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows or None


def _parse_delimited_rows(text: str) -> list[dict[str, str]] | None:
    sample = "\n".join(text.splitlines()[:8]).strip()
    if not sample:
        return None
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.get_dialect("excel")

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        return None
    rows = [dict(row) for row in reader if row]
    return rows or None


def _raw_rows_from_text(file_name: str, text: str) -> tuple[list[dict[str, Any]], str]:
    suffix = Path(file_name).suffix.casefold()
    if suffix == ".json":
        try:
            rows = _extract_rows_from_json(json.loads(text))
        except (json.JSONDecodeError, ValueError):
            raise ValueError("The uploaded JSON document is not a structured row collection.")
        return rows, "json"
    if suffix == ".jsonl":
        rows = _parse_jsonl_rows(text)
        if rows is None:
            raise ValueError("The uploaded JSONL document is not a collection of JSON objects.")
        return rows, "jsonl"

    json_rows = _parse_json_rows(text)
    if json_rows is not None:
        return json_rows, "json"

    jsonl_rows = _parse_jsonl_rows(text)
    if jsonl_rows is not None:
        return jsonl_rows, "jsonl"

    markdown_rows = _parse_markdown_table_rows(text)
    if markdown_rows is not None:
        return markdown_rows, "markdown_table"

    delimited_rows = _parse_delimited_rows(text)
    if delimited_rows is not None:
        return delimited_rows, "delimited_text"

    raise ValueError(
        "Could not detect a structured table in the uploaded document. Supported formats: CSV, TSV, JSON, JSONL, TXT, markdown table, and text-based PDF."
    )


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in NULL_TOKENS:
        return None
    return text


def _build_preview_text(records: list[FinancialRecord]) -> str:
    if not records:
        return "No structured records were parsed."

    preview_lines = []
    for record in records[:4]:
        parts = [record.company_name]
        if record.industry:
            parts.append(f"industry={record.industry}")
        if record.year is not None:
            parts.append(f"year={record.year}")
        if record.revenue is not None:
            parts.append(f"revenue={record.revenue:.2f}")
        if record.operating_profit is not None:
            parts.append(f"operating_profit={record.operating_profit:.2f}")
        if record.net_profit is not None:
            parts.append(f"net_profit={record.net_profit:.2f}")
        if record.employees is not None:
            parts.append(f"employees={record.employees:.0f}")
        preview_lines.append(" | ".join(parts))
    return "\n".join(preview_lines)


def parse_financial_document(
    file_name: str,
    media_type: str,
    content: bytes,
) -> ParsedFinancialDocument:
    text, base_format = _decode_document_content(file_name, media_type, content)
    raw_rows, structured_format = _raw_rows_from_text(file_name, text)
    source_format = structured_format if base_format == "text" else f"{base_format}:{structured_format}"

    detected_columns: list[str] = []
    header_map: dict[str, str] = {}
    records: list[FinancialRecord] = []
    dropped_missing_company = 0
    dropped_missing_numeric = 0

    if raw_rows:
        detected_columns = list(raw_rows[0].keys())
        header_map = {
            header: canonical
            for header in detected_columns
            if (canonical := _canonical_header(header)) is not None
        }

    for index, row in enumerate(raw_rows):
        company_value = next(
            (
                row[header]
                for header, canonical in header_map.items()
                if canonical == "company_name"
            ),
            None,
        )
        company_name = _coerce_text(company_value)
        if company_name is None:
            dropped_missing_company += 1
            continue

        industry = _coerce_text(
            next((row[header] for header, canonical in header_map.items() if canonical == "industry"), None)
        )
        revenue = _parse_number(
            next((row[header] for header, canonical in header_map.items() if canonical == "revenue"), None)
        )
        operating_profit = _parse_number(
            next(
                (
                    row[header]
                    for header, canonical in header_map.items()
                    if canonical == "operating_profit"
                ),
                None,
            )
        )
        net_profit = _parse_number(
            next((row[header] for header, canonical in header_map.items() if canonical == "net_profit"), None)
        )
        employees = _parse_number(
            next((row[header] for header, canonical in header_map.items() if canonical == "employees"), None)
        )
        year = _parse_year(
            next((row[header] for header, canonical in header_map.items() if canonical == "year"), None)
        )

        if all(value is None for value in (revenue, operating_profit, net_profit, employees)):
            dropped_missing_numeric += 1
            continue

        records.append(
            FinancialRecord(
                company_name=company_name,
                industry=industry,
                revenue=revenue,
                operating_profit=operating_profit,
                net_profit=net_profit,
                employees=employees,
                year=year,
                source_index=index,
                raw=dict(row),
            )
        )

    if not records:
        raise ValueError(
            "No usable company records were found. Expected columns such as company, revenue, operating profit, net profit, employees, industry, or year."
        )

    industries = sorted({record.industry for record in records if record.industry})
    years = sorted({record.year for record in records if record.year is not None})
    graph_summary = {
        "record_count": len(records),
        "unique_companies": len({_normalize_text(record.company_name) for record in records}),
        "industry_count": len(industries),
        "year_count": len(years),
        "records_with_sum_fields": sum(record.revenue is not None for record in records),
        "records_with_count_fields": sum(
            record.revenue is not None or record.operating_profit is not None or record.net_profit is not None
            for record in records
        ),
        "records_with_predict_fields": sum(
            record.year is not None and record.revenue is not None and record.employees is not None
            for record in records
        ),
        "estimated_edges": _estimate_same_attribute_edges(records),
    }
    parse_summary = {
        "source_format": source_format,
        "raw_row_count": len(raw_rows),
        "parsed_record_count": len(records),
        "dropped_missing_company": dropped_missing_company,
        "dropped_missing_numeric": dropped_missing_numeric,
        "detected_columns": detected_columns,
        "mapped_fields": sorted(set(header_map.values())),
        "industries": industries[:10],
        "years": years,
    }

    return ParsedFinancialDocument(
        file_name=file_name,
        media_type=media_type,
        source_format=source_format,
        records=records,
        detected_columns=detected_columns,
        preview_text=_build_preview_text(records),
        parse_summary=parse_summary,
        graph_summary=graph_summary,
    )


def _estimate_same_attribute_edges(records: list[FinancialRecord]) -> int:
    by_industry: dict[str, int] = defaultdict(int)
    no_industry = 0
    for record in records:
        if record.industry:
            by_industry[record.industry] += 1
        else:
            no_industry += 1

    total = 0
    for size in by_industry.values():
        total += size * (size - 1)
    if no_industry > 1:
        total += no_industry * (no_industry - 1)
    return total


def _merge_records(records: list[FinancialRecord]) -> list[FinancialRecord]:
    grouped: dict[tuple[str, int | None], list[FinancialRecord]] = defaultdict(list)
    for record in records:
        grouped[(_normalize_text(record.company_name), record.year)].append(record)

    merged: list[FinancialRecord] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: item.source_index)
        latest = ordered[-1]

        def pick_text(field: str) -> str | None:
            for item in reversed(ordered):
                value = getattr(item, field)
                if value:
                    return value
            return None

        def pick_number(field: str) -> float | None:
            for item in reversed(ordered):
                value = getattr(item, field)
                if value is not None:
                    return value
            return None

        merged.append(
            FinancialRecord(
                company_name=pick_text("company_name") or latest.company_name,
                industry=pick_text("industry"),
                revenue=pick_number("revenue"),
                operating_profit=pick_number("operating_profit"),
                net_profit=pick_number("net_profit"),
                employees=pick_number("employees"),
                year=latest.year,
                source_index=latest.source_index,
                raw=latest.raw,
            )
        )
    return sorted(merged, key=lambda item: (item.year or 0, item.company_name.casefold()))


def _extract_query_year(query: str) -> int | None:
    match = YEAR_PATTERN.search(query)
    if match is None:
        return None
    return int(match.group(1))


def _match_best_candidate(query: str, candidates: list[str]) -> str | None:
    normalized_query = _normalize_text(query)
    matches: list[tuple[int, str]] = []
    for candidate in candidates:
        normalized_candidate = _normalize_text(candidate)
        if normalized_candidate and normalized_candidate in normalized_query:
            matches.append((len(normalized_candidate), candidate))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def _select_year(records: list[FinancialRecord], query: str) -> tuple[int | None, bool]:
    available_years = sorted({record.year for record in records if record.year is not None})
    query_year = _extract_query_year(query)
    if query_year is not None:
        return query_year, False
    if len(available_years) > 1:
        return available_years[-1], True
    if available_years:
        return available_years[-1], False
    return None, False


def _parse_sum_attribute(query: str) -> str:
    normalized_query = _normalize_text(query)
    for attribute, aliases in QUERY_ATTR_ALIASES.items():
        for alias in aliases:
            if _normalize_text(alias) in normalized_query:
                return attribute
    return "revenue"


def _normalize_count_query(query: str) -> str:
    normalized = query
    replacements = (
        (r"greater than or equal to|at least|不少于|不低于|大于等于", ">="),
        (r"less than or equal to|at most|no more than|不超过|至多|小于等于", "<="),
        (r"greater than|more than|above|大于|高于", ">"),
        (r"less than|below|under|小于|低于", "<"),
        (r"equal to|equals|等于", "=="),
    )
    for pattern, token in replacements:
        normalized = re.sub(pattern, f" {token} ", normalized, flags=re.IGNORECASE)
    return normalized


def _parse_count_conditions(query: str) -> list[dict[str, Any]]:
    normalized = _normalize_count_query(query)
    conditions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()

    for attribute, aliases in COUNT_QUERY_ALIASES.items():
        alias_pattern = "|".join(
            sorted((re.escape(alias) for alias in aliases), key=len, reverse=True)
        )
        pattern = re.compile(
            rf"(?:{alias_pattern})\s*(==|>=|<=|>|<)\s*({NUMERIC_VALUE_PATTERN.pattern})",
            re.IGNORECASE,
        )
        for match in pattern.finditer(normalized):
            value = _parse_number(match.group(2))
            if value is None:
                continue
            condition = (attribute, match.group(1), float(value))
            if condition in seen:
                continue
            seen.add(condition)
            conditions.append(
                {
                    "attribute": attribute,
                    "op": match.group(1),
                    "value": float(value),
                }
            )

    conditions.sort(key=lambda item: query.find(item["attribute"].replace("_", " ")))
    return conditions[:2]


def _apply_common_filters(records: list[FinancialRecord], query: str) -> tuple[list[FinancialRecord], dict[str, Any]]:
    merged = _merge_records(records)
    matched_industry = _match_best_candidate(
        query,
        sorted({record.industry for record in merged if record.industry}),
    )
    query_year = _extract_query_year(query)
    selected_year, default_year_applied = _select_year(merged, query)

    filtered = merged
    if selected_year is not None:
        year_filtered = [record for record in filtered if record.year == selected_year]
        if query_year is not None and not year_filtered:
            raise ValueError(f"The uploaded document does not contain rows for year {query_year}.")
        if year_filtered:
            filtered = year_filtered
    if matched_industry is not None:
        industry_filtered = [record for record in filtered if record.industry == matched_industry]
        if industry_filtered:
            filtered = industry_filtered

    metadata = {
        "matched_industry": matched_industry,
        "selected_year": selected_year,
        "default_year_applied": default_year_applied,
        "input_record_count": len(records),
        "filtered_record_count": len(filtered),
    }
    return filtered, metadata


def _build_company_graph(
    graph_id: str,
    records: list[FinancialRecord],
) -> dict[str, Any]:
    nodes = []
    edges = []
    for index, record in enumerate(records):
        nodes.append(
            {
                "id": f"n{index}",
                "type": "company",
                "name": record.company_name,
                "attributes": {
                    "industry": record.industry or "UNKNOWN",
                    "revenue": float(record.revenue or 0.0),
                    "operating_profit": float(record.operating_profit or 0.0),
                    "net_profit": float(record.net_profit or 0.0),
                    **({"employees": float(record.employees)} if record.employees is not None else {}),
                },
            }
        )

    all_missing_industry = all(record.industry is None for record in records)
    for src_index in range(len(records)):
        for dst_index in range(src_index + 1, len(records)):
            same_industry = records[src_index].industry and records[src_index].industry == records[dst_index].industry
            if same_industry or all_missing_industry:
                edges.append(
                    {
                        "source": f"n{src_index}",
                        "target": f"n{dst_index}",
                        "relation": "same_industry" if same_industry else "same_document",
                    }
                )

    return {
        "graph_id": graph_id,
        "nodes": nodes,
        "edges": edges,
    }


def _build_same_year_graph(
    graph_id: str,
    records: list[FinancialRecord],
) -> dict[str, Any]:
    graph = _build_company_graph(graph_id, records)
    edges = []
    for src_index in range(len(records)):
        for dst_index in range(src_index + 1, len(records)):
            edges.append(
                {
                    "source": f"n{src_index}",
                    "target": f"n{dst_index}",
                    "relation": "same_year",
                }
            )
    graph["edges"] = edges
    return graph


def _condition_matches(record: FinancialRecord, condition: dict[str, Any]) -> bool:
    value = getattr(record, condition["attribute"])
    if value is None:
        return False
    target = float(condition["value"])
    op = condition["op"]
    if op == "==":
        return value == target
    if op == "<":
        return value < target
    if op == "<=":
        return value <= target
    if op == ">":
        return value > target
    if op == ">=":
        return value >= target
    raise ValueError(f"Unsupported condition operator: {op}")


def _build_sum_example(
    document: ParsedFinancialDocument,
    query: str,
    session_id: str,
) -> BuiltQueryExample:
    filtered_records, metadata = _apply_common_filters(document.records, query)
    target_attr = _parse_sum_attribute(query)
    usable_records = [
        record for record in filtered_records if getattr(record, target_attr) is not None
    ]
    if not usable_records:
        raise ValueError(
            f"The uploaded document does not contain usable `{target_attr}` values for the requested SUM query."
        )

    answer = float(sum(getattr(record, target_attr) or 0.0 for record in usable_records))
    example = {
        "sample_id": f"{session_id}_sum",
        "split": "inference",
        "graph": _build_company_graph(f"{session_id}_sum_graph", usable_records),
        "task": {
            "kind": "sum",
            "query_text": query,
            "target": {
                "node_type": "company",
                "node_id": None,
                "node_name": None,
                "attribute": target_attr,
            },
        },
        "answer": answer,
    }
    metadata.update(
        {
            "target_attr": target_attr,
            "ground_truth_available": True,
            "graph_record_count": len(usable_records),
        }
    )
    return BuiltQueryExample("sum", example, answer, metadata)


def _build_count_example(
    document: ParsedFinancialDocument,
    query: str,
    session_id: str,
) -> BuiltQueryExample:
    filtered_records, metadata = _apply_common_filters(document.records, query)
    conditions = _parse_count_conditions(query)
    if not conditions:
        raise ValueError(
            "Could not parse numeric count conditions from the query. Use patterns like `revenue >= 1000` or `net profit < 50`."
        )

    usable_records = [
        record
        for record in filtered_records
        if all(getattr(record, condition["attribute"]) is not None for condition in conditions)
    ]
    if not usable_records:
        raise ValueError("The uploaded document does not contain rows with the numeric fields required by the COUNT query.")

    answer = float(
        sum(
            1
            for record in usable_records
            if all(_condition_matches(record, condition) for condition in conditions)
        )
    )

    target: dict[str, Any] = {
        "node_type": "company",
    }
    if len(conditions) == 1:
        target["attribute"] = conditions[0]["attribute"]
        target["condition"] = {
            "op": conditions[0]["op"],
            "value": conditions[0]["value"],
        }
    else:
        target["conditions"] = conditions

    example = {
        "sample_id": f"{session_id}_count",
        "split": "inference",
        "graph": _build_company_graph(f"{session_id}_count_graph", usable_records),
        "task": {
            "kind": "count",
            "query_text": query,
            "target": target,
        },
        "answer": answer,
    }
    metadata.update(
        {
            "conditions": conditions,
            "ground_truth_available": True,
            "graph_record_count": len(usable_records),
        }
    )
    return BuiltQueryExample("count", example, answer, metadata)


def _build_predict_resources(records: list[FinancialRecord]) -> PredictDocumentResources:
    predict_records = [
        record
        for record in _merge_records(records)
        if record.year is not None and record.revenue is not None and record.employees is not None
    ]
    company_vocab = {
        company_name: index
        for index, company_name in enumerate(sorted({record.company_name for record in predict_records}))
    }
    year_vocab = {
        year: index
        for index, year in enumerate(sorted({record.year for record in predict_records if record.year is not None}))
    }
    history_map = {
        (record.company_name, int(record.year)): (float(record.revenue), float(record.employees))
        for record in predict_records
        if record.year is not None and record.revenue is not None and record.employees is not None
    }
    return PredictDocumentResources(
        company_vocab=company_vocab,
        year_vocab=year_vocab,
        history_map=history_map,
    )


def _build_predict_example(
    document: ParsedFinancialDocument,
    query: str,
    session_id: str,
) -> BuiltQueryExample:
    merged = _merge_records(document.records)
    predict_records = [
        record
        for record in merged
        if record.year is not None and record.revenue is not None and record.employees is not None
    ]
    if not predict_records:
        raise ValueError(
            "The uploaded document does not contain multi-year company rows with both revenue and employees, which are required by the predict expert."
        )

    company_name = _match_best_candidate(
        query,
        sorted({record.company_name for record in predict_records}),
    )
    if company_name is None:
        raise ValueError("Could not determine which company to predict from the query.")

    query_year = _extract_query_year(query)
    company_years = sorted(
        {
            record.year
            for record in predict_records
            if record.company_name == company_name and record.year is not None
        }
    )
    if not company_years:
        raise ValueError(f"The uploaded document does not contain yearly history for `{company_name}`.")
    selected_year = query_year or company_years[-1]

    year_records = [record for record in predict_records if record.year == selected_year]
    if not year_records:
        raise ValueError(f"The uploaded document does not contain a full company graph for year {selected_year}.")

    target_record = next(
        (
            record
            for record in year_records
            if _normalize_text(record.company_name) == _normalize_text(company_name)
        ),
        None,
    )
    if target_record is None or target_record.revenue is None:
        raise ValueError(
            f"The uploaded document does not contain the target company `{company_name}` for year {selected_year}."
        )

    resources = _build_predict_resources(predict_records)
    if company_name not in resources.company_vocab or selected_year not in resources.year_vocab:
        raise ValueError("The uploaded document does not contain enough history to build predict expert resources.")

    graph = _build_same_year_graph(f"{session_id}_predict_graph_{selected_year}", year_records)
    target_node_id = next(
        node["id"]
        for node in graph["nodes"]
        if _normalize_text(node["name"]) == _normalize_text(company_name)
    )
    example = {
        "sample_id": f"{session_id}_predict",
        "split": "inference",
        "graph": graph,
        "task": {
            "kind": "predict",
            "query_text": query,
        },
        "target": {
            "node_type": "company",
            "node_id": target_node_id,
            "node_name": company_name,
            "attribute": "revenue",
        },
        "answer": float(target_record.revenue),
    }

    metadata = {
        "selected_year": selected_year,
        "default_year_applied": query_year is None and len(company_years) > 1,
        "target_company": company_name,
        "history_years": company_years,
        "ground_truth_available": True,
        "graph_record_count": len(year_records),
    }
    return BuiltQueryExample("predict", example, float(target_record.revenue), metadata, resources)


def build_query_example(
    document: ParsedFinancialDocument,
    query: str,
    expert_id: str,
    session_id: str,
) -> BuiltQueryExample:
    if expert_id == "sum":
        return _build_sum_example(document, query, session_id)
    if expert_id == "count":
        return _build_count_example(document, query, session_id)
    if expert_id == "predict":
        return _build_predict_example(document, query, session_id)
    raise ValueError(f"Unsupported expert id: {expert_id}")


def build_query_suggestions(document: ParsedFinancialDocument) -> list[str]:
    records = _merge_records(document.records)
    suggestions: list[str] = []

    industries = sorted({record.industry for record in records if record.industry and record.revenue is not None})
    if industries:
        suggestions.append(f"{industries[0]}行业的收入总和是多少")

    numeric_records = [record for record in records if record.revenue is not None]
    if numeric_records:
        threshold = numeric_records[min(1, len(numeric_records) - 1)].revenue
        suggestions.append(f"请统计收入不低于 {threshold:.1f} 的公司数量")

    predict_ready = [
        record
        for record in records
        if record.year is not None and record.revenue is not None and record.employees is not None
    ]
    if predict_ready:
        target = sorted(
            predict_ready,
            key=lambda item: (item.year or 0, item.company_name.casefold()),
            reverse=True,
        )[0]
        suggestions.append(f"请预测 {target.company_name} 在 {target.year} 年的收入")

    if not suggestions:
        suggestions = [
            "请统计收入不低于 1000 的公司数量",
            "请计算收入总和",
        ]
    return suggestions[:3]


def ingest_document(
    file_name: str,
    media_type: str,
    content: bytes,
) -> UploadedDocument:
    text, base_format = _decode_document_content(file_name, media_type, content)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    preview_text = text[:800].strip() if text.strip() else f"Uploaded file: {file_name}"
    numeric_matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    year_matches = YEAR_PATTERN.findall(text)

    structured_document: ParsedFinancialDocument | None = None
    structured_error: str | None = None
    try:
        structured_document = parse_financial_document(file_name, media_type, content)
        preview_text = structured_document.preview_text
    except ValueError as exc:
        structured_error = str(exc)

    if structured_document is not None:
        parse_summary = {
            **structured_document.parse_summary,
            "document_format": base_format,
            "text_char_count": len(text),
            "line_count": len(lines),
            "llm_extraction_ready": True,
            "structured_parse_available": True,
        }
        graph_summary = {
            **structured_document.graph_summary,
            "llm_extraction_ready": True,
        }
        query_examples = build_query_suggestions(structured_document)
    else:
        parse_summary = {
            "document_format": base_format,
            "text_char_count": len(text),
            "line_count": len(lines),
            "numeric_fact_candidates": len(numeric_matches),
            "year_candidates": len(year_matches),
            "structured_parse_available": False,
            "structured_parse_error": structured_error,
            "llm_extraction_ready": True,
        }
        graph_summary = {
            "record_count": 0,
            "estimated_nodes": max(1, min(len(numeric_matches) + 1, 64)),
            "estimated_edges": max(0, min(len(numeric_matches) * 2, 256)),
            "llm_extraction_ready": True,
        }
        query_examples = [
            "收入总和是多少",
            "请统计 revenue >= 1000 的公司数量",
            "预测某公司的 revenue",
        ]

    return UploadedDocument(
        file_name=file_name,
        media_type=media_type or "application/octet-stream",
        text=text,
        preview_text=preview_text,
        parse_summary=parse_summary,
        graph_summary=graph_summary,
        structured_document=structured_document,
        query_examples=query_examples[:3],
    )


def build_parsed_document_from_records(
    file_name: str,
    media_type: str,
    records: list[FinancialRecord],
    source_format: str = "llm",
) -> ParsedFinancialDocument:
    if not records:
        raise ValueError("At least one financial record is required.")

    industries = sorted({record.industry for record in records if record.industry})
    years = sorted({record.year for record in records if record.year is not None})
    detected_columns = [
        "company",
        "industry",
        "year",
        "revenue",
        "operating_profit",
        "net_profit",
        "employees",
    ]
    parse_summary = {
        "source_format": source_format,
        "raw_row_count": len(records),
        "parsed_record_count": len(records),
        "dropped_missing_company": 0,
        "dropped_missing_numeric": 0,
        "detected_columns": detected_columns,
        "mapped_fields": [
            "company_name",
            "industry",
            "year",
            "revenue",
            "operating_profit",
            "net_profit",
            "employees",
        ],
        "industries": industries[:10],
        "years": years,
        "llm_extraction_ready": True,
    }
    graph_summary = {
        "record_count": len(records),
        "unique_companies": len({_normalize_text(record.company_name) for record in records}),
        "industry_count": len(industries),
        "year_count": len(years),
        "records_with_sum_fields": sum(record.revenue is not None for record in records),
        "records_with_count_fields": sum(
            record.revenue is not None or record.operating_profit is not None or record.net_profit is not None
            for record in records
        ),
        "records_with_predict_fields": sum(
            record.year is not None and record.revenue is not None and record.employees is not None
            for record in records
        ),
        "estimated_edges": _estimate_same_attribute_edges(records),
        "llm_extraction_ready": True,
    }
    return ParsedFinancialDocument(
        file_name=file_name,
        media_type=media_type or "application/octet-stream",
        source_format=source_format,
        records=records,
        detected_columns=detected_columns,
        preview_text=_build_preview_text(records),
        parse_summary=parse_summary,
        graph_summary=graph_summary,
    )
