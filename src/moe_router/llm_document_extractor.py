from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from .document_graph import (
    FinancialRecord,
    UploadedDocument,
    build_parsed_document_from_records,
)
from .graph_extraction import (
    GraphExtractionResult,
    build_extraction_plan,
    normalize_llm_extraction_metadata,
)


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class LLMExtractorConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 45.0

    @classmethod
    def from_env(cls) -> "LLMExtractorConfig":
        env_file_values = _load_env_file(Path.cwd() / ".env")
        get_value = lambda *keys: _first_non_empty(os.getenv(key) or env_file_values.get(key) for key in keys)
        api_key = (
            get_value("LLM_GRAPH_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY")
            or None
        )
        explicit_base_url = get_value("LLM_GRAPH_BASE_URL", "OPENAI_BASE_URL")
        if explicit_base_url:
            base_url = explicit_base_url.rstrip("/")
        elif get_value("DASHSCOPE_API_KEY"):
            base_url = DEFAULT_DASHSCOPE_BASE_URL
        elif api_key:
            base_url = DEFAULT_OPENAI_BASE_URL
        else:
            base_url = ""

        model = (
            get_value("LLM_GRAPH_MODEL", "OPENAI_MODEL")
            or ("qwen-plus" if get_value("DASHSCOPE_API_KEY") else "")
        ).strip()
        timeout_raw = get_value("LLM_GRAPH_TIMEOUT_SECONDS") or "45"
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError:
            timeout_seconds = 45.0
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

    @property
    def available(self) -> bool:
        if not self.base_url or not self.model:
            return False
        if self.base_url == DEFAULT_OPENAI_BASE_URL and not self.api_key:
            return False
        return True


class LLMDocumentExtractor:
    def __init__(self, config: LLMExtractorConfig | None = None) -> None:
        self.config = config or LLMExtractorConfig.from_env()

    @property
    def available(self) -> bool:
        return self.config.available

    def extract(
        self,
        uploaded_document: UploadedDocument,
        query: str,
        expert_id: str,
    ):
        return self.extract_with_metadata(uploaded_document, query, expert_id).document

    def extract_with_metadata(
        self,
        uploaded_document: UploadedDocument,
        query: str,
        expert_id: str,
    ) -> GraphExtractionResult:
        if not self.available:
            raise RuntimeError("LLM document extraction is not configured.")

        plan = build_extraction_plan(query, expert_id)
        payload = self._chat_completion(
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": self._user_prompt(
                        uploaded_document.text,
                        query,
                        expert_id,
                        plan,
                    ),
                },
            ]
        )
        content = self._extract_content(payload)
        parsed = self._parse_response_json(content)
        records = self._records_from_payload(parsed)
        if not records:
            raise ValueError("The LLM did not return any usable financial records.")
        document = build_parsed_document_from_records(
            uploaded_document.file_name,
            uploaded_document.media_type,
            records,
            source_format="llm_query_conditioned",
        )
        return GraphExtractionResult(
            document=document,
            metadata=normalize_llm_extraction_metadata(parsed, plan, document),
        )

    def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": messages,
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network path
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM extraction request failed with HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"LLM extraction request failed: {exc.reason}") from exc

    def _system_prompt(self) -> str:
        return (
            "You extract a query-conditioned company financial graph from a small document.\n"
            "Return JSON only.\n"
            "The JSON schema is:\n"
            "{\n"
            '  "extraction_scope": string,\n'
            '  "records": [\n'
            "    {\n"
            '      "company_name": string,\n'
            '      "industry": string or null,\n'
            '      "year": integer or null,\n'
            '      "revenue": number or null,\n'
            '      "operating_profit": number or null,\n'
            '      "net_profit": number or null,\n'
            '      "employees": number or null,\n'
            '      "source_lines": [integer] or [],\n'
            '      "evidence": string or null\n'
            "    }\n"
            "  ],\n"
            '  "edges": [{"source": string, "target": string, "relation": string, "year": integer or null}],\n'
            '  "evidence_spans": [{"id": string, "line_indices": [integer], "text": string, "confidence": number}],\n'
            '  "units": {"revenue": string, "operating_profit": string, "net_profit": string, "employees": string},\n'
            '  "confidence": number,\n'
            '  "assumptions": [string]\n'
            "}\n"
            "Rules:\n"
            "- Only include facts directly supported by the document.\n"
            "- Emit one row per company-year when year-specific facts are present.\n"
            "- If a field is missing, return null.\n"
            "- Keep company names and industry labels exactly as written when possible.\n"
            "- Extract the smallest sufficient set of records needed to build the graph for the current query.\n"
            "- Preserve enough peer rows for graph context, not just the final answer row.\n"
            "- For predict queries, include enough company-year rows to preserve the target company's history.\n"
            "- For count queries, include the candidate universe, not only records that satisfy the condition.\n"
            "- Add source_lines and evidence whenever the document has line-like or table-like evidence.\n"
            "- Use 1-based line numbers for evidence_spans.line_indices.\n"
            "- Never return explanations, markdown, or comments."
        )

    def _user_prompt(
        self,
        document_text: str,
        query: str,
        expert_id: str,
        plan: dict[str, Any],
    ) -> str:
        expert_guidance = {
            "sum": (
                "Extract company rows relevant to the aggregation target. "
                "If the query names an industry or year, keep rows from that slice. "
                "Preserve peer companies that contribute to the total. "
                "Revenue, operating_profit, and net_profit matter most."
            ),
            "count": (
                "Extract company rows needed for numeric filtering. "
                "If the query mentions thresholds or comparisons, keep rows whose fields are needed to evaluate them. "
                "If industry or year is mentioned, preserve that slice. "
                "Revenue, operating_profit, and net_profit matter most."
            ),
            "predict": (
                "Extract rows for the target company's history and peer companies in the target year. "
                "Also include the immediately preceding year for those peer companies whenever present, "
                "because the predict expert estimates revenue growth from peer growth context. "
                "Year, revenue, and employees are critical. "
                "Include all available earlier rows for the target company when present."
            ),
        }.get(expert_id, "Extract company-level financial rows.")
        return (
            f"Expert type: {expert_id}\n"
            f"User query: {query}\n"
            f"Extraction plan: {json.dumps(plan, ensure_ascii=False)}\n"
            f"Extraction guidance: {expert_guidance}\n\n"
            "Document lines are 1-based. Preserve source line numbers when possible.\n\n"
            "Document:\n"
            f"{document_text}"
        )

    def _extract_content(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM response does not contain choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text", "")))
            if parts:
                return "\n".join(parts)
        raise ValueError("LLM response does not contain text content.")

    def _parse_response_json(self, content: str) -> dict[str, Any]:
        stripped = content.strip()
        candidates = [stripped]
        fenced = JSON_BLOCK_PATTERN.search(stripped)
        if fenced:
            candidates.insert(0, fenced.group(1))
        brace_candidate = self._slice_first_json_object(stripped)
        if brace_candidate:
            candidates.insert(0, brace_candidate)

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("LLM response is not valid JSON.")

    def _slice_first_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return None

    def _records_from_payload(self, payload: dict[str, Any]) -> list[FinancialRecord]:
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise ValueError("LLM JSON payload must contain a `records` list.")

        records: list[FinancialRecord] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            company_name = self._coerce_text(row.get("company_name"))
            if company_name is None:
                continue
            revenue = self._coerce_number(row.get("revenue"))
            operating_profit = self._coerce_number(row.get("operating_profit"))
            net_profit = self._coerce_number(row.get("net_profit"))
            employees = self._coerce_number(row.get("employees"))
            if all(value is None for value in (revenue, operating_profit, net_profit, employees)):
                continue
            records.append(
                FinancialRecord(
                    company_name=company_name,
                    industry=self._coerce_text(row.get("industry")),
                    year=self._coerce_year(row.get("year")),
                    revenue=revenue,
                    operating_profit=operating_profit,
                    net_profit=net_profit,
                    employees=employees,
                    source_index=index,
                    raw=dict(row),
                )
            )
        return records

    def _coerce_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _coerce_number(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _coerce_year(self, value: Any) -> int | None:
        number = self._coerce_number(value)
        if number is None:
            return None
        year = int(number)
        return year if 1900 <= year <= 2100 else None


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _first_non_empty(values) -> str | None:
    for value in values:
        if value:
            return value
    return None
