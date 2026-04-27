from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib import error, request

from .llm_document_extractor import LLMExtractorConfig


PREDICT_HINTS = ("预测", "forecast", "predict", "projection", "project")
COUNT_HINTS = ("统计", "数量", "多少", "count", "how many", "多少家")
SUM_HINTS = ("总和", "合计", "总收入", "sum", "total", "aggregate", "总计")
METRIC_ALIASES = {
    "revenue": ("revenue", "收入", "营收", "sales", "营业收入"),
    "operating_profit": ("operating profit", "营业利润", "经营利润"),
    "net_profit": ("net profit", "净利润", "利润"),
    "employees": ("employees", "员工", "人数", "headcount"),
}
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class QueryRewrite:
    text: str
    weight: float
    source: str


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(pattern.casefold() in lowered for pattern in patterns)


def _extract_metric(query: str) -> str:
    lowered = query.casefold()
    for metric, aliases in METRIC_ALIASES.items():
        if any(alias.casefold() in lowered for alias in aliases):
            return metric
    return "revenue"


def infer_intent_expert(query: str) -> str | None:
    normalized = _normalize_whitespace(query)
    if _contains_any(normalized, PREDICT_HINTS):
        return "predict"
    if _contains_any(normalized, SUM_HINTS):
        return "sum"
    if _contains_any(normalized, COUNT_HINTS):
        return "count"
    return None


def _heuristic_predict_rewrites(query: str) -> list[QueryRewrite]:
    metric = _extract_metric(query)
    rewrites = [
        QueryRewrite(text=query, weight=1.0, source="original"),
        QueryRewrite(text=f"预测 某公司 的 {metric}", weight=0.95, source="heuristic"),
        QueryRewrite(text=f"请预测某公司在某年的 {metric}", weight=0.92, source="heuristic"),
        QueryRewrite(text=f"predict company {metric}", weight=0.88, source="heuristic"),
        QueryRewrite(text=f"forecast company {metric}", weight=0.82, source="heuristic"),
    ]
    return rewrites


def _heuristic_count_rewrites(query: str) -> list[QueryRewrite]:
    metric = _extract_metric(query)
    rewrites = [
        QueryRewrite(text=query, weight=1.0, source="original"),
        QueryRewrite(text=f"统计 公司 {metric} 条件 的 数量", weight=0.95, source="heuristic"),
        QueryRewrite(text=f"count companies with {metric} threshold", weight=0.88, source="heuristic"),
        QueryRewrite(text="统计满足数值条件的公司数量", weight=0.84, source="heuristic"),
    ]
    return rewrites


def _heuristic_sum_rewrites(query: str) -> list[QueryRewrite]:
    metric = _extract_metric(query)
    rewrites = [
        QueryRewrite(text=query, weight=1.0, source="original"),
        QueryRewrite(text=f"计算 某行业 公司 {metric} 总和", weight=0.95, source="heuristic"),
        QueryRewrite(text=f"sum company {metric} by industry", weight=0.88, source="heuristic"),
        QueryRewrite(text=f"total {metric} for companies in a year", weight=0.84, source="heuristic"),
    ]
    return rewrites


class QueryRewriter:
    def __init__(
        self,
        config: LLMExtractorConfig | None = None,
        enable_llm: bool = True,
    ) -> None:
        self.config = config or LLMExtractorConfig.from_env()
        self.enable_llm = enable_llm

    def rewrite(self, query: str) -> list[QueryRewrite]:
        heuristic = self._heuristic_rewrites(query)
        llm_rewrites: list[QueryRewrite] = []
        if self.enable_llm and self.config.available:
            try:
                llm_rewrites = self._llm_rewrites(query)
            except Exception:
                llm_rewrites = []
        return self._deduplicate([*heuristic, *llm_rewrites])

    def _heuristic_rewrites(self, query: str) -> list[QueryRewrite]:
        normalized = _normalize_whitespace(query)
        intent = infer_intent_expert(normalized)
        if intent == "predict":
            return _heuristic_predict_rewrites(normalized)
        if intent == "count":
            return _heuristic_count_rewrites(normalized)
        if intent == "sum":
            return _heuristic_sum_rewrites(normalized)
        return [
            QueryRewrite(text=normalized, weight=1.0, source="original"),
            QueryRewrite(text=f"分析 数值问题 {normalized}", weight=0.8, source="heuristic"),
        ]

    def _llm_rewrites(self, query: str) -> list[QueryRewrite]:
        payload = self._chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite retrieval queries for routing.\n"
                        "Return JSON only in the format {\"rewrites\": [\"...\", \"...\"]}.\n"
                        "Rules:\n"
                        "- Preserve the task intent.\n"
                        "- Abstract away entity names and exact numbers when helpful.\n"
                        "- Produce short retrieval-oriented rewrites.\n"
                        "- At least one rewrite should be a generic intent template.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Original query: {query}",
                },
            ]
        )
        content = self._extract_content(payload)
        parsed = self._parse_response_json(content)
        rewrites = parsed.get("rewrites", [])
        if not isinstance(rewrites, list):
            return []
        return [
            QueryRewrite(text=_normalize_whitespace(str(item)), weight=0.9, source="llm")
            for item in rewrites
            if str(item).strip()
        ][:4]

    def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": messages,
        }
        headers = {"Content-Type": "application/json"}
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
            raise RuntimeError(f"Query rewrite request failed with HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"Query rewrite request failed: {exc.reason}") from exc

    def _extract_content(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Query rewrite response does not contain choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    texts.append(str(item.get("text", "")))
            if texts:
                return "\n".join(texts)
        raise ValueError("Query rewrite response does not contain text content.")

    def _parse_response_json(self, content: str) -> dict[str, Any]:
        stripped = content.strip()
        candidates = [stripped]
        fenced = JSON_BLOCK_PATTERN.search(stripped)
        if fenced:
            candidates.insert(0, fenced.group(1))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("Query rewrite output is not valid JSON.")

    def _deduplicate(self, rewrites: list[QueryRewrite]) -> list[QueryRewrite]:
        seen: set[str] = set()
        output: list[QueryRewrite] = []
        for rewrite in rewrites:
            normalized = rewrite.text.casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(rewrite)
        return output


@lru_cache(maxsize=1)
def get_default_query_rewriter() -> QueryRewriter:
    return QueryRewriter()
