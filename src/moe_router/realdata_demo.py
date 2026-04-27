from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
import uuid
from typing import Any

import torch
from torch_geometric.data import Batch, Data

from realdata_experts.common import signed_log1p
from realdata_experts.count_v2.data import count_example_to_data
from realdata_experts.count_v2.model import CountV2Model
from realdata_experts.predict_v5.data import (
    PredictV5Resources,
    predict_example_to_sample as predict_v5_example_to_sample,
)
from realdata_experts.predict_v5.model import PredictV5GRUModel
from realdata_experts.predict_v6.data import (
    PredictV6Resources,
    predict_example_to_sample as predict_v6_example_to_sample,
)
from realdata_experts.predict_tabular.artifact import (
    load_artifact as load_predict_tabular_artifact,
    predict_growth as predict_tabular_growth,
)
from realdata_experts.sum_v2.data import sum_example_to_data
from realdata_experts.sum_v2.model import SumV2Model

from .document_graph import (
    ParsedFinancialDocument,
    build_query_example,
    ingest_document,
    UploadedDocument,
)
from .graph_extraction import (
    build_structured_extraction_metadata,
    validate_extracted_graph,
)
from .llm_document_extractor import LLMDocumentExtractor
from .paths import REPO_ROOT
from .router import MoERouter


@dataclass(frozen=True)
class RealdataArtifact:
    expert_id: str
    checkpoint_path: str
    description: str


@dataclass
class DocumentSession:
    session_id: str
    file_name: str
    media_type: str
    size_bytes: int
    document_text: str
    preview_text: str
    parse_summary: dict[str, Any]
    graph_summary: dict[str, Any]
    query_examples: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "file_name": self.file_name,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "document_text": self.document_text,
            "preview_text": self.preview_text,
            "parse_summary": self.parse_summary,
            "graph_summary": self.graph_summary,
            "query_examples": self.query_examples,
        }


ARTIFACTS: dict[str, RealdataArtifact] = {
    "sum": RealdataArtifact(
        expert_id="sum",
        checkpoint_path=str(REPO_ROOT / "runs" / "realdata_experts" / "sum_v2" / "best_model.pt"),
        description="SUM v2 expert on sum_dataset2.jsonl",
    ),
    "count": RealdataArtifact(
        expert_id="count",
        checkpoint_path=str(REPO_ROOT / "runs" / "realdata_experts" / "count_v2" / "best_model.pt"),
        description="COUNT v2 expert on count_600.jsonl",
    ),
    "predict": RealdataArtifact(
        expert_id="predict",
        checkpoint_path=str(
            REPO_ROOT
            / "runs"
            / "realdata_experts"
            / "predict_v5_lstm_h32_residual_015_e120"
            / "best_model.pt"
        ),
        description="PREDICT v5 LSTM r0.15 (history sequence + peer-growth residual)",
    ),
    "predict_tabular": RealdataArtifact(
        expert_id="predict",
        checkpoint_path=str(
            REPO_ROOT
            / "runs"
            / "realdata_experts"
            / "predict_tabular_elastic_net_flat_residual"
            / "artifact.joblib"
        ),
        description="PREDICT tabular ElasticNet flat residual (best revenue MAPE)",
    ),
}


class RealdataDemoService:
    # Year range used during v5 training for consistent year_value normalization.
    PREDICT_TRAINING_YEARS = list(range(2016, 2026))
    PREDICT_TRAINING_REVENUE_MEDIAN = 1e10
    PREDICT_MODEL_WEIGHT = 0.35
    PREDICT_MIN_GROWTH = -0.5
    PREDICT_MAX_GROWTH = 0.7

    def __init__(self, router_dir: str | Path) -> None:
        self.router = MoERouter.load(router_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sessions: dict[str, DocumentSession] = {}
        self.uploaded_documents: dict[str, UploadedDocument] = {}
        self.llm_extractor = LLMDocumentExtractor()

        self.sum_model = self._load_sum_model()
        self.count_model = self._load_count_model()
        self.predict_tabular_artifact = self._load_predict_tabular_artifact()
        self.predict_model = self._load_predict_model()

    def _load_sum_model(self) -> SumV2Model:
        payload = torch.load(ARTIFACTS["sum"].checkpoint_path, map_location=self.device)
        model = SumV2Model(**payload["model_config"]).to(self.device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model

    def _load_count_model(self) -> CountV2Model:
        payload = torch.load(ARTIFACTS["count"].checkpoint_path, map_location=self.device)
        model = CountV2Model(**payload["model_config"]).to(self.device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model

    def _load_predict_model(self) -> PredictV5GRUModel:
        payload = torch.load(ARTIFACTS["predict"].checkpoint_path, map_location=self.device)
        self.predict_history_len = int(payload.get("history_len", 5))
        model = PredictV5GRUModel(**payload["model_config"]).to(self.device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model

    def _load_predict_tabular_artifact(self) -> dict[str, Any] | None:
        artifact_path = Path(ARTIFACTS["predict_tabular"].checkpoint_path)
        if not artifact_path.exists():
            return None
        return load_predict_tabular_artifact(artifact_path)

    def _predict_sum(self, example: dict) -> float:
        graph = sum_example_to_data(example)
        batch = Batch.from_data_list([graph]).to(self.device)
        with torch.no_grad():
            prediction, _ = self.sum_model(batch.x, batch.node_values, batch.batch, batch.query_attr)
        return float(prediction.item())

    def _predict_count(self, example: dict) -> int:
        graph = count_example_to_data(example)
        batch = Batch.from_data_list([graph]).to(self.device)
        with torch.no_grad():
            prediction, _ = self.count_model(batch.x, batch.node_values, batch.batch, batch.query_conditions)
        return round(float(prediction.item()))

    def _predict_predict(
        self,
        example: dict,
        extracted_document: ParsedFinancialDocument,
    ) -> tuple[float, dict[str, Any]]:
        history_map: dict[tuple[str, int], tuple[float, float]] = {}
        for r in extracted_document.records:
            if r.year is not None and r.revenue is not None and r.employees is not None:
                history_map[(r.company_name, r.year)] = (float(r.revenue), float(r.employees))

        target_name = example["target"]["node_name"]
        selected_year = self._extract_year_from_graph_id(example["graph"]["graph_id"])
        prev_revenue = 0.0
        if selected_year is not None:
            prev_revenue = float(history_map.get((target_name, selected_year - 1), (0.0, 0.0))[0])

        if self.predict_tabular_artifact is not None:
            try:
                return self._predict_predict_tabular(
                    example,
                    history_map,
                    prev_revenue,
                )
            except Exception as exc:
                prediction, details = self._predict_predict_v5(example, history_map, prev_revenue)
                details["tabular_fallback_error"] = str(exc)
                return prediction, details

        return self._predict_predict_v5(example, history_map, prev_revenue)

    def _predict_predict_tabular(
        self,
        example: dict,
        history_map: dict[tuple[str, int], tuple[float, float]],
        prev_revenue: float,
    ) -> tuple[float, dict[str, Any]]:
        if self.predict_tabular_artifact is None:
            raise RuntimeError("Predict tabular artifact is not loaded.")

        scaled_example, scaled_history_map, input_scale_factor = self._scale_predict_v5_inputs(
            example,
            history_map,
        )
        resources = PredictV6Resources(
            year_vocab={y: i for i, y in enumerate(self.PREDICT_TRAINING_YEARS)},
            history_map=scaled_history_map,
        )
        sample = predict_v6_example_to_sample(
            scaled_example,
            resources,
            history_len=int(self.predict_tabular_artifact.get("history_len", 5)),
            max_peers=int(self.predict_tabular_artifact.get("max_peers", 32)),
        )
        raw_growth = predict_tabular_growth(sample, self.predict_tabular_artifact)
        growth_details = self._stabilize_predict_v5_growth(sample, raw_growth)

        decoded = self._decode_growth_to_revenue(growth_details["final_growth"], prev_revenue)
        return decoded, {
            **growth_details,
            "prev_revenue": prev_revenue,
            "input_scale_factor": input_scale_factor,
            "history_len": int(self.predict_tabular_artifact.get("history_len", 5)),
            "max_peers": int(self.predict_tabular_artifact.get("max_peers", 32)),
            "model": self.predict_tabular_artifact.get("model_name", "elastic_net"),
            "feature_set": self.predict_tabular_artifact.get("feature_set", "flat"),
            "target_mode": self.predict_tabular_artifact.get("target_mode", "residual"),
            "artifact": ARTIFACTS["predict_tabular"].checkpoint_path,
            "strategy": "predict_tabular_elastic_net_flat_residual_scaled_growth",
        }

    def _predict_predict_v5(
        self,
        example: dict,
        history_map: dict[tuple[str, int], tuple[float, float]],
        prev_revenue: float,
    ) -> tuple[float, dict[str, Any]]:
        scaled_example, scaled_history_map, input_scale_factor = self._scale_predict_v5_inputs(
            example,
            history_map,
        )
        resources = PredictV5Resources(
            year_vocab={y: i for i, y in enumerate(self.PREDICT_TRAINING_YEARS)},
            history_map=scaled_history_map,
        )
        sample = predict_v5_example_to_sample(
            scaled_example,
            resources,
            history_len=getattr(self, "predict_history_len", 5),
        )
        batch = {key: value.unsqueeze(0).to(self.device) for key, value in sample.items()}
        with torch.no_grad():
            pred_growth = self.predict_model(
                batch["history_seq"],
                batch["history_mask"],
                batch["static_features"],
                batch["baseline_growth"],
            )
        raw_growth = float(pred_growth[0].cpu())
        growth_details = self._stabilize_predict_v5_growth(sample, raw_growth)

        decoded = self._decode_growth_to_revenue(growth_details["final_growth"], prev_revenue)
        return decoded, {
            **growth_details,
            "prev_revenue": prev_revenue,
            "input_scale_factor": input_scale_factor,
            "history_len": getattr(self, "predict_history_len", 5),
            "strategy": "predict_v5_lstm_scaled_growth",
        }

    @staticmethod
    def _extract_year_from_graph_id(graph_id: str) -> int | None:
        for token in graph_id.replace("-", "_").split("_"):
            if token.isdigit() and len(token) == 4:
                return int(token)
        return None

    @staticmethod
    def _scale_predict_v5_inputs(
        example: dict,
        history_map: dict[tuple[str, int], tuple[float, float]],
    ) -> tuple[dict, dict[tuple[str, int], tuple[float, float]], float]:
        revenues = [float(revenue) for revenue, _ in history_map.values() if revenue > 0]
        for node in example["graph"]["nodes"]:
            revenue = node.get("attributes", {}).get("revenue")
            if revenue is not None and float(revenue) > 0:
                revenues.append(float(revenue))
        if not revenues:
            return example, history_map, 1.0

        median_revenue = sorted(revenues)[len(revenues) // 2]
        if median_revenue <= 0:
            return example, history_map, 1.0
        factor = RealdataDemoService.PREDICT_TRAINING_REVENUE_MEDIAN / median_revenue
        if 0.1 < factor < 10.0:
            return example, history_map, 1.0

        scaled_example = copy.deepcopy(example)
        scaled_history_map = {
            key: (float(revenue) * factor if revenue > 0 else float(revenue), float(employees))
            for key, (revenue, employees) in history_map.items()
        }
        for node in scaled_example["graph"]["nodes"]:
            attrs = node.get("attributes", {})
            if attrs.get("revenue") is not None:
                attrs["revenue"] = float(attrs["revenue"]) * factor
        return scaled_example, scaled_history_map, factor

    @staticmethod
    def _stabilize_predict_v5_growth(sample: dict[str, torch.Tensor], raw_growth: float) -> dict[str, Any]:
        static_features = sample["static_features"].view(-1)
        baseline = float(sample["baseline_growth"].view(-1)[0].cpu())
        peer_std = float(static_features[6].cpu())
        n_peers = int(round(float(static_features[12].cpu()) * 100.0))
        history_count = int(round(float(static_features[13].cpu()) * max(sample["history_mask"].numel(), 1)))

        trust_band = max(0.08, (3.0 * max(peer_std, 0.0)) + 0.04)
        if n_peers < 2:
            trust_band = max(trust_band, 0.18)
        if history_count < 2:
            trust_band = max(trust_band, 0.22)
        trust_band = min(trust_band, 0.35)

        clipped_model_growth = min(max(raw_growth, baseline - trust_band), baseline + trust_band)
        final_growth = min(
            max(clipped_model_growth, RealdataDemoService.PREDICT_MIN_GROWTH),
            RealdataDemoService.PREDICT_MAX_GROWTH,
        )
        return {
            "raw_model_growth": raw_growth,
            "baseline_growth": baseline,
            "clipped_model_growth": clipped_model_growth,
            "final_growth": final_growth,
            "peer_growth_std": peer_std,
            "n_peers": n_peers,
            "history_count": history_count,
            "trust_band": trust_band,
        }

    @staticmethod
    def _decode_growth_to_revenue(growth: float, prev_revenue: float) -> float:
        if prev_revenue <= 0:
            return 0.0
        decoded = math.expm1(signed_log1p(prev_revenue) + growth)
        if not math.isfinite(decoded):
            return 0.0
        return float(max(decoded, 0.0))

    def _build_document_summary(
        self,
        file_name: str,
        media_type: str,
        content: bytes,
    ) -> tuple[DocumentSession, UploadedDocument]:
        uploaded_document = ingest_document(file_name, media_type, content)
        session_id = f"doc_{uuid.uuid4().hex[:10]}"

        session = DocumentSession(
            session_id=session_id,
            file_name=file_name or uploaded_document.file_name,
            media_type=media_type or uploaded_document.media_type or "application/octet-stream",
            size_bytes=len(content),
            document_text=uploaded_document.text,
            preview_text=uploaded_document.preview_text,
            parse_summary=uploaded_document.parse_summary,
            graph_summary=uploaded_document.graph_summary,
            query_examples=uploaded_document.query_examples,
        )
        self.sessions[session.session_id] = session
        self.uploaded_documents[session.session_id] = uploaded_document
        return session, uploaded_document

    def create_document_session(
        self,
        file_name: str,
        media_type: str,
        content: bytes,
    ) -> dict[str, object]:
        session, _ = self._build_document_summary(file_name, media_type, content)
        return session.to_dict()

    def get_document_session(self, session_id: str) -> dict[str, object]:
        session = self.sessions[session_id]
        return session.to_dict()

    def _serialize_record(
        self,
        record: Any,
        in_graph_keys: set[tuple[str, int | None]],
        target_company: str | None,
        selected_year: int | None,
    ) -> dict[str, Any]:
        return {
            "company_name": record.company_name,
            "industry": record.industry,
            "year": record.year,
            "revenue": record.revenue,
            "operating_profit": record.operating_profit,
            "net_profit": record.net_profit,
            "employees": record.employees,
            "in_graph": (record.company_name, record.year) in in_graph_keys,
            "is_target": target_company is not None and record.company_name == target_company,
            "is_history": (
                target_company is not None
                and record.company_name == target_company
                and selected_year is not None
                and record.year is not None
                and record.year < selected_year
            ),
            "source_lines": record.raw.get("source_lines") if isinstance(record.raw, dict) else None,
            "evidence": record.raw.get("evidence") if isinstance(record.raw, dict) else None,
        }

    def _build_evidence_view(
        self,
        uploaded_document: UploadedDocument,
        extracted_document: Any,
        example: dict[str, Any],
        graph_build: dict[str, Any],
        extraction_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lines = [line.rstrip() for line in uploaded_document.text.splitlines() if line.strip()]
        company_names = [node["name"] for node in example["graph"]["nodes"]]
        matched_industry = graph_build.get("matched_industry")
        selected_year = graph_build.get("selected_year")
        query_text = example["task"]["query_text"]
        query_tokens = {token.lower() for token in query_text.replace("。", " ").split() if len(token) >= 3}
        llm_line_reasons = self._llm_evidence_line_reasons(extraction_metadata)

        evidence_lines = []
        for index, line in enumerate(lines):
            reasons: list[str] = []
            lower_line = line.lower()
            for name in company_names:
                if name.lower() in lower_line:
                    reasons.append(name)
            if matched_industry and str(matched_industry).lower() in lower_line:
                reasons.append(str(matched_industry))
            if selected_year is not None and str(selected_year) in line:
                reasons.append(str(selected_year))
            if any(token in lower_line for token in query_tokens):
                reasons.append("query")
            if index + 1 in llm_line_reasons:
                reasons.extend(llm_line_reasons[index + 1])
            evidence_lines.append(
                {
                    "index": index + 1,
                    "text": line,
                    "relevant": bool(reasons),
                    "reasons": sorted(set(reasons))[:4],
                }
            )

        in_graph_keys = {
            (node["name"], graph_build.get("selected_year"))
            for node in example["graph"]["nodes"]
        }
        if graph_build.get("selected_year") is None:
            in_graph_keys = {(node["name"], None) for node in example["graph"]["nodes"]}

        extracted_records = [
            self._serialize_record(
                record,
                in_graph_keys,
                graph_build.get("target_company"),
                graph_build.get("selected_year"),
            )
            for record in extracted_document.records
        ]

        return {
            "line_count": len(evidence_lines),
            "relevant_line_count": sum(1 for line in evidence_lines if line["relevant"]),
            "lines": evidence_lines,
            "records": extracted_records,
        }

    @staticmethod
    def _llm_evidence_line_reasons(extraction_metadata: dict[str, Any] | None) -> dict[int, list[str]]:
        if not extraction_metadata:
            return {}
        reasons: dict[int, list[str]] = {}
        for span in extraction_metadata.get("llm", {}).get("evidence_spans", []):
            if not isinstance(span, dict):
                continue
            label = span.get("id") or "LLM证据"
            for line_index in span.get("line_indices", []):
                try:
                    line_no = int(line_index)
                except (TypeError, ValueError):
                    continue
                reasons.setdefault(line_no, []).append(str(label))
        return reasons

    def _build_graph_view(
        self,
        example: dict[str, Any],
        graph_build: dict[str, Any],
        route: dict[str, Any],
    ) -> dict[str, Any]:
        expert_labels = {
            "sum": "求和专家",
            "count": "计数专家",
            "predict": "预测专家",
        }
        metric_labels = {
            "revenue": "收入",
            "operating_profit": "营业利润",
            "net_profit": "净利润",
            "employees": "员工数",
        }
        op_labels = {
            "==": "=",
            ">=": "≥",
            "<=": "≤",
            ">": ">",
            "<": "<",
        }

        def metric_text(key: str, value: Any) -> str:
            label = metric_labels.get(key, key)
            if value is None:
                return label
            numeric = float(value)
            abs_value = abs(numeric)
            if abs_value >= 1e8:
                rendered = f"{numeric / 1e8:.2f}亿"
            elif abs_value >= 1e4:
                rendered = f"{numeric / 1e4:.2f}万"
            elif abs_value >= 100:
                rendered = f"{numeric:.1f}"
            elif abs_value >= 1:
                rendered = f"{numeric:.2f}"
            else:
                rendered = f"{numeric:.4f}"
            return f"{label} {rendered}"

        nodes: list[dict[str, Any]] = [
            {
                "id": "query",
                "label": "当前问题",
                "kind": "query",
                "accent": "query",
                "subtitle": example["task"]["query_text"],
            },
            {
                "id": "expert",
                "label": expert_labels.get(route["selected_expert"], route["selected_expert"]),
                "kind": "expert",
                "accent": route["selected_expert"],
                "subtitle": "数值推理执行",
            },
        ]
        edges: list[dict[str, Any]] = [
            {"source": "query", "target": "expert", "label": "专家选择"},
        ]

        filter_specs = []
        if graph_build.get("matched_industry"):
            filter_specs.append(("industry", f'行业 = {graph_build["matched_industry"]}'))
        if graph_build.get("selected_year") is not None:
            filter_specs.append(("year", f'年份 = {graph_build["selected_year"]}'))
        if graph_build.get("target_attr"):
            attr = metric_labels.get(str(graph_build["target_attr"]), str(graph_build["target_attr"]))
            filter_specs.append(("metric", f"统计指标 = {attr}"))
        if graph_build.get("target_company"):
            filter_specs.append(("target", f'目标公司 = {graph_build["target_company"]}'))
        for index, condition in enumerate(graph_build.get("conditions", []), start=1):
            attr = metric_labels.get(condition["attribute"], condition["attribute"])
            op = op_labels.get(condition["op"], condition["op"])
            filter_specs.append(
                (
                    f"condition_{index}",
                    f"{attr} {op} {condition['value']}",
                )
            )

        for filter_id, label in filter_specs:
            node_id = f"filter_{filter_id}"
            nodes.append(
                {
                    "id": node_id,
                    "label": label,
                    "kind": "filter",
                    "accent": "filter",
                    "subtitle": "抽取条件",
                }
            )
            edges.append({"source": "query", "target": node_id, "label": "抽取约束"})

        selected_year = graph_build.get("selected_year")
        target_company = graph_build.get("target_company")
        for node in example["graph"]["nodes"]:
            node_id = f'graph_{node["id"]}'
            attributes = node["attributes"]
            metrics = []
            for key in ("revenue", "operating_profit", "net_profit", "employees"):
                if key in attributes:
                    metrics.append(metric_text(key, attributes[key]))
            nodes.append(
                {
                    "id": node_id,
                    "label": node["name"],
                    "kind": "company",
                    "accent": "target" if node["name"] == target_company else "company",
                    "subtitle": " · ".join(metrics[:2]) or "公司节点",
                }
            )
            edges.append({"source": "expert", "target": node_id, "label": "图谱输入"})

        if target_company and selected_year is not None:
            history_years = graph_build.get("history_years", [])
            for year in history_years:
                if year >= selected_year:
                    continue
                history_id = f"history_{year}"
                nodes.append(
                    {
                        "id": history_id,
                        "label": f"{year} 年历史",
                        "kind": "history",
                        "accent": "history",
                        "subtitle": target_company,
                    }
                )
                edges.append({"source": history_id, "target": "expert", "label": "历史上下文"})

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "num_nodes": len(example["graph"]["nodes"]),
                "num_edges": len(example["graph"]["edges"]),
                "mode": graph_build.get("extraction_mode"),
            },
        }

    def _extract_query_document(self, session_id: str, query: str, expert_id: str):
        uploaded_document = self.uploaded_documents[session_id]

        if self.llm_extractor.available:
            try:
                extraction = self.llm_extractor.extract_with_metadata(
                    uploaded_document,
                    query,
                    expert_id,
                )
                return extraction.document, "llm", extraction.metadata
            except Exception:
                if uploaded_document.structured_document is None:
                    raise

        if uploaded_document.structured_document is not None:
            return (
                uploaded_document.structured_document,
                "structured_fallback",
                build_structured_extraction_metadata(
                    uploaded_document.structured_document,
                    query,
                    expert_id,
                ),
            )

        raise ValueError(
            "LLM extraction is not available and the uploaded document could not be parsed into structured financial rows locally."
        )

    def run(self, query: str, top_k: int = 7, session_id: str | None = None) -> dict[str, Any]:
        if session_id is None:
            raise ValueError("A document session is required before running the demo.")
        if session_id not in self.sessions or session_id not in self.uploaded_documents:
            raise KeyError("Session not found.")

        decision = self.router.route(query, top_k=top_k).to_dict()
        extracted_document, extraction_mode, extraction_metadata = self._extract_query_document(
            session_id,
            query,
            decision["selected_expert"],
        )
        try:
            built = build_query_example(
                extracted_document,
                query,
                decision["selected_expert"],
                session_id,
            )
        except ValueError as exc:
            uploaded_document = self.uploaded_documents[session_id]
            if extraction_mode != "llm" or uploaded_document.structured_document is None:
                raise
            extracted_document = uploaded_document.structured_document
            extraction_mode = "structured_fallback_after_llm_build_error"
            extraction_metadata = build_structured_extraction_metadata(
                extracted_document,
                query,
                decision["selected_expert"],
            )
            extraction_metadata["fallback_reason"] = str(exc)
            built = build_query_example(
                extracted_document,
                query,
                decision["selected_expert"],
                session_id,
            )
        example = built.example

        if decision["selected_expert"] == "sum":
            prediction = self._predict_sum(example)
            predict_details = None
        elif decision["selected_expert"] == "count":
            prediction = self._predict_count(example)
            predict_details = None
        elif decision["selected_expert"] == "predict":
            prediction, predict_details = self._predict_predict(example, extracted_document)
        else:
            raise ValueError(f"Unsupported expert: {decision['selected_expert']}")

        answer = float(built.answer)
        graph_build = {
            **built.metadata,
            "extraction_mode": extraction_mode,
            "llm_extractor_available": self.llm_extractor.available,
        }
        extraction_validation = validate_extracted_graph(
            extracted_document,
            extraction_metadata.get("plan", {}),
            graph_build,
            example,
            extraction_metadata,
        )
        graph_extraction = {
            **extraction_metadata,
            "validation": extraction_validation,
        }
        graph_build.update(
            {
                "graph_quality_score": extraction_validation["quality_score"],
                "expert_ready": extraction_validation["expert_ready"],
                "validation_error_count": len(extraction_validation["errors"]),
                "validation_warning_count": len(extraction_validation["warnings"]),
            }
        )
        result = {
            "query": query,
            "document": self.sessions[session_id].to_dict(),
            "graph_extraction": graph_extraction,
            "evidence": self._build_evidence_view(
                self.uploaded_documents[session_id],
                extracted_document,
                example,
                graph_build,
                graph_extraction,
            ),
            "retrieve": {
                "top_k": top_k,
                "neighbors": decision["neighbors"],
            },
            "route": {
                "selected_expert": decision["selected_expert"],
                "confidence": decision["confidence"],
                "expert_scores": decision["expert_scores"],
            },
            "graph_view": self._build_graph_view(
                example,
                graph_build,
                decision,
            ),
            "execute": {
                "artifact": ARTIFACTS[decision["selected_expert"]].__dict__,
                "dataset": "uploaded_document",
                "sample_id": example["sample_id"],
                "graph_id": example["graph"]["graph_id"],
                "task_kind": example["task"]["kind"],
                "query_text": example["task"]["query_text"],
                "num_nodes": len(example["graph"]["nodes"]),
                "num_edges": len(example["graph"]["edges"]),
                "graph_build": graph_build,
                "graph_extraction": graph_extraction,
                "predict_details": predict_details,
            },
            "result": {
                "prediction": prediction,
                "answer": answer,
                "abs_error": abs(prediction - answer),
            },
        }
        return result

    def examples(self) -> list[str]:
        return [
            "LIFE INSURANCE行业的收入总和是多少",
            "请统计收入等于 694.1 的公司数量",
            "请预测 Company_3 的收入",
        ]
