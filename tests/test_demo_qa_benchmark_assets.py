import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from moe_router.document_graph import FinancialRecord, build_parsed_document_from_records
from moe_router.graph_extraction import GraphExtractionResult


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "data" / "demo" / "qa_benchmark_v1"
RUNNER_PATH = REPO_ROOT / "scripts" / "moe" / "run_demo_qa_benchmark.py"


def _load_benchmark_runner():
    spec = importlib.util.spec_from_file_location("run_demo_qa_benchmark", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_demo_qa_benchmark_assets_exist_and_are_balanced() -> None:
    dataset_path = BENCHMARK_ROOT / "qa_dataset.json"
    assert dataset_path.exists()

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    domains = payload["domains"]
    qas = payload["qas"]

    assert len(domains) == 5
    assert len(qas) == 50

    domain_ids = {domain["domain_id"] for domain in domains}
    assert domain_ids == {
        "insurance",
        "banking",
        "healthcare",
        "retail",
        "industrial_equipment",
    }

    qa_counts: dict[str, int] = {domain_id: 0 for domain_id in domain_ids}
    for qa in qas:
        assert qa["domain_id"] in domain_ids
        assert qa["expected_expert"] in {"sum", "count", "predict"}
        assert isinstance(qa["answer"], (int, float))
        qa_counts[qa["domain_id"]] += 1
        document_path = BENCHMARK_ROOT / qa["document_file"]
        assert document_path.exists()

    assert set(qa_counts.values()) == {10}


def test_demo_qa_benchmark_graph_extraction_cache(tmp_path: Path) -> None:
    calls = {"count": 0}
    document = SimpleNamespace(
        file_name="memo.md",
        media_type="text/markdown",
        text="Alpha revenue was 100 in 2024.",
    )

    def fake_extract(uploaded_document, query: str, expert_id: str) -> GraphExtractionResult:
        calls["count"] += 1
        parsed = build_parsed_document_from_records(
            uploaded_document.file_name,
            uploaded_document.media_type,
            [
                FinancialRecord(
                    company_name="Alpha",
                    industry="TEST",
                    revenue=100.0,
                    operating_profit=None,
                    net_profit=None,
                    employees=10.0,
                    year=2024,
                    source_index=0,
                    raw={"source_lines": [1]},
                )
            ],
            source_format="llm_query_conditioned",
        )
        return GraphExtractionResult(
            document=parsed,
            metadata={
                "source": "llm",
                "plan": {"expert_id": expert_id},
                "llm": {"confidence": 1.0, "evidence_spans": []},
                "draft_summary": {"record_count": 1, "edge_count": 0, "evidence_span_count": 0},
            },
        )

    service = SimpleNamespace(
        llm_extractor=SimpleNamespace(
            config=SimpleNamespace(model="mock-model"),
            extract_with_metadata=fake_extract,
        )
    )
    runner = _load_benchmark_runner()
    runner._install_graph_extraction_cache(service, tmp_path, refresh_cache=False)

    first = service.llm_extractor.extract_with_metadata(document, "收入总和是多少", "sum")
    second = service.llm_extractor.extract_with_metadata(document, "收入总和是多少", "sum")

    assert calls["count"] == 1
    assert first.metadata["cache"]["hit"] is False
    assert second.metadata["cache"]["hit"] is True
    assert second.document.records[0].company_name == "Alpha"
