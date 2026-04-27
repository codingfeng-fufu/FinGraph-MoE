import json

from moe_router.document_graph import ingest_document
from moe_router.llm_document_extractor import LLMDocumentExtractor, LLMExtractorConfig


def test_llm_document_extractor_builds_parsed_document(monkeypatch) -> None:
    uploaded = ingest_document(
        "memo.txt",
        "text/plain",
        (
            "Alpha Insurance revenue was 1500 in 2024 with employees 115. "
            "In 2023 Alpha Insurance revenue was 1000 with employees 110. "
            "Beta Insurance revenue was 1200 in 2024."
        ).encode("utf-8"),
    )
    extractor = LLMDocumentExtractor(
        LLMExtractorConfig(
            base_url="http://localhost:11434/v1",
            model="mock-model",
            api_key=None,
        )
    )

    def fake_chat_completion(messages):
        assert "Alpha Insurance" in messages[1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "confidence": 0.94,
                                "units": {"revenue": "millions"},
                                "evidence_spans": [
                                    {
                                        "id": "ev_1",
                                        "line_indices": [1, 2],
                                        "text": "Alpha Insurance revenue facts",
                                        "confidence": 0.9,
                                    }
                                ],
                                "records": [
                                    {
                                        "company_name": "Alpha Insurance",
                                        "industry": "LIFE INSURANCE",
                                        "year": 2024,
                                        "revenue": 1500,
                                        "operating_profit": 180,
                                        "net_profit": 70,
                                        "employees": 115,
                                        "source_lines": [1],
                                        "evidence": "Alpha Insurance revenue was 1500 in 2024",
                                    },
                                    {
                                        "company_name": "Alpha Insurance",
                                        "industry": "LIFE INSURANCE",
                                        "year": 2023,
                                        "revenue": 1000,
                                        "operating_profit": 120,
                                        "net_profit": 60,
                                        "employees": 110,
                                    },
                                    {
                                        "company_name": "Beta Insurance",
                                        "industry": "LIFE INSURANCE",
                                        "year": 2024,
                                        "revenue": 1200,
                                        "operating_profit": 160,
                                        "net_profit": 90,
                                        "employees": None,
                                    },
                                ]
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(extractor, "_chat_completion", fake_chat_completion)
    parsed = extractor.extract(uploaded, "预测 Alpha Insurance 在 2024 的 revenue", "predict")

    assert parsed.source_format == "llm_query_conditioned"
    assert parsed.parse_summary["parsed_record_count"] == 3
    assert parsed.graph_summary["records_with_predict_fields"] == 2
    assert parsed.records[0].company_name == "Alpha Insurance"


def test_llm_document_extractor_returns_graph_extraction_metadata(monkeypatch) -> None:
    uploaded = ingest_document(
        "memo.txt",
        "text/plain",
        "Alpha Insurance revenue was 1500 in 2024 with employees 115.".encode("utf-8"),
    )
    extractor = LLMDocumentExtractor(
        LLMExtractorConfig(
            base_url="http://localhost:11434/v1",
            model="mock-model",
            api_key=None,
        )
    )

    def fake_chat_completion(messages):
        assert "Extraction plan" in messages[1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "confidence": 0.8,
                                "evidence_spans": [
                                    {
                                        "id": "ev_1",
                                        "line_indices": [1],
                                        "text": "Alpha Insurance revenue was 1500",
                                    }
                                ],
                                "records": [
                                    {
                                        "company_name": "Alpha Insurance",
                                        "industry": "LIFE INSURANCE",
                                        "year": 2024,
                                        "revenue": 1500,
                                        "operating_profit": None,
                                        "net_profit": None,
                                        "employees": 115,
                                    }
                                ],
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(extractor, "_chat_completion", fake_chat_completion)
    result = extractor.extract_with_metadata(uploaded, "预测 Alpha Insurance 在 2024 年的收入", "predict")

    assert result.metadata["source"] == "llm"
    assert result.metadata["plan"]["expert_id"] == "predict"
    assert result.metadata["draft_summary"]["evidence_span_count"] == 1
    assert result.metadata["llm"]["confidence"] == 0.8


def test_llm_config_reads_qwen_defaults_from_dotenv(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for key in (
        "DASHSCOPE_API_KEY",
        "LLM_GRAPH_API_KEY",
        "OPENAI_API_KEY",
        "LLM_GRAPH_BASE_URL",
        "OPENAI_BASE_URL",
        "LLM_GRAPH_MODEL",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    config = LLMExtractorConfig.from_env()

    assert config.api_key == "test-key"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.model == "qwen-plus"
    assert config.available is True
