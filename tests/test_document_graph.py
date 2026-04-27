from moe_router.document_graph import (
    build_query_example,
    build_query_suggestions,
    ingest_document,
    parse_financial_document,
)


CSV_DOCUMENT = """Company,Industry,Year,Revenue,Operating Profit,Net Profit,Employees
Alpha Insurance,LIFE INSURANCE,2023,1000,120,60,110
Alpha Insurance,LIFE INSURANCE,2024,1500,180,70,115
Beta Insurance,LIFE INSURANCE,2024,1200,160,90,130
Gamma Bank,BANKS,2024,900,100,40,210
"""


def test_parse_financial_document_from_csv() -> None:
    document = parse_financial_document(
        "financials.csv",
        "text/csv",
        CSV_DOCUMENT.encode("utf-8"),
    )

    assert document.source_format == "delimited_text"
    assert document.parse_summary["parsed_record_count"] == 4
    assert "company_name" in document.parse_summary["mapped_fields"]
    assert document.graph_summary["record_count"] == 4
    assert document.graph_summary["records_with_predict_fields"] == 4
    assert "Alpha Insurance" in document.preview_text


def test_build_query_suggestions_uses_uploaded_document() -> None:
    document = parse_financial_document(
        "financials.csv",
        "text/csv",
        CSV_DOCUMENT.encode("utf-8"),
    )

    suggestions = build_query_suggestions(document)

    assert len(suggestions) == 3
    assert suggestions[0].endswith("收入总和是多少")
    assert "收入不低于" in suggestions[1]
    assert "请预测" in suggestions[2]


def test_build_sum_query_example_defaults_to_latest_year() -> None:
    document = parse_financial_document(
        "financials.csv",
        "text/csv",
        CSV_DOCUMENT.encode("utf-8"),
    )

    built = build_query_example(
        document,
        "LIFE INSURANCE行业的收入总和是多少",
        "sum",
        "doc_test",
    )

    assert built.expert_id == "sum"
    assert built.answer == 2700.0
    assert built.metadata["selected_year"] == 2024
    assert built.metadata["default_year_applied"] is True
    assert built.metadata["matched_industry"] == "LIFE INSURANCE"
    assert built.example["task"]["target"]["attribute"] == "revenue"
    assert len(built.example["graph"]["nodes"]) == 2


def test_build_count_query_example_parses_multiple_conditions() -> None:
    document = parse_financial_document(
        "financials.csv",
        "text/csv",
        CSV_DOCUMENT.encode("utf-8"),
    )

    built = build_query_example(
        document,
        "请统计 revenue >= 1000 and net profit < 80 的公司数量",
        "count",
        "doc_test",
    )

    assert built.expert_id == "count"
    assert built.answer == 1.0
    assert built.metadata["selected_year"] == 2024
    assert len(built.metadata["conditions"]) == 2
    assert built.example["task"]["target"]["conditions"][0]["attribute"] == "revenue"
    assert len(built.example["graph"]["nodes"]) == 3


def test_build_predict_query_example_creates_history_resources() -> None:
    document = parse_financial_document(
        "financials.csv",
        "text/csv",
        CSV_DOCUMENT.encode("utf-8"),
    )

    built = build_query_example(
        document,
        "预测 Alpha Insurance 在 2024 的 revenue",
        "predict",
        "doc_test",
    )

    assert built.expert_id == "predict"
    assert built.answer == 1500.0
    assert built.metadata["selected_year"] == 2024
    assert built.metadata["target_company"] == "Alpha Insurance"
    assert built.predict_resources is not None
    assert built.predict_resources.history_map[("Alpha Insurance", 2023)] == (1000.0, 110.0)
    assert built.example["target"]["node_name"] == "Alpha Insurance"
    assert built.example["target"]["attribute"] == "revenue"
    assert len(built.example["graph"]["nodes"]) == 3


def test_ingest_document_allows_unstructured_small_text() -> None:
    uploaded = ingest_document(
        "memo.txt",
        "text/plain",
        (
            "Alpha Insurance reported revenue 1500 and net profit 70 in 2024. "
            "Beta Insurance reported revenue 1200 and net profit 90 in 2024."
        ).encode("utf-8"),
    )

    assert uploaded.structured_document is None
    assert uploaded.parse_summary["llm_extraction_ready"] is True
    assert uploaded.parse_summary["structured_parse_available"] is False
    assert "Alpha Insurance" in uploaded.preview_text
