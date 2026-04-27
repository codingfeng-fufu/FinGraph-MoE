from moe_router.document_graph import parse_financial_document, build_query_example
from moe_router.graph_extraction import (
    build_extraction_plan,
    build_structured_extraction_metadata,
    validate_extracted_graph,
)


DOCUMENT = """Company,Industry,Year,Revenue,Operating Profit,Net Profit,Employees
Alpha Insurance,LIFE INSURANCE,2023,1000,120,60,110
Alpha Insurance,LIFE INSURANCE,2024,1500,180,70,115
Beta Insurance,LIFE INSURANCE,2023,1100,150,80,125
Beta Insurance,LIFE INSURANCE,2024,1200,160,90,130
Gamma Insurance,LIFE INSURANCE,2024,900,100,40,210
"""


def test_count_extraction_plan_requires_candidate_universe() -> None:
    plan = build_extraction_plan(
        "请统计 2024 年收入不低于 1000 且净利润不低于 60 的保险公司数量。",
        "count",
    )

    assert plan["graph_scope"] == "candidate_universe"
    assert plan["needs_candidate_universe"] is True
    assert "revenue" in plan["required_fields"]
    assert "net_profit" in plan["required_fields"]


def test_predict_extraction_validation_checks_history_and_peer_previous_year() -> None:
    document = parse_financial_document(
        "financials.csv",
        "text/csv",
        DOCUMENT.encode("utf-8"),
    )
    query = "请预测 Alpha Insurance 在 2024 年的收入。"
    built = build_query_example(document, query, "predict", "doc_test")
    metadata = build_structured_extraction_metadata(document, query, "predict")

    validation = validate_extracted_graph(
        document,
        metadata["plan"],
        built.metadata,
        built.example,
        metadata,
    )

    assert validation["schema_valid"] is True
    assert validation["expert_ready"] is True
    assert validation["expert_checks"]["target_history_count"] == 1
    assert validation["expert_checks"]["peer_previous_year_count"] == 1
    assert validation["quality_score"] >= 0.8
