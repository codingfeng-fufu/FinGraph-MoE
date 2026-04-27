from __future__ import annotations

from pathlib import Path

from realdata_experts.common import extract_year_from_graph_id, read_jsonl

from .records import QueryRecord


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _build_router_anchors(start_index: int) -> list[QueryRecord]:
    anchor_specs = {
        "sum": [
            "计算 某行业 公司 收入 总和",
            "请计算某一年某行业公司的收入总和",
            "计算 某行业 公司 净利润 总和",
            "计算 某行业 公司 营业利润 总和",
            "sum company revenue by industry",
            "total company revenue in an industry",
        ],
        "count": [
            "统计 满足 数值 条件 的 公司 数量",
            "请统计某年满足收入条件的公司数量",
            "请统计满足净利润条件的公司数量",
            "请统计满足营业利润条件的公司数量",
            "count companies with revenue threshold",
            "count companies matching numeric conditions",
        ],
        "predict": [
            "预测 某公司 的 revenue",
            "请预测某公司在某年的收入",
            "predict company revenue",
            "forecast company revenue for a year",
        ],
    }

    records: list[QueryRecord] = []
    index = start_index
    for expert, texts in anchor_specs.items():
        for anchor_idx, text in enumerate(texts):
            records.append(
                QueryRecord(
                    query_id=f"anchor_{expert}_{index:05d}",
                    expert=expert,
                    text=text,
                    metadata={
                        "source_dataset": "router_anchors",
                        "anchor_id": f"{expert}_{anchor_idx}",
                    },
                )
            )
            index += 1
    return records


def build_realdata_query_bank() -> list[QueryRecord]:
    records: list[QueryRecord] = []
    index = 0

    sum_rows = _read_jsonl_if_exists(REPO_ROOT / "data" / "realdata_inputs" / "sum_dataset2.jsonl")
    for row in sum_rows:
        if row.get("split") != "train":
            continue
        records.append(
            QueryRecord(
                query_id=f"real_sum_{index:05d}",
                expert="sum",
                text=row["task"]["query_text"],
                metadata={
                    "source_dataset": "data/realdata_inputs/sum_dataset2.jsonl",
                    "sample_id": row["sample_id"],
                },
            )
        )
        index += 1

    count_rows = _read_jsonl_if_exists(REPO_ROOT / "data" / "realdata_inputs" / "count_600.jsonl")
    for row in count_rows:
        if row.get("split") != "train":
            continue
        records.append(
            QueryRecord(
                query_id=f"real_count_{index:05d}",
                expert="count",
                text=row["task"]["query_text"],
                metadata={
                    "source_dataset": "data/realdata_inputs/count_600.jsonl",
                    "sample_id": row["sample_id"],
                },
            )
        )
        index += 1

    predict_rows = _read_jsonl_if_exists(REPO_ROOT / "data" / "realdata_inputs" / "prediction_dataset.jsonl")
    for row in predict_rows:
        year = extract_year_from_graph_id(row["graph"]["graph_id"])
        if year is None or year > 2022:
            continue
        records.append(
            QueryRecord(
                query_id=f"real_predict_{index:05d}",
                expert="predict",
                text=row["task"]["query_text"],
                metadata={
                    "source_dataset": "data/realdata_inputs/prediction_dataset.jsonl",
                    "sample_id": row["sample_id"],
                    "year": str(year),
                },
            )
        )
        index += 1

    records.extend(_build_router_anchors(index))
    return records
