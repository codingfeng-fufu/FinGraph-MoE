from dataclasses import dataclass

from moe_router.realdata_query_bank import build_realdata_query_bank
from moe_router.router import MoERouter
from moe_router.query_rewrite import QueryRewrite
from moe_router.records import QueryRecord


@dataclass
class FakeEncoder:
    keyword_vectors: dict[str, list[float]]

    def encode_corpus(self, texts: list[str]):
        return self._encode(texts)

    def encode_queries(self, texts: list[str]):
        return self._encode(texts)

    def _encode(self, texts: list[str]):
        import numpy as np

        encoded = []
        for text in texts:
            lowered = text.casefold()
            vector = np.zeros(3, dtype=np.float32)
            for keyword, values in self.keyword_vectors.items():
                if keyword in lowered:
                    vector += np.array(values, dtype=np.float32)
            if not vector.any():
                vector += np.array([0.01, 0.01, 0.01], dtype=np.float32)
            vector = vector / np.linalg.norm(vector)
            encoded.append(vector)
        return np.stack(encoded)


@dataclass
class FakeRewriter:
    def rewrite(self, query: str):
        return [QueryRewrite(text=query, weight=1.0, source="original")]


def test_realdata_query_bank_contains_three_experts() -> None:
    records = build_realdata_query_bank()
    experts = {record.expert for record in records}
    assert experts == {"sum", "count", "predict"}
    assert any(record.metadata.get("source_dataset") == "router_anchors" for record in records)


def test_realdata_router_basic_queries() -> None:
    records = [
        QueryRecord("sum_0", "sum", "行业收入总和", {"source_dataset": "test"}),
        QueryRecord("count_0", "count", "满足条件的公司数量", {"source_dataset": "test"}),
        QueryRecord("predict_0", "predict", "预测某公司的收入", {"source_dataset": "test"}),
        QueryRecord("anchor_predict", "predict", "请预测某公司在某年的 revenue", {"source_dataset": "router_anchors"}),
    ]
    router = MoERouter(
        encoder=FakeEncoder(
            {
                "总和": [1.0, 0.0, 0.0],
                "统计": [0.0, 1.0, 0.0],
                "数量": [0.0, 1.0, 0.0],
                "预测": [0.0, 0.0, 1.0],
                "公司": [0.02, 0.02, 0.02],
                "收入": [0.02, 0.02, 0.02],
            }
        ),
        query_rewriter=FakeRewriter(),
    ).fit(records)

    sum_decision = router.route("LIFE INSURANCE行业的收入总和是多少")
    count_decision = router.route("请统计满足条件的公司数量")
    predict_decision = router.route("请预测 Company_3 在 2024 年的收入")

    assert sum_decision.selected_expert == "sum"
    assert count_decision.selected_expert == "count"
    assert predict_decision.selected_expert == "predict"
