from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from realdata_experts.common import read_jsonl, signed_log1p


COUNT_ATTRS = ("revenue", "net_profit", "operating_profit")
COUNT_OPS = ("==", "<", "<=", ">", ">=")
MAX_CONDITIONS = 2


@dataclass(frozen=True)
class CountV2SplitSummary:
    train_size: int
    val_size: int
    test_size: int


def _build_edge_index(nodes: list[dict], edges: list[dict]) -> torch.Tensor:
    node_to_index = {node["id"]: index for index, node in enumerate(nodes)}
    index_pairs = []
    for edge in edges:
        src = node_to_index[edge["source"]]
        dst = node_to_index[edge["target"]]
        index_pairs.append((src, dst))
        index_pairs.append((dst, src))
    if not index_pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(index_pairs, dtype=torch.long).t().contiguous()


def _normalize_condition(target: dict) -> list[dict]:
    if "condition" in target and target["condition"] is not None:
        cond = target["condition"]
        return [
            {
                "attribute": target["attribute"],
                "op": cond["op"],
                "value": float(cond["value"]),
            }
        ]
    if "conditions" in target and target["conditions"] is not None:
        return [
            {
                "attribute": cond["attribute"],
                "op": cond["op"],
                "value": float(cond["value"]),
            }
            for cond in target["conditions"]
        ]
    raise ValueError("Count target must include either `condition` or `conditions`.")


def _scale_attr(attr_name: str, value: float) -> float:
    if attr_name == "revenue":
        return float(torch.log1p(torch.tensor(max(value, 0.0))).item())
    return signed_log1p(value)


def _encode_conditions(conditions: list[dict]) -> torch.Tensor:
    cond_dim = len(COUNT_ATTRS) + len(COUNT_OPS) + 2
    encoded = torch.zeros((1, MAX_CONDITIONS, cond_dim), dtype=torch.float32)
    for index, cond in enumerate(conditions[:MAX_CONDITIONS]):
        attr_index = COUNT_ATTRS.index(cond["attribute"])
        op_index = COUNT_OPS.index(cond["op"])
        encoded[0, index, attr_index] = 1.0
        encoded[0, index, len(COUNT_ATTRS) + op_index] = 1.0
        encoded[0, index, -2] = _scale_attr(cond["attribute"], cond["value"])
        encoded[0, index, -1] = 1.0
    return encoded


def _compute_match_mask(node_values: torch.Tensor, conditions: list[dict]) -> torch.Tensor:
    mask = torch.ones(node_values.size(0), dtype=torch.bool)
    for cond in conditions:
        column_index = COUNT_ATTRS.index(cond["attribute"])
        column = node_values[:, column_index]
        value = cond["value"]
        if cond["op"] == "==":
            passed = column == value
        elif cond["op"] == "<":
            passed = column < value
        elif cond["op"] == "<=":
            passed = column <= value
        elif cond["op"] == ">":
            passed = column > value
        elif cond["op"] == ">=":
            passed = column >= value
        else:
            raise ValueError(f"Unsupported count operator: {cond['op']}")
        mask &= passed
    return mask.to(torch.float32)


def _example_to_data(example: dict) -> Data:
    graph = example["graph"]
    nodes = graph["nodes"]
    edge_index = _build_edge_index(nodes, graph["edges"])
    node_values_rows = []
    scaled_rows = []
    for node in nodes:
        attrs = node["attributes"]
        revenue = float(attrs["revenue"])
        net_profit = float(attrs["net_profit"])
        operating_profit = float(attrs["operating_profit"])
        node_values_rows.append([revenue, net_profit, operating_profit])
        scaled_rows.append(
            [
                _scale_attr("revenue", revenue),
                _scale_attr("net_profit", net_profit),
                _scale_attr("operating_profit", operating_profit),
            ]
        )

    node_values = torch.tensor(node_values_rows, dtype=torch.float32)
    conditions = _normalize_condition(example["task"]["target"])
    match_mask = _compute_match_mask(node_values, conditions)
    return Data(
        x=torch.tensor(scaled_rows, dtype=torch.float32),
        node_values=node_values,
        edge_index=edge_index,
        query_conditions=_encode_conditions(conditions),
        node_match_mask=match_mask,
        y=torch.tensor([float(example["answer"])], dtype=torch.float32),
    )


def count_example_to_data(example: dict) -> Data:
    return _example_to_data(example)


def load_count_v2_splits(
    jsonl_path: str,
    max_samples: int | None = None,
) -> tuple[list[Data], list[Data], list[Data], CountV2SplitSummary]:
    rows = read_jsonl(jsonl_path, max_samples=max_samples)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows or not val_rows or not test_rows:
        raise ValueError("count_600.jsonl must contain train/val/test splits.")

    train_graphs = [_example_to_data(row) for row in train_rows]
    val_graphs = [_example_to_data(row) for row in val_rows]
    test_graphs = [_example_to_data(row) for row in test_rows]
    summary = CountV2SplitSummary(
        train_size=len(train_graphs),
        val_size=len(val_graphs),
        test_size=len(test_graphs),
    )
    return train_graphs, val_graphs, test_graphs, summary
