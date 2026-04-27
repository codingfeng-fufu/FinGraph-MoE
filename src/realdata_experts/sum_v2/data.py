from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from realdata_experts.common import deterministic_split, read_jsonl, signed_log1p


SUM_ATTRS = ("revenue", "operating_profit", "net_profit")


@dataclass(frozen=True)
class SumV2SplitSummary:
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


def _query_attr_vector(target_attr: str) -> torch.Tensor:
    vector = torch.zeros((1, len(SUM_ATTRS)), dtype=torch.float32)
    vector[0, SUM_ATTRS.index(target_attr)] = 1.0
    return vector


def _example_to_data(example: dict) -> Data:
    graph = example["graph"]
    nodes = graph["nodes"]
    edge_index = _build_edge_index(nodes, graph["edges"])

    raw_values = []
    scaled_values = []
    for node in nodes:
        attrs = node["attributes"]
        row_raw = [float(attrs.get(name, 0.0)) for name in SUM_ATTRS]
        raw_values.append(row_raw)
        scaled_values.append(
            [
                signed_log1p(attrs.get("revenue", 0.0)),
                signed_log1p(attrs.get("operating_profit", 0.0)),
                signed_log1p(attrs.get("net_profit", 0.0)),
            ]
        )

    target_attr = example["task"]["target"]["attribute"]
    return Data(
        x=torch.tensor(scaled_values, dtype=torch.float32),
        node_values=torch.tensor(raw_values, dtype=torch.float32),
        edge_index=edge_index,
        query_attr=_query_attr_vector(target_attr),
        y=torch.tensor([float(example["answer"])], dtype=torch.float32),
    )


def sum_example_to_data(example: dict) -> Data:
    return _example_to_data(example)


def load_sum_v2_splits(
    jsonl_path: str,
    seed: int = 7,
    max_samples: int | None = None,
) -> tuple[list[Data], list[Data], list[Data], SumV2SplitSummary]:
    rows = read_jsonl(jsonl_path, max_samples=max_samples)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]

    if not train_rows or not test_rows:
        raise ValueError("sum_dataset2.jsonl must contain both train and test splits.")

    if len(train_rows) >= 3:
        train_rows, val_rows, held_out_rows = deterministic_split(
            train_rows,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=seed,
        )
        test_rows = [*test_rows, *held_out_rows]
    elif len(train_rows) == 2:
        val_rows = [train_rows[1]]
        train_rows = [train_rows[0]]
    else:
        raise ValueError("sum_dataset2.jsonl requires at least 2 train samples.")

    train_graphs = [_example_to_data(row) for row in train_rows]
    val_graphs = [_example_to_data(row) for row in val_rows]
    test_graphs = [_example_to_data(row) for row in test_rows]
    summary = SumV2SplitSummary(
        train_size=len(train_graphs),
        val_size=len(val_graphs),
        test_size=len(test_graphs),
    )
    return train_graphs, val_graphs, test_graphs, summary
