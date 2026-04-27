"""Data pipeline for PredictV5.

PredictV5 treats revenue prediction as a scale-stable time-series problem:
predict target log-growth from target history and peer growth statistics.
It intentionally avoids company-id embeddings and graph neural layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from realdata_experts.common import extract_year_from_graph_id, read_jsonl, signed_log1p


@dataclass(frozen=True)
class PredictV5SplitSummary:
    train_size: int
    val_size: int
    test_size: int
    years: list[int]
    history_len: int
    static_dim: int


@dataclass(frozen=True)
class PredictV5Resources:
    year_vocab: dict[int, int]
    history_map: dict[tuple[str, int], tuple[float, float]]


class PredictV5Dataset(Dataset):
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def _build_year_vocab(rows: list[dict]) -> dict[int, int]:
    years = sorted({extract_year_from_graph_id(row["graph"]["graph_id"]) for row in rows})
    return {year: index for index, year in enumerate(years)}


def _build_history_map(rows: list[dict]) -> dict[tuple[str, int], tuple[float, float]]:
    history_map: dict[tuple[str, int], tuple[float, float]] = {}
    for row in rows:
        year = extract_year_from_graph_id(row["graph"]["graph_id"])
        if year is None:
            continue
        target = row["target"]
        node_id = target["node_id"]
        node = next(node for node in row["graph"]["nodes"] if node["id"] == node_id)
        attrs = node["attributes"]
        history_map[(target["node_name"], year)] = (
            float(attrs["revenue"]),
            float(attrs["employees"]),
        )
    return history_map


def _log_growth(curr: float | None, prev: float | None) -> float:
    if curr is None or prev is None or curr <= 0 or prev <= 0:
        return 0.0
    return signed_log1p(curr) - signed_log1p(prev)


def _history_sequence(
    company_name: str,
    current_year: int,
    history_map: dict[tuple[str, int], tuple[float, float]],
    history_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[float]] = []
    for year in sorted(year for (name, year) in history_map if name == company_name and year < current_year):
        revenue, employees = history_map[(company_name, year)]
        prev = history_map.get((company_name, year - 1))
        prev_revenue = prev[0] if prev is not None else None
        prev_employees = prev[1] if prev is not None else None
        rows.append(
            [
                signed_log1p(revenue),
                signed_log1p(employees),
                _log_growth(revenue, prev_revenue),
                _log_growth(employees, prev_employees),
                1.0,
            ]
        )
    rows = rows[-history_len:]

    seq = torch.zeros((history_len, 5), dtype=torch.float32)
    mask = torch.zeros(history_len, dtype=torch.float32)
    start = history_len - len(rows)
    for offset, values in enumerate(rows):
        seq[start + offset] = torch.tensor(values, dtype=torch.float32)
        mask[start + offset] = 1.0
    return seq, mask


def _peer_growth_features(
    nodes: list[dict],
    target_node_id: str,
    year: int,
    history_map: dict[tuple[str, int], tuple[float, float]],
) -> dict[str, float]:
    growths: list[float] = []
    employee_growths: list[float] = []
    current_logs: list[float] = []
    previous_logs: list[float] = []

    for node in nodes:
        if node["id"] == target_node_id:
            continue
        name = node["name"]
        attrs = node["attributes"]
        curr_revenue = float(attrs["revenue"])
        curr_employees = float(attrs["employees"])
        prev = history_map.get((name, year - 1))
        if prev is None:
            continue
        prev_revenue, prev_employees = prev
        growths.append(_log_growth(curr_revenue, prev_revenue))
        employee_growths.append(_log_growth(curr_employees, prev_employees))
        current_logs.append(signed_log1p(curr_revenue))
        previous_logs.append(signed_log1p(prev_revenue))

    if not growths:
        return {
            "peer_growth_avg": 0.0,
            "peer_growth_std": 0.0,
            "peer_growth_min": 0.0,
            "peer_growth_max": 0.0,
            "peer_employee_growth_avg": 0.0,
            "peer_current_log_avg": 0.0,
            "peer_previous_log_avg": 0.0,
            "n_peers": 0.0,
        }

    growth_tensor = torch.tensor(growths, dtype=torch.float32)
    employee_tensor = torch.tensor(employee_growths, dtype=torch.float32)
    return {
        "peer_growth_avg": float(growth_tensor.mean().item()),
        "peer_growth_std": float(growth_tensor.std(unbiased=False).item()) if len(growths) > 1 else 0.0,
        "peer_growth_min": float(growth_tensor.min().item()),
        "peer_growth_max": float(growth_tensor.max().item()),
        "peer_employee_growth_avg": float(employee_tensor.mean().item()) if employee_growths else 0.0,
        "peer_current_log_avg": float(torch.tensor(current_logs, dtype=torch.float32).mean().item()),
        "peer_previous_log_avg": float(torch.tensor(previous_logs, dtype=torch.float32).mean().item()),
        "n_peers": float(len(growths)),
    }


def _example_to_sample(
    example: dict,
    year_vocab: dict[int, int],
    history_map: dict[tuple[str, int], tuple[float, float]],
    history_len: int,
) -> dict[str, torch.Tensor]:
    graph = example["graph"]
    year = extract_year_from_graph_id(graph["graph_id"])
    if year is None:
        raise ValueError(f"Missing year in graph id: {graph['graph_id']}")

    target = example["target"]
    target_node_id = target["node_id"]
    target_name = target["node_name"]
    target_node = next(node for node in graph["nodes"] if node["id"] == target_node_id)
    target_attrs = target_node["attributes"]
    revenue = float(target_attrs["revenue"])
    employees = float(target_attrs["employees"])
    prev = history_map.get((target_name, year - 1))
    prev_revenue = float(prev[0]) if prev is not None else 0.0
    prev_employees = float(prev[1]) if prev is not None else 0.0
    has_prev = 1.0 if prev_revenue > 0 and prev_employees > 0 else 0.0

    history_seq, history_mask = _history_sequence(target_name, year, history_map, history_len)
    peer = _peer_growth_features(graph["nodes"], target_node_id, year, history_map)
    min_year = min(year_vocab)
    max_year = max(year_vocab)
    year_value = (year - min_year) / (max_year - min_year) if max_year > min_year else 0.0
    last_history_growth = float(history_seq[-1, 2].item()) if history_mask[-1] > 0 else 0.0
    baseline_growth = (
        0.7 * peer["peer_growth_avg"] + 0.3 * last_history_growth
        if peer["n_peers"] > 0 and history_mask.sum() > 0
        else peer["peer_growth_avg"] if peer["n_peers"] > 0
        else last_history_growth
    )

    static_features = torch.tensor(
        [
            year_value,
            year_value**2,
            signed_log1p(prev_revenue) if prev_revenue > 0 else 0.0,
            signed_log1p(prev_employees) if prev_employees > 0 else 0.0,
            has_prev,
            peer["peer_growth_avg"],
            peer["peer_growth_std"],
            peer["peer_growth_min"],
            peer["peer_growth_max"],
            peer["peer_employee_growth_avg"],
            peer["peer_current_log_avg"],
            peer["peer_previous_log_avg"],
            min(peer["n_peers"] / 100.0, 1.0),
            float(history_mask.sum().item()) / max(history_len, 1),
            signed_log1p(employees),
        ],
        dtype=torch.float32,
    )

    target_growth = _log_growth(revenue, prev_revenue)
    return {
        "history_seq": history_seq,
        "history_mask": history_mask,
        "static_features": static_features,
        "baseline_growth": torch.tensor([baseline_growth], dtype=torch.float32),
        "y": torch.tensor([target_growth], dtype=torch.float32),
        "target_revenue": torch.tensor([revenue], dtype=torch.float32),
        "prev_revenue": torch.tensor([prev_revenue], dtype=torch.float32),
        "has_prev": torch.tensor([has_prev], dtype=torch.float32),
    }


def build_predict_v5_resources(rows: list[dict]) -> PredictV5Resources:
    return PredictV5Resources(
        year_vocab=_build_year_vocab(rows),
        history_map=_build_history_map(rows),
    )


def predict_example_to_sample(
    example: dict,
    resources: PredictV5Resources,
    history_len: int = 5,
) -> dict[str, torch.Tensor]:
    return _example_to_sample(example, resources.year_vocab, resources.history_map, history_len)


def load_predict_v5_splits(
    jsonl_path: str,
    max_samples: int | None = None,
    history_len: int = 5,
) -> tuple[PredictV5Dataset, PredictV5Dataset, PredictV5Dataset, PredictV5SplitSummary]:
    rows = read_jsonl(jsonl_path, max_samples=max_samples)
    resources = build_predict_v5_resources(rows)

    train_rows, val_rows, test_rows = [], [], []
    for row in rows:
        year = extract_year_from_graph_id(row["graph"]["graph_id"])
        if year is None:
            raise ValueError(f"Missing year in graph id: {row['graph']['graph_id']}")
        if year <= 2022:
            train_rows.append(row)
        elif year == 2023:
            val_rows.append(row)
        else:
            test_rows.append(row)

    def convert(items: list[dict]) -> PredictV5Dataset:
        return PredictV5Dataset(
            [_example_to_sample(item, resources.year_vocab, resources.history_map, history_len) for item in items]
        )

    train_ds = convert(train_rows)
    val_ds = convert(val_rows)
    test_ds = convert(test_rows)
    summary = PredictV5SplitSummary(
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
        years=sorted(resources.year_vocab),
        history_len=history_len,
        static_dim=15,
    )
    return train_ds, val_ds, test_ds, summary
