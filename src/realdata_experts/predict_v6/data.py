"""Data pipeline for PredictV6.

PredictV6 combines target-company history with Kumo-style in-context peer
examples. Each peer token is a same-year company whose previous-year record is
known, so the model can learn from peer growth examples inside the document.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from realdata_experts.common import extract_year_from_graph_id, read_jsonl, signed_log1p


@dataclass(frozen=True)
class PredictV6SplitSummary:
    train_size: int
    val_size: int
    test_size: int
    years: list[int]
    history_len: int
    static_dim: int
    context_dim: int
    max_peers: int


@dataclass(frozen=True)
class PredictV6Resources:
    year_vocab: dict[int, int]
    history_map: dict[tuple[str, int], tuple[float, float]]


class PredictV6Dataset(Dataset):
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
        node = next(node for node in row["graph"]["nodes"] if node["id"] == target["node_id"])
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


def _peer_context_and_stats(
    nodes: list[dict],
    target_node_id: str,
    year: int,
    history_map: dict[tuple[str, int], tuple[float, float]],
    max_peers: int,
    target_prev_revenue: float,
    target_prev_employees: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    peer_rows: list[tuple[float, list[float]]] = []
    growths: list[float] = []
    employee_growths: list[float] = []
    current_logs: list[float] = []
    previous_logs: list[float] = []

    target_prev_rev_log = signed_log1p(target_prev_revenue) if target_prev_revenue > 0 else 0.0
    target_prev_emp_log = signed_log1p(target_prev_employees) if target_prev_employees > 0 else 0.0

    for node in nodes:
        if node["id"] == target_node_id:
            continue
        attrs = node["attributes"]
        prev = history_map.get((node["name"], year - 1))
        if prev is None:
            continue

        curr_revenue = float(attrs["revenue"])
        curr_employees = float(attrs["employees"])
        prev_revenue, prev_employees = prev
        revenue_growth = _log_growth(curr_revenue, prev_revenue)
        employee_growth = _log_growth(curr_employees, prev_employees)
        prev_rev_log = signed_log1p(prev_revenue)
        curr_rev_log = signed_log1p(curr_revenue)
        prev_emp_log = signed_log1p(prev_employees)
        curr_emp_log = signed_log1p(curr_employees)
        similarity_distance = abs(prev_rev_log - target_prev_rev_log) if target_prev_revenue > 0 else 0.0
        peer_rows.append(
            (
                similarity_distance,
                [
                    revenue_growth,
                    employee_growth,
                    prev_rev_log,
                    curr_rev_log,
                    prev_emp_log,
                    curr_emp_log,
                    prev_rev_log - target_prev_rev_log,
                    prev_emp_log - target_prev_emp_log,
                    1.0 / (1.0 + similarity_distance),
                    1.0,
                ],
            )
        )
        growths.append(revenue_growth)
        employee_growths.append(employee_growth)
        current_logs.append(curr_rev_log)
        previous_logs.append(prev_rev_log)

    peer_rows.sort(key=lambda item: item[0])
    selected_rows = [row for _, row in peer_rows[:max_peers]]
    context_seq = torch.zeros((max_peers, 10), dtype=torch.float32)
    context_mask = torch.zeros(max_peers, dtype=torch.float32)
    for index, row in enumerate(selected_rows):
        context_seq[index] = torch.tensor(row, dtype=torch.float32)
        context_mask[index] = 1.0

    if not growths:
        return context_seq, context_mask, {
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
    return context_seq, context_mask, {
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
    max_peers: int,
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
    context_seq, context_mask, peer = _peer_context_and_stats(
        graph["nodes"],
        target_node_id,
        year,
        history_map,
        max_peers,
        prev_revenue,
        prev_employees,
    )
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
        "context_seq": context_seq,
        "context_mask": context_mask,
        "static_features": static_features,
        "baseline_growth": torch.tensor([baseline_growth], dtype=torch.float32),
        "y": torch.tensor([target_growth], dtype=torch.float32),
        "target_revenue": torch.tensor([revenue], dtype=torch.float32),
        "prev_revenue": torch.tensor([prev_revenue], dtype=torch.float32),
        "has_prev": torch.tensor([has_prev], dtype=torch.float32),
    }


def build_predict_v6_resources(rows: list[dict]) -> PredictV6Resources:
    return PredictV6Resources(
        year_vocab=_build_year_vocab(rows),
        history_map=_build_history_map(rows),
    )


def predict_example_to_sample(
    example: dict,
    resources: PredictV6Resources,
    history_len: int = 5,
    max_peers: int = 32,
) -> dict[str, torch.Tensor]:
    return _example_to_sample(example, resources.year_vocab, resources.history_map, history_len, max_peers)


def load_predict_v6_splits(
    jsonl_path: str,
    max_samples: int | None = None,
    history_len: int = 5,
    max_peers: int = 32,
) -> tuple[PredictV6Dataset, PredictV6Dataset, PredictV6Dataset, PredictV6SplitSummary]:
    rows = read_jsonl(jsonl_path, max_samples=max_samples)
    resources = build_predict_v6_resources(rows)

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

    def convert(items: list[dict]) -> PredictV6Dataset:
        return PredictV6Dataset(
            [
                _example_to_sample(item, resources.year_vocab, resources.history_map, history_len, max_peers)
                for item in items
            ]
        )

    train_ds = convert(train_rows)
    val_ds = convert(val_rows)
    test_ds = convert(test_rows)
    summary = PredictV6SplitSummary(
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
        years=sorted(resources.year_vocab),
        history_len=history_len,
        static_dim=15,
        context_dim=10,
        max_peers=max_peers,
    )
    return train_ds, val_ds, test_ds, summary
