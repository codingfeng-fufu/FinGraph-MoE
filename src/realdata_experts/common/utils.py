from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, TypeVar

import torch


T = TypeVar("T")


def read_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def deterministic_split(
    items: Iterable[T],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 7,
) -> tuple[list[T], list[T], list[T]]:
    items_list = list(items)
    if len(items_list) < 3:
        raise ValueError("At least 3 items are required for a train/val/test split.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    rng = random.Random(seed)
    indices = list(range(len(items_list)))
    rng.shuffle(indices)

    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)

    train_items = [items_list[index] for index in indices[:train_end]]
    val_items = [items_list[index] for index in indices[train_end:val_end]]
    test_items = [items_list[index] for index in indices[val_end:]]
    return train_items, val_items, test_items


def signed_log1p(value: float | int) -> float:
    scalar = float(value)
    return float(torch.sign(torch.tensor(scalar)) * torch.log1p(torch.tensor(abs(scalar))))


def extract_year_from_graph_id(graph_id: str) -> int | None:
    match = re.search(r"(20\d{2})", graph_id)
    if match is None:
        return None
    return int(match.group(1))

