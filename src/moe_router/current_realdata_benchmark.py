from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from realdata_experts.common import deterministic_split, extract_year_from_graph_id, read_jsonl, signed_log1p
from realdata_experts.count_v2.data import count_example_to_data
from realdata_experts.count_v2.model import CountV2Model
from realdata_experts.predict_tabular.artifact import (
    load_artifact as load_predict_tabular_artifact,
    predict_growth as predict_tabular_growth,
)
from realdata_experts.predict_v6.data import build_predict_v6_resources, predict_example_to_sample
from realdata_experts.sum_v2.data import sum_example_to_data
from realdata_experts.sum_v2.model import SumV2Model

from .paths import REPO_ROOT
from .router import MoERouter


CURRENT_BENCHMARK_VERSION = "current_realdata_benchmark_v1"
DEFAULT_PERTURBATIONS = (
    "clean",
    "numeric_noise_1pct",
    "numeric_noise_5pct",
    "drop_edges_50pct",
    "predict_drop_peers_50pct",
)
NUMERIC_ATTRS = ("revenue", "operating_profit", "net_profit", "employees")
SUM_ATTRS = ("revenue", "operating_profit", "net_profit")
COUNT_ATTRS = ("revenue", "net_profit", "operating_profit")
COUNT_OPS = ("==", "<", "<=", ">", ">=")


@dataclass(frozen=True)
class CurrentArtifacts:
    device: torch.device
    sum_model: SumV2Model
    count_model: CountV2Model
    predict_artifact: dict[str, Any]
    predict_resources: Any


def _artifact_paths() -> dict[str, str]:
    return {
        "sum": str(REPO_ROOT / "runs/realdata_experts/sum_v2/best_model.pt"),
        "count": str(REPO_ROOT / "runs/realdata_experts/count_v2/best_model.pt"),
        "predict_tabular": str(
            REPO_ROOT
            / "runs/realdata_experts/predict_tabular_elastic_net_flat_residual"
            / "artifact.joblib"
        ),
    }


def _load_current_artifacts(device: torch.device, predict_rows: list[dict]) -> CurrentArtifacts:
    paths = _artifact_paths()

    sum_payload = torch.load(paths["sum"], map_location=device)
    sum_model = SumV2Model(**sum_payload["model_config"]).to(device)
    sum_model.load_state_dict(sum_payload["model_state_dict"])
    sum_model.eval()

    count_payload = torch.load(paths["count"], map_location=device)
    count_model = CountV2Model(**count_payload["model_config"]).to(device)
    count_model.load_state_dict(count_payload["model_state_dict"])
    count_model.eval()

    predict_artifact = load_predict_tabular_artifact(paths["predict_tabular"])
    predict_resources = build_predict_v6_resources(predict_rows)

    return CurrentArtifacts(
        device=device,
        sum_model=sum_model,
        count_model=count_model,
        predict_artifact=predict_artifact,
        predict_resources=predict_resources,
    )


def collect_current_benchmark_rows(
    limit_per_task: int | None = None,
    min_examples_per_task: int = 200,
) -> dict[str, list[dict]]:
    sum_rows = read_jsonl(REPO_ROOT / "data/realdata_inputs/sum_dataset2.jsonl")
    sum_train_rows = [row for row in sum_rows if row["split"] == "train"]
    sum_test_rows = [row for row in sum_rows if row["split"] == "test"]
    if len(sum_train_rows) >= 3:
        _, _, sum_heldout_rows = deterministic_split(
            sum_train_rows,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=7,
        )
        sum_eval_rows = [*sum_test_rows, *sum_heldout_rows]
    else:
        sum_eval_rows = sum_test_rows

    count_rows = read_jsonl(REPO_ROOT / "data/realdata_inputs/count_600.jsonl")
    count_eval_rows = [row for row in count_rows if row["split"] == "test"]

    predict_rows = read_jsonl(REPO_ROOT / "data/realdata_inputs/prediction_dataset.jsonl")
    predict_eval_rows = [
        row
        for row in predict_rows
        if (extract_year_from_graph_id(row["graph"]["graph_id"]) or 0) >= 2024
    ]

    sum_eval_rows = _ensure_min_examples(
        "sum",
        sum_eval_rows,
        min_examples_per_task=min_examples_per_task,
        seed=211,
    )
    count_eval_rows = _ensure_min_examples(
        "count",
        count_eval_rows,
        min_examples_per_task=min_examples_per_task,
        seed=223,
    )
    predict_eval_rows = _ensure_min_examples(
        "predict",
        predict_eval_rows,
        min_examples_per_task=min_examples_per_task,
        seed=227,
    )

    if limit_per_task is not None:
        limit = max(limit_per_task, 0)
        sum_eval_rows = sum_eval_rows[:limit]
        count_eval_rows = count_eval_rows[:limit]
        predict_eval_rows = predict_eval_rows[:limit]

    return {
        "sum": sum_eval_rows,
        "count": count_eval_rows,
        "predict": predict_eval_rows,
        "_predict_all": predict_rows,
    }


def _ensure_min_examples(
    task: str,
    rows: list[dict],
    min_examples_per_task: int,
    seed: int,
) -> list[dict]:
    if min_examples_per_task <= 0 or len(rows) >= min_examples_per_task:
        return rows
    missing = min_examples_per_task - len(rows)
    if task == "sum":
        extra = _generate_sum_benchmark_rows(rows, missing, seed)
    elif task == "count":
        extra = _generate_count_benchmark_rows(rows, missing, seed)
    elif task == "predict":
        extra = _generate_predict_benchmark_rows(rows, missing, seed)
    else:
        raise ValueError(f"Unsupported task for benchmark generation: {task}")
    return [*rows, *extra]


def _generate_sum_benchmark_rows(base_rows: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    output: list[dict] = []
    if not base_rows:
        raise ValueError("Cannot generate SUM benchmark rows without base rows.")
    for index in range(count):
        base = copy.deepcopy(base_rows[index % len(base_rows)])
        nodes = list(base["graph"]["nodes"])
        rng.shuffle(nodes)
        subset_size = min(len(nodes), 1 + (index % max(min(len(nodes), 8), 1)))
        selected_nodes = nodes[:subset_size]
        attr = SUM_ATTRS[index % len(SUM_ATTRS)]
        industry = selected_nodes[0].get("attributes", {}).get("industry", "UNKNOWN")
        answer = float(sum(float(node["attributes"].get(attr, 0.0) or 0.0) for node in selected_nodes))
        node_ids = {node["id"] for node in selected_nodes}
        edges = [
            edge
            for edge in base["graph"].get("edges", [])
            if edge.get("source") in node_ids and edge.get("target") in node_ids
        ]
        output.append(
            {
                **base,
                "sample_id": f"bench_sum_synth_{index:04d}",
                "split": "benchmark_synthetic",
                "benchmark_source": "synthetic_extra",
                "graph": {
                    **base["graph"],
                    "graph_id": f"bench_sum_synth_graph_{index:04d}",
                    "nodes": selected_nodes,
                    "edges": edges,
                },
                "task": {
                    "kind": "sum",
                    "query_text": f"请计算 {industry} 行业公司的 {attr} 总和",
                    "target": {
                        "node_type": "company",
                        "node_id": None,
                        "node_name": None,
                        "attribute": attr,
                    },
                },
                "answer": answer,
            }
        )
    return output


def _generate_count_benchmark_rows(base_rows: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    output: list[dict] = []
    if not base_rows:
        raise ValueError("Cannot generate COUNT benchmark rows without base rows.")
    for index in range(count):
        base = copy.deepcopy(base_rows[index % len(base_rows)])
        attr = COUNT_ATTRS[index % len(COUNT_ATTRS)]
        op = COUNT_OPS[(index // len(COUNT_ATTRS)) % len(COUNT_OPS)]
        nodes = base["graph"]["nodes"]
        values = sorted(float(node["attributes"][attr]) for node in nodes if attr in node["attributes"])
        if not values:
            raise ValueError(f"Cannot generate COUNT benchmark without values for {attr}.")
        if op == "==":
            value = values[(index * 17) % len(values)]
        else:
            quantile = [0.2, 0.35, 0.5, 0.65, 0.8][index % 5]
            value = values[min(int(len(values) * quantile), len(values) - 1)]
        if index % 4 == 3:
            second_attr = COUNT_ATTRS[(index + 1) % len(COUNT_ATTRS)]
            second_values = sorted(float(node["attributes"][second_attr]) for node in nodes if second_attr in node["attributes"])
            second_value = second_values[min(int(len(second_values) * 0.55), len(second_values) - 1)]
            conditions = [
                {"attribute": attr, "op": op, "value": float(value)},
                {"attribute": second_attr, "op": ">=", "value": float(second_value)},
            ]
            target = {"node_type": "company", "conditions": conditions}
            query_text = (
                f"请统计 {attr} {op} {_format_number(value)} 且 "
                f"{second_attr} >= {_format_number(second_value)} 的公司数量"
            )
        else:
            conditions = [{"attribute": attr, "op": op, "value": float(value)}]
            target = {
                "node_type": "company",
                "attribute": attr,
                "condition": {"op": op, "value": float(value)},
            }
            query_text = f"请统计 {attr} {op} {_format_number(value)} 的公司数量"
        answer = float(_count_matches(nodes, conditions))
        output.append(
            {
                **base,
                "sample_id": f"bench_count_synth_{index:04d}",
                "split": "benchmark_synthetic",
                "benchmark_source": "synthetic_extra",
                "task": {
                    "kind": "count",
                    "query_text": query_text,
                    "target": target,
                },
                "answer": answer,
            }
        )
    return output


def _generate_predict_benchmark_rows(base_rows: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    output: list[dict] = []
    if not base_rows:
        raise ValueError("Cannot generate PREDICT benchmark rows without base rows.")
    for index in range(count):
        row = copy.deepcopy(base_rows[index % len(base_rows)])
        factor = 1.0 + rng.uniform(-0.02, 0.02)
        for node in row["graph"]["nodes"]:
            attrs = node.get("attributes", {})
            if attrs.get("revenue") is not None:
                attrs["revenue"] = max(float(attrs["revenue"]) * factor, 0.0)
            if attrs.get("employees") is not None:
                attrs["employees"] = max(float(attrs["employees"]) * (1.0 + rng.uniform(-0.01, 0.01)), 0.0)
        target_node_id = row["target"]["node_id"]
        target_node = next(node for node in row["graph"]["nodes"] if node["id"] == target_node_id)
        row["sample_id"] = f"bench_predict_synth_{index:04d}"
        row["split"] = "benchmark_synthetic"
        row["benchmark_source"] = "synthetic_extra"
        row["answer"] = float(target_node["attributes"]["revenue"])
        output.append(row)
    return output


def _count_matches(nodes: list[dict], conditions: list[dict[str, Any]]) -> int:
    count = 0
    for node in nodes:
        attrs = node["attributes"]
        matched = True
        for condition in conditions:
            current = float(attrs[condition["attribute"]])
            value = float(condition["value"])
            op = condition["op"]
            if op == "==":
                passed = current == value
            elif op == "<":
                passed = current < value
            elif op == "<=":
                passed = current <= value
            elif op == ">":
                passed = current > value
            elif op == ">=":
                passed = current >= value
            else:
                raise ValueError(f"Unsupported count op: {op}")
            matched = matched and passed
        count += int(matched)
    return count


def _format_number(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _predict_sum(model: SumV2Model, example: dict, device: torch.device) -> tuple[float, dict[str, Any]]:
    graph = sum_example_to_data(example)
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        prediction, node_gates = model(batch.x, batch.node_values, batch.batch, batch.query_attr)
    return float(prediction.item()), {
        "node_gate_mean": float(node_gates.mean().cpu().item()) if node_gates.numel() else 0.0,
        "node_gate_min": float(node_gates.min().cpu().item()) if node_gates.numel() else 0.0,
        "node_gate_max": float(node_gates.max().cpu().item()) if node_gates.numel() else 0.0,
    }


def _predict_count(model: CountV2Model, example: dict, device: torch.device) -> tuple[float, dict[str, Any]]:
    graph = count_example_to_data(example)
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        prediction, node_scores = model(batch.x, batch.node_values, batch.batch, batch.query_conditions)
    return float(prediction.item()), {
        "rounded_prediction": float(round(float(prediction.item()))),
        "node_score_mean": float(node_scores.mean().cpu().item()) if node_scores.numel() else 0.0,
        "node_score_min": float(node_scores.min().cpu().item()) if node_scores.numel() else 0.0,
        "node_score_max": float(node_scores.max().cpu().item()) if node_scores.numel() else 0.0,
    }


def _predict_predict_tabular(
    artifact: dict[str, Any],
    resources: Any,
    example: dict,
) -> tuple[float, dict[str, Any]]:
    sample = predict_example_to_sample(
        example,
        resources,
        history_len=int(artifact.get("history_len", 5)),
        max_peers=int(artifact.get("max_peers", 32)),
    )
    growth = predict_tabular_growth(sample, artifact)
    baseline_growth = float(sample["baseline_growth"].view(-1)[0].item())
    prev_revenue = float(sample["prev_revenue"].view(-1)[0].item())
    prediction = _decode_growth_to_revenue(growth, prev_revenue)
    baseline_prediction = _decode_growth_to_revenue(baseline_growth, prev_revenue)
    static = sample["static_features"].view(-1)
    history_count = int(round(float(static[13].item()) * max(sample["history_mask"].numel(), 1)))
    n_peers = int(round(float(static[12].item()) * 100.0))
    return prediction, {
        "growth": float(growth),
        "baseline_growth": baseline_growth,
        "baseline_prediction": baseline_prediction,
        "prev_revenue": prev_revenue,
        "history_count": history_count,
        "n_peers": n_peers,
        "model_name": artifact.get("model_name", "elastic_net"),
        "feature_set": artifact.get("feature_set", "flat"),
        "target_mode": artifact.get("target_mode", "residual"),
    }


def _decode_growth_to_revenue(growth: float, prev_revenue: float) -> float:
    if prev_revenue <= 0:
        return 0.0
    value = math.expm1(signed_log1p(prev_revenue) + growth)
    if not math.isfinite(value):
        return 0.0
    return float(max(value, 0.0))


def run_current_realdata_benchmark(
    router_dir: str | Path = REPO_ROOT / "data/moe_router_realdata_v1",
    output_file: str | Path | None = REPO_ROOT / "runs/realdata_experts/current_realdata_benchmark.json",
    markdown_file: str | Path | None = REPO_ROOT / "runs/realdata_experts/current_realdata_benchmark.md",
    perturbations: tuple[str, ...] = DEFAULT_PERTURBATIONS,
    limit_per_task: int | None = None,
    min_examples_per_task: int = 200,
    skip_router: bool = False,
    include_records: bool = True,
) -> dict[str, Any]:
    rows = collect_current_benchmark_rows(
        limit_per_task=limit_per_task,
        min_examples_per_task=min_examples_per_task,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = _load_current_artifacts(device, rows["_predict_all"])

    records: list[dict[str, Any]] = []
    for perturbation in perturbations:
        for task in ("sum", "count", "predict"):
            for row in rows[task]:
                if perturbation == "predict_drop_peers_50pct" and task != "predict":
                    continue
                example = perturb_example(row, task, perturbation)
                prediction, details = _predict_task(task, example, artifacts)
                answer = float(row["answer"])
                records.append(
                    {
                        "task": task,
                        "perturbation": perturbation,
                        "sample_id": str(row.get("sample_id", "")),
                        "query": row["task"]["query_text"],
                        "answer": answer,
                        "prediction": prediction,
                        "abs_error": abs(prediction - answer),
                        "rel_error": abs(prediction - answer) / max(abs(answer), 1e-6),
                        "benchmark_source": row.get("benchmark_source", "heldout"),
                        "details": details,
                        "slices": _slice_tags(task, row, details),
                    }
                )

    router_summary = None if skip_router else _run_router_benchmark(router_dir, rows)
    summary = {
        "benchmark_version": CURRENT_BENCHMARK_VERSION,
        "router_dir": str(Path(router_dir).resolve()),
        "device": str(device),
        "artifacts": _artifact_paths(),
        "dataset_sizes": {
            "sum": len(rows["sum"]),
            "count": len(rows["count"]),
            "predict": len(rows["predict"]),
        },
        "dataset_sources": {
            task: dict(Counter(row.get("benchmark_source", "heldout") for row in rows[task]))
            for task in ("sum", "count", "predict")
        },
        "min_examples_per_task": min_examples_per_task,
        "perturbations": list(perturbations),
        "router": router_summary,
        "experts": _summarize_expert_records(records),
        "slices": _summarize_slices([record for record in records if record["perturbation"] == "clean"]),
        "records": records if include_records else [],
    }

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if markdown_file is not None:
        markdown_path = Path(markdown_file)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown_report(summary), encoding="utf-8")

    return summary


def _predict_task(
    task: str,
    example: dict,
    artifacts: CurrentArtifacts,
) -> tuple[float, dict[str, Any]]:
    if task == "sum":
        return _predict_sum(artifacts.sum_model, example, artifacts.device)
    if task == "count":
        return _predict_count(artifacts.count_model, example, artifacts.device)
    if task == "predict":
        return _predict_predict_tabular(
            artifacts.predict_artifact,
            artifacts.predict_resources,
            example,
        )
    raise ValueError(f"Unsupported task: {task}")


def _run_router_benchmark(router_dir: str | Path, rows: dict[str, list[dict]]) -> dict[str, Any]:
    router = MoERouter.load(router_dir)
    route_records: list[dict[str, Any]] = []
    for task in ("sum", "count", "predict"):
        for row in rows[task]:
            decision = router.route(row["task"]["query_text"], top_k=7)
            route_records.append(
                {
                    "task": task,
                    "sample_id": str(row.get("sample_id", "")),
                    "benchmark_source": row.get("benchmark_source", "heldout"),
                    "selected_expert": decision.selected_expert,
                    "correct": decision.selected_expert == task,
                    "confidence": decision.confidence,
                    "top_neighbor_expert": decision.neighbors[0].expert if decision.neighbors else None,
                    "num_rewrites": len(decision.rewrites),
                }
            )

    overall_correct = sum(1 for record in route_records if record["correct"])
    by_task = {}
    for task in ("sum", "count", "predict"):
        task_records = [record for record in route_records if record["task"] == task]
        by_task[task] = {
            "num_examples": len(task_records),
            "accuracy": sum(1 for record in task_records if record["correct"]) / max(len(task_records), 1),
            "avg_confidence": _mean([float(record["confidence"]) for record in task_records]),
        }
    confusion = Counter((record["task"], record["selected_expert"]) for record in route_records)
    return {
        "num_examples": len(route_records),
        "accuracy": overall_correct / max(len(route_records), 1),
        "by_task": by_task,
        "confusion": {
            f"{actual}->{predicted}": count
            for (actual, predicted), count in sorted(confusion.items())
        },
        "records": route_records,
    }


def perturb_example(example: dict, task: str, perturbation: str) -> dict:
    cloned = copy.deepcopy(example)
    if perturbation == "clean":
        return cloned
    if perturbation == "numeric_noise_1pct":
        _apply_numeric_noise(cloned, sigma=0.01, seed_material=f"{task}:{example.get('sample_id')}:noise1")
        return cloned
    if perturbation == "numeric_noise_5pct":
        _apply_numeric_noise(cloned, sigma=0.05, seed_material=f"{task}:{example.get('sample_id')}:noise5")
        return cloned
    if perturbation == "drop_edges_50pct":
        _drop_edges(cloned, keep_probability=0.5, seed_material=f"{task}:{example.get('sample_id')}:drop_edges")
        return cloned
    if perturbation == "predict_drop_peers_50pct":
        if task == "predict":
            _drop_predict_peers(cloned, keep_probability=0.5, seed_material=f"{task}:{example.get('sample_id')}:drop_peers")
        return cloned
    raise ValueError(f"Unsupported perturbation: {perturbation}")


def _apply_numeric_noise(example: dict, sigma: float, seed_material: str) -> None:
    rng = random.Random(_stable_seed(seed_material))
    for node in example.get("graph", {}).get("nodes", []):
        attrs = node.get("attributes", {})
        for attr in NUMERIC_ATTRS:
            value = attrs.get(attr)
            if value is None:
                continue
            noisy = float(value) * (1.0 + rng.gauss(0.0, sigma))
            if attr in {"revenue", "employees"}:
                noisy = max(noisy, 0.0)
            attrs[attr] = noisy


def _drop_edges(example: dict, keep_probability: float, seed_material: str) -> None:
    graph = example.get("graph", {})
    rng = random.Random(_stable_seed(seed_material))
    graph["edges"] = [
        edge
        for edge in graph.get("edges", [])
        if rng.random() <= keep_probability
    ]


def _drop_predict_peers(example: dict, keep_probability: float, seed_material: str) -> None:
    graph = example.get("graph", {})
    target_id = example.get("target", {}).get("node_id")
    rng = random.Random(_stable_seed(seed_material))
    kept_nodes = []
    for node in graph.get("nodes", []):
        if node.get("id") == target_id or rng.random() <= keep_probability:
            kept_nodes.append(node)
    kept_ids = {node["id"] for node in kept_nodes}
    graph["nodes"] = kept_nodes
    graph["edges"] = [
        edge
        for edge in graph.get("edges", [])
        if edge.get("source") in kept_ids and edge.get("target") in kept_ids
    ]


def _stable_seed(material: str) -> int:
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _slice_tags(task: str, row: dict, details: dict[str, Any]) -> dict[str, str]:
    graph = row["graph"]
    node_count = len(graph.get("nodes", []))
    tags = {
        "node_count": _bucket_numeric(node_count, [(3, "1-3"), (8, "4-8")], "9+"),
        "source": row.get("benchmark_source", "heldout"),
    }
    if task == "sum":
        tags["target_attr"] = row["task"]["target"]["attribute"]
        tags["answer_bucket"] = _bucket_numeric(float(row["answer"]), [(1e9, "<1B"), (1e10, "1B-10B")], ">=10B")
    elif task == "count":
        conditions = _count_conditions(row)
        tags["condition_count"] = str(len(conditions))
        if conditions:
            tags["first_attr"] = str(conditions[0]["attribute"])
            tags["first_op"] = str(conditions[0]["op"])
        tags["answer_bucket"] = _bucket_numeric(float(row["answer"]), [(0.5, "0"), (2.5, "1-2")], "3+")
    elif task == "predict":
        year = extract_year_from_graph_id(graph["graph_id"])
        tags["year"] = str(year) if year is not None else "unknown"
        tags["n_peers"] = _bucket_numeric(float(details.get("n_peers", 0)), [(1, "0-1"), (5, "2-5")], "6+")
        tags["history_count"] = _bucket_numeric(float(details.get("history_count", 0)), [(1, "0-1"), (3, "2-3")], "4+")
    return tags


def _count_conditions(row: dict) -> list[dict[str, Any]]:
    target = row["task"]["target"]
    if target.get("condition") is not None:
        cond = dict(target["condition"])
        cond["attribute"] = target["attribute"]
        return [cond]
    return list(target.get("conditions") or [])


def _bucket_numeric(value: float, boundaries: list[tuple[float, str]], fallback: str) -> str:
    for upper, label in boundaries:
        if value <= upper:
            return label
    return fallback


def _summarize_expert_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["perturbation"]][record["task"]].append(record)

    summary: dict[str, Any] = {}
    for perturbation, by_task in grouped.items():
        summary[perturbation] = {}
        for task, task_records in by_task.items():
            summary[perturbation][task] = _task_record_metrics(task, task_records)
        if "predict" in by_task:
            summary[perturbation]["predict_baseline"] = _predict_baseline_metrics(by_task["predict"])
    return summary


def _summarize_slices(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task = record["task"]
        for key, value in record["slices"].items():
            grouped[f"{task}.{key}={value}"].append(record)
    return {
        name: _task_record_metrics(items[0]["task"], items)
        for name, items in sorted(grouped.items())
        if len(items) >= 3
    }


def _task_record_metrics(task: str, records: list[dict[str, Any]]) -> dict[str, float]:
    predictions = [float(record["prediction"]) for record in records]
    answers = [float(record["answer"]) for record in records]
    metrics = _regression_metrics(predictions, answers)
    metrics["num_examples"] = float(len(records))
    if task == "count":
        rounded = [max(round(value), 0) for value in predictions]
        target_rounded = [round(value) for value in answers]
        metrics["rounded_mae"] = _mean([abs(pred - target) for pred, target in zip(rounded, target_rounded, strict=True)])
        metrics["exact_match"] = _mean([1.0 if pred == target else 0.0 for pred, target in zip(rounded, target_rounded, strict=True)])
    return metrics


def _predict_baseline_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    baseline_predictions = [float(record["details"]["baseline_prediction"]) for record in records]
    answers = [float(record["answer"]) for record in records]
    metrics = _regression_metrics(baseline_predictions, answers)
    metrics["num_examples"] = float(len(records))
    return metrics


def _regression_metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    if not predictions:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "nonzero_mape": 0.0,
            "median_abs_error": 0.0,
            "p90_abs_error": 0.0,
            "r2": 0.0,
        }
    pred = torch.tensor(predictions, dtype=torch.float32)
    target = torch.tensor(targets, dtype=torch.float32)
    abs_error = (pred - target).abs()
    mse = F.mse_loss(pred, target).item()
    ss_tot = ((target - target.mean()) ** 2).sum()
    ss_res = ((target - pred) ** 2).sum()
    nonzero_mask = target.abs() > 1e-6
    nonzero_mape = (
        (abs_error[nonzero_mask] / target[nonzero_mask].abs()).mean().item()
        if bool(nonzero_mask.any())
        else 0.0
    )
    return {
        "mae": float(abs_error.mean().item()),
        "rmse": float(mse**0.5),
        "mape": float((abs_error / target.abs().clamp_min(1e-6)).mean().item()),
        "nonzero_mape": float(nonzero_mape),
        "median_abs_error": float(abs_error.median().item()),
        "p90_abs_error": float(torch.quantile(abs_error, 0.9).item()),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot.item() > 0 else 0.0,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Current Real-Data Benchmark",
        "",
        f"- version: `{summary['benchmark_version']}`",
        f"- device: `{summary['device']}`",
        f"- router: `{summary['router_dir']}`",
        "",
        "## Dataset Sizes",
        "",
        "| task | examples |",
        "| --- | ---: |",
    ]
    for task, size in summary["dataset_sizes"].items():
        lines.append(f"| {task} | {size} |")
    lines.extend(
        [
            "",
            "Dataset sources:",
            "",
            "| task | heldout | synthetic_extra |",
            "| --- | ---: | ---: |",
        ]
    )
    for task in ("sum", "count", "predict"):
        sources = summary.get("dataset_sources", {}).get(task, {})
        lines.append(
            f"| {task} | {int(sources.get('heldout', 0))} | {int(sources.get('synthetic_extra', 0))} |"
        )

    if summary.get("router") is not None:
        router = summary["router"]
        lines.extend(
            [
                "",
                "## Router",
                "",
                f"- overall accuracy: `{router['accuracy']:.4f}`",
                "",
                "| task | examples | accuracy | avg confidence |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for task, metrics in router["by_task"].items():
            lines.append(
                f"| {task} | {metrics['num_examples']} | {metrics['accuracy']:.4f} | {metrics['avg_confidence']:.4f} |"
            )
        misroutes = [
            record
            for record in router.get("records", [])
            if not record.get("correct", False)
        ][:10]
        if misroutes:
            lines.extend(
                [
                    "",
                    "Top router misroutes:",
                    "",
                    "| expected | selected | source | sample_id | confidence |",
                    "| --- | --- | --- | --- | ---: |",
                ]
            )
            for record in misroutes:
                lines.append(
                    f"| {record['task']} | {record['selected_expert']} | {record.get('benchmark_source', 'heldout')} | "
                    f"{record['sample_id']} | {float(record['confidence']):.4f} |"
                )

    lines.extend(
        [
            "",
            "## Expert Metrics By Perturbation",
            "",
            "| perturbation | task | key metric | MAE | RMSE | MAPE | R2 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for perturbation, by_task in summary["experts"].items():
        for task, metrics in by_task.items():
            if task == "predict_baseline":
                key_metric = metrics["mape"]
            elif task == "count":
                key_metric = metrics.get("exact_match", 0.0)
            else:
                key_metric = metrics["mape"]
            display_mape = metrics.get("nonzero_mape", metrics["mape"]) if task == "count" else metrics["mape"]
            lines.append(
                f"| {perturbation} | {task} | {key_metric:.4f} | {metrics['mae']:.4f} | "
                f"{metrics['rmse']:.4f} | {display_mape:.4f} | {metrics['r2']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Clean Slice Metrics",
            "",
            "| slice | examples | MAE | MAPE | R2 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in summary["slices"].items():
        display_mape = metrics.get("nonzero_mape", metrics["mape"]) if name.startswith("count.") else metrics["mape"]
        lines.append(
            f"| {name} | {int(metrics['num_examples'])} | {metrics['mae']:.4f} | "
            f"{display_mape:.4f} | {metrics['r2']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run current real-data benchmark for active experts.")
    parser.add_argument("--router-dir", default=str(REPO_ROOT / "data/moe_router_realdata_v1"))
    parser.add_argument("--output-file", default=str(REPO_ROOT / "runs/realdata_experts/current_realdata_benchmark.json"))
    parser.add_argument("--markdown-file", default=str(REPO_ROOT / "runs/realdata_experts/current_realdata_benchmark.md"))
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--min-examples-per-task", type=int, default=200)
    parser.add_argument("--skip-router", action="store_true")
    parser.add_argument("--no-records", action="store_true")
    parser.add_argument(
        "--perturbation",
        action="append",
        choices=DEFAULT_PERTURBATIONS,
        help="Perturbation to run. Can be passed multiple times. Defaults to all.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    perturbations = tuple(args.perturbation) if args.perturbation else DEFAULT_PERTURBATIONS
    summary = run_current_realdata_benchmark(
        router_dir=args.router_dir,
        output_file=args.output_file,
        markdown_file=args.markdown_file,
        perturbations=perturbations,
        limit_per_task=args.limit_per_task,
        min_examples_per_task=args.min_examples_per_task,
        skip_router=args.skip_router,
        include_records=not args.no_records,
    )
    printable = {
        "benchmark_version": summary["benchmark_version"],
        "device": summary["device"],
        "dataset_sizes": summary["dataset_sizes"],
        "dataset_sources": summary["dataset_sources"],
        "router_accuracy": None if summary["router"] is None else summary["router"]["accuracy"],
        "expert_clean": summary["experts"].get("clean", {}),
        "output_file": args.output_file,
        "markdown_file": args.markdown_file,
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
