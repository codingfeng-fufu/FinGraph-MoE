import copy

from moe_router.current_realdata_benchmark import (
    _bucket_numeric,
    _generate_count_benchmark_rows,
    _regression_metrics,
    _summarize_slices,
    perturb_example,
)


def test_bucket_numeric_uses_first_matching_boundary() -> None:
    assert _bucket_numeric(3, [(3, "small"), (8, "medium")], "large") == "small"
    assert _bucket_numeric(5, [(3, "small"), (8, "medium")], "large") == "medium"
    assert _bucket_numeric(9, [(3, "small"), (8, "medium")], "large") == "large"


def test_numeric_noise_perturbation_is_deterministic_and_non_mutating() -> None:
    example = {
        "sample_id": "s1",
        "graph": {
            "nodes": [
                {
                    "id": "n0",
                    "attributes": {
                        "revenue": 100.0,
                        "operating_profit": 10.0,
                        "net_profit": 5.0,
                        "employees": 20.0,
                    },
                }
            ],
            "edges": [],
        },
    }
    original = copy.deepcopy(example)

    first = perturb_example(example, "sum", "numeric_noise_5pct")
    second = perturb_example(example, "sum", "numeric_noise_5pct")

    assert example == original
    assert first == second
    assert first["graph"]["nodes"][0]["attributes"]["revenue"] != 100.0
    assert first["graph"]["nodes"][0]["attributes"]["revenue"] >= 0.0


def test_regression_metrics_reports_expected_values() -> None:
    metrics = _regression_metrics([1.0, 3.0], [1.0, 1.0])

    assert metrics["mae"] == 1.0
    assert abs(metrics["rmse"] - 2**0.5) < 1e-6
    assert metrics["mape"] == 1.0


def test_summarize_slices_groups_clean_records() -> None:
    records = [
        {
            "task": "sum",
            "prediction": float(i),
            "answer": float(i),
            "slices": {"target_attr": "revenue", "node_count": "1-3"},
        }
        for i in range(3)
    ]

    summary = _summarize_slices(records)

    assert "sum.target_attr=revenue" in summary
    assert summary["sum.target_attr=revenue"]["num_examples"] == 3.0
    assert summary["sum.target_attr=revenue"]["mae"] == 0.0


def test_generate_count_benchmark_rows_marks_synthetic_and_computes_answer() -> None:
    base = {
        "sample_id": "base",
        "split": "test",
        "graph": {
            "graph_id": "g",
            "nodes": [
                {
                    "id": "n0",
                    "type": "company",
                    "name": "A",
                    "attributes": {"revenue": 10.0, "net_profit": 1.0, "operating_profit": 2.0},
                },
                {
                    "id": "n1",
                    "type": "company",
                    "name": "B",
                    "attributes": {"revenue": 20.0, "net_profit": 3.0, "operating_profit": 4.0},
                },
            ],
            "edges": [],
        },
        "task": {
            "kind": "count",
            "query_text": "base",
            "target": {
                "node_type": "company",
                "attribute": "revenue",
                "condition": {"op": ">", "value": 0.0},
            },
        },
        "answer": 2,
    }

    rows = _generate_count_benchmark_rows([base], 1, seed=1)

    assert rows[0]["benchmark_source"] == "synthetic_extra"
    assert rows[0]["sample_id"] == "bench_count_synth_0000"
    assert isinstance(rows[0]["answer"], float)
