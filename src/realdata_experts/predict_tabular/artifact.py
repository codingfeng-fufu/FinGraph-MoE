"""Train and serve the tabular residual predictor.

The current best real-data prediction run is ElasticNet over flattened
PredictV6 features, predicting a residual over the peer/history baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from realdata_experts.common import signed_log1p
from realdata_experts.predict_v6.data import load_predict_v6_splits


def context_summary(sample: dict[str, torch.Tensor]) -> np.ndarray:
    context = sample["context_seq"]
    mask = sample["context_mask"].view(-1, 1)
    count = float(mask.sum().item())
    if count <= 0:
        return np.zeros(context.shape[-1] * 4 + 1, dtype=np.float32)

    masked = context * mask
    mean = masked.sum(dim=0) / count
    centered = (context - mean) * mask
    std = torch.sqrt((centered**2).sum(dim=0) / count)
    large_pos = torch.full_like(context, 1e9)
    large_neg = torch.full_like(context, -1e9)
    minimum = torch.where(mask > 0, context, large_pos).min(dim=0).values
    maximum = torch.where(mask > 0, context, large_neg).max(dim=0).values
    return torch.cat([mean, std, minimum, maximum, torch.tensor([count / max(context.shape[0], 1)])]).numpy()


def sample_to_features(sample: dict[str, torch.Tensor], feature_set: str = "flat") -> np.ndarray:
    parts = [
        sample["static_features"].numpy(),
        sample["history_seq"].reshape(-1).numpy(),
        sample["history_mask"].numpy(),
    ]
    if feature_set == "summary":
        parts.append(context_summary(sample))
    elif feature_set == "flat":
        parts.extend([sample["context_seq"].reshape(-1).numpy(), sample["context_mask"].numpy()])
    else:
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    return np.concatenate(parts).astype(np.float32)


def dataset_to_arrays(dataset, feature_set: str = "flat") -> dict[str, np.ndarray]:
    samples = list(dataset)
    return {
        "x": np.stack([sample_to_features(sample, feature_set) for sample in samples]),
        "y": np.array([float(sample["y"].item()) for sample in samples], dtype=np.float32),
        "baseline": np.array([float(sample["baseline_growth"].item()) for sample in samples], dtype=np.float32),
        "prev_revenue": np.array([float(sample["prev_revenue"].item()) for sample in samples], dtype=np.float32),
        "target_revenue": np.array([float(sample["target_revenue"].item()) for sample in samples], dtype=np.float32),
        "has_prev": np.array([float(sample["has_prev"].item()) > 0.5 for sample in samples], dtype=bool),
    }


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_t = torch.tensor(pred, dtype=torch.float32)
    target_t = torch.tensor(target, dtype=torch.float32)
    mse = F.mse_loss(pred_t, target_t).item()
    mae = F.l1_loss(pred_t, target_t).item()
    ss_tot = ((target_t - target_t.mean()) ** 2).sum()
    ss_res = ((target_t - pred_t) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot.item() > 0 else 0.0
    return {"mse": mse, "rmse": mse**0.5, "mae": mae, "r2": r2}


def revenue_metrics(
    growth: np.ndarray,
    prev_revenue: np.ndarray,
    target_revenue: np.ndarray,
    has_prev: np.ndarray,
) -> dict[str, float]:
    if not has_prev.any():
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "coverage": 0.0}
    growth = growth[has_prev]
    prev_revenue = prev_revenue[has_prev]
    target_revenue = target_revenue[has_prev]
    prev_log = np.array([signed_log1p(float(value)) for value in prev_revenue], dtype=np.float32)
    pred_revenue = np.expm1(prev_log + growth).clip(min=0.0)
    abs_error = np.abs(pred_revenue - target_revenue)
    squared_error = (pred_revenue - target_revenue) ** 2
    return {
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "mape": float((abs_error / np.clip(np.abs(target_revenue), 1e-6, None)).mean()),
        "coverage": float(has_prev.mean()),
    }


def evaluate_growth(pred: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, float]:
    growth = regression_metrics(pred, arrays["y"])
    revenue = revenue_metrics(pred, arrays["prev_revenue"], arrays["target_revenue"], arrays["has_prev"])
    baseline_growth = regression_metrics(arrays["baseline"], arrays["y"])
    baseline_revenue = revenue_metrics(
        arrays["baseline"],
        arrays["prev_revenue"],
        arrays["target_revenue"],
        arrays["has_prev"],
    )
    return {
        **{f"growth_{key}": value for key, value in growth.items()},
        **{f"revenue_{key}": value for key, value in revenue.items()},
        **{f"baseline_growth_{key}": value for key, value in baseline_growth.items()},
        **{f"baseline_revenue_{key}": value for key, value in baseline_revenue.items()},
    }


def build_elastic_net_model() -> Any:
    return make_pipeline(
        StandardScaler(),
        ElasticNetCV(alphas=np.logspace(-4, 1, 20), l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=5000),
    )


def train_artifact(
    dataset_file: str = "data/realdata_inputs/prediction_dataset.jsonl",
    output_dir: str | Path = "runs/realdata_experts/predict_tabular_elastic_net_flat_residual",
    seed: int = 23,
    history_len: int = 5,
    max_peers: int = 32,
    feature_set: str = "flat",
    target_mode: str = "residual",
) -> dict[str, Any]:
    if target_mode not in {"direct", "residual"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    train_ds, val_ds, test_ds, summary = load_predict_v6_splits(
        dataset_file,
        history_len=history_len,
        max_peers=max_peers,
    )
    train = dataset_to_arrays(train_ds, feature_set)
    val = dataset_to_arrays(val_ds, feature_set)
    test = dataset_to_arrays(test_ds, feature_set)
    target_train = train["y"] if target_mode == "direct" else train["y"] - train["baseline"]

    model = build_elastic_net_model()
    model.fit(train["x"], target_train)

    def predict(arrays: dict[str, np.ndarray]) -> np.ndarray:
        raw = model.predict(arrays["x"]).astype(np.float32)
        return raw if target_mode == "direct" else (arrays["baseline"] + raw).astype(np.float32)

    val_pred = predict(val)
    test_pred = predict(test)
    payload = {
        "model": model,
        "model_name": "elastic_net",
        "feature_set": feature_set,
        "target_mode": target_mode,
        "history_len": history_len,
        "max_peers": max_peers,
        "seed": seed,
        "split_summary": summary.__dict__,
        "val_metrics": evaluate_growth(val_pred, val),
        "test_metrics": evaluate_growth(test_pred, test),
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_dir / "artifact.joblib")
    (out_dir / "metrics.json").write_text(
        _json_dumps_without_model(payload),
        encoding="utf-8",
    )
    return {**payload, "artifact_path": str(out_dir / "artifact.joblib")}


def _json_dumps_without_model(payload: dict[str, Any]) -> str:
    import json

    serializable = {key: value for key, value in payload.items() if key != "model"}
    return json.dumps(serializable, indent=2)


def load_artifact(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_growth(sample: dict[str, torch.Tensor], artifact: dict[str, Any]) -> float:
    feature_set = str(artifact.get("feature_set", "flat"))
    target_mode = str(artifact.get("target_mode", "residual"))
    features = sample_to_features(sample, feature_set).reshape(1, -1)
    raw = float(artifact["model"].predict(features)[0])
    baseline = float(sample["baseline_growth"].view(-1)[0].item())
    return raw if target_mode == "direct" else baseline + raw

