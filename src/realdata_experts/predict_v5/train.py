"""Train PredictV5: recurrent time-series predictor with peer-growth context."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from realdata_experts.common import signed_log1p

from .data import load_predict_v5_splits
from .model import PredictV5GRUModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _predict(model: PredictV5GRUModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["history_seq"],
        batch["history_mask"],
        batch["static_features"],
        batch["baseline_growth"],
    )


def train_one_epoch(
    model: PredictV5GRUModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = _predict(model, batch)
        loss = F.smooth_l1_loss(pred, batch["y"].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = pred.numel()
        total_loss += float(loss.item()) * batch_size
        total += batch_size
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    model: PredictV5GRUModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    baselines: list[torch.Tensor] = []
    prev_revenues: list[torch.Tensor] = []
    target_revenues: list[torch.Tensor] = []
    has_prevs: list[torch.Tensor] = []

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        pred = _predict(model, batch)
        preds.append(pred.cpu())
        targets.append(batch["y"].view(-1).cpu())
        baselines.append(batch["baseline_growth"].view(-1).cpu())
        prev_revenues.append(batch["prev_revenue"].view(-1).cpu())
        target_revenues.append(batch["target_revenue"].view(-1).cpu())
        has_prevs.append(batch["has_prev"].view(-1).cpu())

    pred = torch.cat(preds)
    target = torch.cat(targets)
    baseline = torch.cat(baselines)
    prev_revenue = torch.cat(prev_revenues)
    target_revenue = torch.cat(target_revenues)
    has_prev = torch.cat(has_prevs) > 0.5

    growth_metrics = _regression_metrics(pred, target)
    baseline_metrics = _regression_metrics(baseline, target)
    revenue_metrics = _revenue_metrics(pred, prev_revenue, target_revenue, has_prev)
    baseline_revenue_metrics = _revenue_metrics(baseline, prev_revenue, target_revenue, has_prev)
    return {
        "loss": F.smooth_l1_loss(pred, target).item(),
        **{f"growth_{key}": value for key, value in growth_metrics.items()},
        **{f"baseline_growth_{key}": value for key, value in baseline_metrics.items()},
        **{f"revenue_{key}": value for key, value in revenue_metrics.items()},
        **{f"baseline_revenue_{key}": value for key, value in baseline_revenue_metrics.items()},
    }


def _regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    rmse = mse**0.5
    ss_tot = ((target - target.mean()) ** 2).sum()
    ss_res = ((target - pred) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot.item() > 0 else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def _revenue_metrics(
    pred_growth: torch.Tensor,
    prev_revenue: torch.Tensor,
    target_revenue: torch.Tensor,
    has_prev: torch.Tensor,
) -> dict[str, float]:
    if not has_prev.any():
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "coverage": 0.0}
    pred_growth = pred_growth[has_prev]
    prev_revenue = prev_revenue[has_prev]
    target_revenue = target_revenue[has_prev]
    prev_log = torch.tensor([signed_log1p(value.item()) for value in prev_revenue], dtype=torch.float32)
    pred_revenue = torch.expm1(prev_log + pred_growth).clamp_min(0.0)
    abs_error = (pred_revenue - target_revenue).abs()
    squared_error = (pred_revenue - target_revenue) ** 2
    mape = (abs_error / target_revenue.abs().clamp_min(1e-6)).mean().item()
    return {
        "mae": float(abs_error.mean().item()),
        "rmse": float(squared_error.mean().sqrt().item()),
        "mape": float(mape),
        "coverage": float(has_prev.float().mean().item()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PredictV5 recurrent time-series expert.")
    parser.add_argument("--dataset-file", default="data/realdata_inputs/prediction_dataset.jsonl")
    parser.add_argument("--output-dir", default="runs/realdata_experts/predict_v5")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--rnn-type", choices=("gru", "lstm"), default="gru")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    return parser


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_ds, val_ds, test_ds, summary = load_predict_v5_splits(
        args.dataset_file,
        max_samples=args.max_samples,
        history_len=args.history_len,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = PredictV5GRUModel(
        history_dim=5,
        static_dim=summary.static_dim,
        hidden_dim=args.hidden_dim,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.scheduler_patience)

    best_state = None
    best_epoch = 0
    best_loss = float("inf")
    no_improve = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model.")
    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "history_dim": 5,
        "static_dim": summary.static_dim,
        "hidden_dim": args.hidden_dim,
        "residual_scale": args.residual_scale,
        "dropout": args.dropout,
        "rnn_type": args.rnn_type,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "best_epoch": best_epoch,
            "history_len": args.history_len,
        },
        out_dir / "best_model.pt",
    )
    results = {
        "split_summary": summary.__dict__,
        "best_epoch": best_epoch,
        "model_config": model_config,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint_path": str(out_dir / "best_model.pt"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = build_arg_parser()
    print(json.dumps(run_training(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
