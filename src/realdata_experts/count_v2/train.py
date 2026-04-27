from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from .data import load_count_v2_splits
from .model import CountV2Model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: CountV2Model, loader: DataLoader, optimizer: AdamW, device: torch.device, aux_weight: float) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        graph_counts, node_scores = model(batch.x, batch.node_values, batch.batch, batch.query_conditions)
        graph_targets = batch.y.view(-1)
        graph_loss = F.mse_loss(graph_counts, graph_targets)
        node_loss = F.binary_cross_entropy(node_scores, batch.node_match_mask)
        loss = graph_loss + aux_weight * node_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(model: CountV2Model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    predictions_all = []
    targets_all = []
    for batch in loader:
        batch = batch.to(device)
        graph_counts, _ = model(batch.x, batch.node_values, batch.batch, batch.query_conditions)
        graph_targets = batch.y.view(-1)
        loss = F.mse_loss(graph_counts, graph_targets)
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
        predictions_all.append(graph_counts.cpu())
        targets_all.append(graph_targets.cpu())

    predictions = torch.cat(predictions_all)
    targets = torch.cat(targets_all)
    mse = F.mse_loss(predictions, targets).item()
    mae = F.l1_loss(predictions, targets).item()
    rmse = mse**0.5
    rounded = predictions.round().clamp_min(0)
    rounded_mae = F.l1_loss(rounded, targets).item()
    exact_match = float((rounded == targets).float().mean().item())
    return {
        "loss": total_loss / max(total_graphs, 1),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rounded_mae": rounded_mae,
        "exact_match": exact_match,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the real-data COUNT v2 expert.")
    parser.add_argument("--dataset-file", type=str, default="data/realdata_inputs/count_600.jsonl")
    parser.add_argument("--output-dir", type=str, default="runs/realdata_experts/count_v2")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    return parser


def run_training(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_graphs, val_graphs, test_graphs, split_summary = load_count_v2_splits(
        args.dataset_file,
        max_samples=args.max_samples,
    )
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    model = CountV2Model(hidden_dim=args.hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.scheduler_patience)

    best_state = None
    best_epoch = 0
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.aux_weight)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["mae"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_exact_match": val_metrics["exact_match"],
            }
        )
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.early_stop_patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model.")
    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"in_dim": 3, "hidden_dim": args.hidden_dim},
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    results = {
        "split_summary": split_summary.__dict__,
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    print(json.dumps(run_training(args), indent=2))


if __name__ == "__main__":
    main()
