from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from .data import COUNT_ATTRS, COUNT_OPS, MAX_CONDITIONS


class CountV2Model(nn.Module):
    def __init__(self, in_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        cond_dim = len(COUNT_ATTRS) + len(COUNT_OPS) + 2
        self.query_encoder = nn.Sequential(
            nn.Linear(MAX_CONDITIONS * cond_dim, hidden_dim),
            nn.ReLU(),
        )
        self.calibration_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + MAX_CONDITIONS * (1 + len(COUNT_OPS) + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.compare_temperature = nn.Parameter(torch.tensor(0.2))
        final_linear = self.calibration_head[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.constant_(final_linear.bias, 4.0)

    def _soft_match(
        self,
        scaled_node_values: torch.Tensor,
        query_conditions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attr_dim = len(COUNT_ATTRS)
        op_dim = len(COUNT_OPS)
        attr_onehot = query_conditions[..., :attr_dim]
        op_onehot = query_conditions[..., attr_dim : attr_dim + op_dim]
        value = query_conditions[..., -2]
        active = query_conditions[..., -1]

        selected_values = (scaled_node_values.unsqueeze(1) * attr_onehot).sum(dim=-1)
        diff = selected_values - value
        temp = self.compare_temperature.abs().clamp_min(1e-3)

        eq_score = torch.exp(-torch.abs(diff) / temp)
        lt_score = torch.sigmoid((-diff) / temp)
        lte_score = torch.sigmoid((-diff + 1e-3) / temp)
        gt_score = torch.sigmoid(diff / temp)
        gte_score = torch.sigmoid((diff + 1e-3) / temp)
        op_scores = torch.stack((eq_score, lt_score, lte_score, gt_score, gte_score), dim=-1)

        cond_scores = (op_scores * op_onehot).sum(dim=-1)
        cond_scores = cond_scores * active + (1.0 - active)
        return cond_scores.prod(dim=-1), diff

    def forward(
        self,
        x: torch.Tensor,
        node_values: torch.Tensor,
        batch: torch.Tensor,
        query_conditions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_hidden = self.node_encoder(x)
        batch_size = query_conditions.size(0)
        query_flat = query_conditions.view(batch_size, -1)
        query_hidden = self.query_encoder(query_flat)
        query_hidden_nodes = query_hidden[batch]

        # Comparisons must happen in the same scaled space as the encoded query thresholds.
        soft_match, diff = self._soft_match(x, query_conditions[batch])
        attr_dim = len(COUNT_ATTRS)
        op_dim = len(COUNT_OPS)
        op_onehot = query_conditions[batch][..., attr_dim : attr_dim + op_dim]
        active = query_conditions[batch][..., -1]
        calib_features = torch.cat(
            (
                node_hidden,
                query_hidden_nodes,
                diff,
                active,
                op_onehot.reshape(op_onehot.size(0), -1),
            ),
            dim=-1,
        )
        node_scores = soft_match * torch.sigmoid(self.calibration_head(calib_features).squeeze(-1))
        graph_counts = global_add_pool(node_scores, batch).squeeze(-1)
        return graph_counts, node_scores
