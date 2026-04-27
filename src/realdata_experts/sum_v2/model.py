from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool


class SumV2Model(nn.Module):
    def __init__(self, in_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        final_linear = self.gate_head[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.constant_(final_linear.bias, 4.0)

    def forward(
        self,
        x: torch.Tensor,
        node_values: torch.Tensor,
        batch: torch.Tensor,
        query_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_hidden = self.node_encoder(x)
        query_hidden = self.query_encoder(query_attr)
        query_hidden_nodes = query_hidden[batch]

        gate_logits = self.gate_head(torch.cat((node_hidden, query_hidden_nodes), dim=-1))
        node_gates = torch.sigmoid(gate_logits).squeeze(-1)
        selected_amount = (node_values * query_attr[batch]).sum(dim=-1)
        base_sum = global_add_pool(node_gates * selected_amount, batch).squeeze(-1)

        pooled_hidden = global_mean_pool(node_hidden, batch)
        correction = self.correction_head(torch.cat((pooled_hidden, query_attr), dim=-1)).squeeze(-1)
        prediction = base_sum + 0.05 * correction
        return prediction, node_gates

