from __future__ import annotations

import torch
import torch.nn as nn


class PredictV5GRUModel(nn.Module):
    """Sequence-only revenue growth predictor.

    The model predicts a bounded residual around a simple peer/history
    baseline. This keeps inference stable on small documents and unseen
    company names without relying on company embeddings or graph layers.
    """

    def __init__(
        self,
        history_dim: int = 5,
        static_dim: int = 15,
        hidden_dim: int = 64,
        residual_scale: float = 0.35,
        dropout: float = 0.05,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        normalized_rnn_type = rnn_type.lower()
        if normalized_rnn_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        self.rnn_type = normalized_rnn_type
        self.residual_scale = residual_scale
        encoder_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.history_encoder = encoder_cls(
            input_size=history_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        history_seq: torch.Tensor,
        history_mask: torch.Tensor,
        static_features: torch.Tensor,
        baseline_growth: torch.Tensor,
    ) -> torch.Tensor:
        masked_history = history_seq * history_mask.unsqueeze(-1)
        _, hidden_state = self.history_encoder(masked_history)
        hidden = hidden_state[0] if self.rnn_type == "lstm" else hidden_state
        history_repr = hidden.squeeze(0)
        static_repr = self.static_encoder(static_features)
        history_available = (history_mask.sum(dim=-1, keepdim=True) > 0).to(static_features.dtype)
        features = torch.cat(
            [
                history_repr,
                static_repr,
                baseline_growth.view(-1, 1),
                history_available,
            ],
            dim=-1,
        )
        residual = self.residual_scale * torch.tanh(self.head(features)).squeeze(-1)
        return baseline_growth.view(-1) + residual
