from __future__ import annotations

import math

import torch
import torch.nn as nn


class PredictV6Model(nn.Module):
    """Hybrid history + in-context peer predictor.

    The model keeps the stable peer/history baseline from v5, then learns a
    bounded residual from optional target history and optional peer-context
    examples. This lets us compare sequence-only, Kumo-style context-only, and
    hybrid variants under one architecture.
    """

    def __init__(
        self,
        history_dim: int = 5,
        static_dim: int = 15,
        context_dim: int = 10,
        hidden_dim: int = 64,
        residual_scale: float = 0.1,
        dropout: float = 0.05,
        history_mode: str = "lstm",
        context_mode: str = "attention",
    ) -> None:
        super().__init__()
        normalized_history_mode = history_mode.lower()
        normalized_context_mode = context_mode.lower()
        if normalized_history_mode not in {"none", "gru", "lstm"}:
            raise ValueError(f"Unsupported history_mode: {history_mode}")
        if normalized_context_mode not in {"none", "deepsets", "attention"}:
            raise ValueError(f"Unsupported context_mode: {context_mode}")

        self.hidden_dim = hidden_dim
        self.residual_scale = residual_scale
        self.history_mode = normalized_history_mode
        self.context_mode = normalized_context_mode

        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if self.history_mode == "gru":
            self.history_encoder = nn.GRU(history_dim, hidden_dim, batch_first=True)
        elif self.history_mode == "lstm":
            self.history_encoder = nn.LSTM(history_dim, hidden_dim, batch_first=True)
        else:
            self.history_encoder = None

        if self.context_mode != "none":
            self.context_token_encoder = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.context_query = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.context_token_encoder = None
            self.context_query = None

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        history_seq: torch.Tensor,
        history_mask: torch.Tensor,
        context_seq: torch.Tensor,
        context_mask: torch.Tensor,
        static_features: torch.Tensor,
        baseline_growth: torch.Tensor,
    ) -> torch.Tensor:
        static_repr = self.static_encoder(static_features)
        history_repr, history_available = self._encode_history(history_seq, history_mask, static_features)
        context_repr, context_available = self._encode_context(context_seq, context_mask, static_repr)
        features = torch.cat(
            [
                static_repr,
                history_repr,
                context_repr,
                baseline_growth.view(-1, 1),
                history_available,
                context_available,
            ],
            dim=-1,
        )
        residual = self.residual_scale * torch.tanh(self.head(features)).squeeze(-1)
        return baseline_growth.view(-1) + residual

    def _encode_history(
        self,
        history_seq: torch.Tensor,
        history_mask: torch.Tensor,
        static_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = (history_mask.sum(dim=-1, keepdim=True) > 0).to(static_features.dtype)
        if self.history_encoder is None:
            return torch.zeros((static_features.shape[0], self.hidden_dim), device=static_features.device), available

        masked_history = history_seq * history_mask.unsqueeze(-1)
        _, hidden_state = self.history_encoder(masked_history)
        hidden = hidden_state[0] if self.history_mode == "lstm" else hidden_state
        return hidden.squeeze(0) * available, available

    def _encode_context(
        self,
        context_seq: torch.Tensor,
        context_mask: torch.Tensor,
        static_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        available = (context_mask.sum(dim=-1, keepdim=True) > 0).to(static_repr.dtype)
        if self.context_token_encoder is None:
            return torch.zeros((static_repr.shape[0], self.hidden_dim), device=static_repr.device), available

        token_repr = self.context_token_encoder(context_seq) * context_mask.unsqueeze(-1)
        if self.context_mode == "deepsets":
            denom = context_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return token_repr.sum(dim=1) / denom, available

        query = self.context_query(static_repr)
        scores = (token_repr * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(context_mask <= 0, -1e9)
        weights = torch.softmax(scores, dim=-1) * context_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context_repr = torch.bmm(weights.unsqueeze(1), token_repr).squeeze(1)
        return context_repr, available
