"""Original permutation-invariant hand-set network for Poker44 chunks."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class OriginalHandSetNetwork(nn.Module):
    """Encode hands, contextualize them, and pool the chunk with a learned query."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.config: dict[str, Any] = {
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "heads": int(heads),
            "dropout": float(dropout),
        }
        self.hand_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.context_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.pool_query, mean=0.0, std=0.02)
        self.pool_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hands: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Return one bot logit per chunk.

        `hands` is `[batch, hands, features]`; `valid_mask` is true for real
        hands. No positional encoding is used, so chunk order cannot become a
        shortcut.
        """
        encoded = self.hand_encoder(hands)
        padding_mask = ~valid_mask.bool()
        contextual, _ = self.context_attention(
            encoded,
            encoded,
            encoded,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        contextual = self.context_norm(encoded + contextual)

        mask_float = valid_mask.unsqueeze(-1).to(contextual.dtype)
        count = mask_float.sum(dim=1).clamp_min(1.0)
        mean_pool = (contextual * mask_float).sum(dim=1) / count
        masked = contextual.masked_fill(~valid_mask.unsqueeze(-1), torch.finfo(contextual.dtype).min)
        max_pool = masked.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        query = self.pool_query.expand(contextual.shape[0], -1, -1)
        learned_pool, _ = self.pool_attention(
            query,
            contextual,
            contextual,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        pooled = torch.cat([learned_pool[:, 0, :], mean_pool, max_pool], dim=1)
        return self.head(pooled).squeeze(-1)
