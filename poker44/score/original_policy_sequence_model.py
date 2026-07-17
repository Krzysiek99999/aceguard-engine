"""Natural-unit hierarchical policy sequence network for Poker44."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from poker44.score.chunk_sequence_model import (
    _ACTION_TYPE_VOCAB,
    _AMOUNT_BUCKET_VOCAB_SIZE,
    _AttentionPool,
    _STREET_VOCAB,
    ChunkSetTransformer,
    SequenceModelConfig,
)


class OriginalPolicySequenceNetwork(ChunkSetTransformer):
    """Encode action syntax, unordered policy and ordered policy drift.

    The inherited branch is permutation invariant across hands. A second hand
    encoder receives position embeddings and captures changes across the
    natural hand sequence. The two views are fused only after independent
    pooling, which keeps the representation useful when one source has weak or
    noisy hand order.
    """

    def __init__(
        self,
        config: SequenceModelConfig,
        *,
        temporal_layers: int = 1,
        temporal_dropout: float | None = None,
    ) -> None:
        super().__init__(config)
        d_model = int(config.d_model)
        dropout = float(config.dropout if temporal_dropout is None else temporal_dropout)
        self.temporal_layers = int(temporal_layers)
        self.temporal_dropout = dropout
        self.hand_position_emb = nn.Embedding(int(config.max_hands_per_chunk), d_model)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config.n_heads),
            dim_feedforward=d_model * int(config.ff_mult),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=self.temporal_layers,
            enable_nested_tensor=False,
        )
        self.temporal_pool = _AttentionPool(d_model, int(config.n_heads), dropout)
        self.fusion_norm = nn.LayerNorm(d_model * 4)
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Masked-policy pretraining predicts only miner-visible categories.
        self.masked_action_head = nn.Linear(d_model, len(_ACTION_TYPE_VOCAB))
        self.masked_street_head = nn.Linear(d_model, len(_STREET_VOCAB))
        self.masked_amount_head = nn.Linear(d_model, _AMOUNT_BUCKET_VOCAB_SIZE)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "sequence_config": self.config.to_dict(),
            "temporal_layers": int(self.temporal_layers),
            "temporal_dropout": float(self.temporal_dropout),
        }

    @staticmethod
    def _masked_moments(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = mask.unsqueeze(-1).to(values.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        mean = (values * weight).sum(dim=1) / count
        centered = (values - mean.unsqueeze(1)) * weight
        std = torch.sqrt((centered.square().sum(dim=1) / count).clamp_min(1e-8))
        return mean, std

    @staticmethod
    def _masked_edge_delta(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean(last third) - mean(first third) for each natural hand sequence."""
        batch, hands, _ = values.shape
        positions = torch.arange(hands, device=values.device).unsqueeze(0).expand(batch, -1)
        lengths = mask.sum(dim=1).clamp_min(1)
        first_cut = torch.div(lengths + 2, 3, rounding_mode="floor").unsqueeze(1)
        last_cut = (lengths - torch.div(lengths, 3, rounding_mode="floor")).unsqueeze(1)
        first_mask = mask & (positions < first_cut)
        last_mask = mask & (positions >= last_cut)
        first_mean, _ = OriginalPolicySequenceNetwork._masked_moments(values, first_mask)
        last_mean, _ = OriginalPolicySequenceNetwork._masked_moments(values, last_mask)
        return last_mean - first_mean

    def encode_policy(
        self,
        *,
        action_type: torch.Tensor,
        street: torch.Tensor,
        actor_role: torch.Tensor,
        actor_alias: torch.Tensor,
        amount_bucket: torch.Tensor,
        pot_flow: torch.Tensor,
        pot_frac: torch.Tensor,
        street_pos: torch.Tensor,
        first_in_street: torch.Tensor,
        cont: torch.Tensor,
        action_mask: torch.Tensor,
        hand_mask: torch.Tensor,
        hand_end: torch.Tensor,
        hand_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-hand embeddings and per-action contextual states."""
        batch, hands, actions = action_type.shape
        flat_shape = (batch * hands, actions)
        position_ids = torch.arange(actions, device=action_type.device).unsqueeze(0).expand(flat_shape)
        flat_action_type = action_type.reshape(flat_shape)
        flat_street = street.reshape(flat_shape)
        flat_actor = actor_role.reshape(flat_shape)
        flat_actor_alias = actor_alias.reshape(flat_shape)
        flat_amount_bucket = amount_bucket.reshape(flat_shape)
        flat_pot_flow = pot_flow.reshape(flat_shape)
        flat_pot_frac = pot_frac.reshape(flat_shape)
        flat_street_pos = street_pos.reshape(flat_shape)
        flat_first_in_street = first_in_street.reshape(flat_shape)
        flat_cont = cont.reshape(batch * hands, actions, -1)
        flat_action_mask = action_mask.reshape(flat_shape)

        embedded = (
            self.action_type_emb(flat_action_type)
            + self.street_emb(flat_street)
            + self.actor_emb(flat_actor)
            + self.action_pos_emb(position_ids)
        )
        if int(self.config.schema_version) >= 2:
            embedded = (
                embedded
                + self.actor_alias_emb(flat_actor_alias)
                + self.street_pos_emb(flat_street_pos)
                + self.first_in_street_emb(flat_first_in_street)
            )
        if bool(self.config.use_amount_features):
            embedded = (
                embedded
                + self.amount_bucket_emb(flat_amount_bucket)
                + self.pot_flow_emb(flat_pot_flow)
                + self.cont_proj(flat_cont)
            )
        if int(self.config.schema_version) >= 3 and bool(self.config.use_amount_features):
            embedded = embedded + self.pot_frac_emb(flat_pot_frac)
        embedded = self.input_dropout(self.input_norm(embedded))

        # TransformerEncoder must see at least one valid key in padded hands.
        safe_action_mask = flat_action_mask.clone()
        empty_hands = ~safe_action_mask.any(dim=1)
        if empty_hands.any():
            safe_action_mask[empty_hands, 0] = True
            embedded = embedded.clone()
            embedded[empty_hands, 0] = 0.0
        key_padding = ~safe_action_mask
        action_states = self.action_encoder(embedded, src_key_padding_mask=key_padding)
        hand_emb = self.action_pool(action_states, key_padding_mask=key_padding)
        hand_emb = hand_emb.masked_fill(empty_hands.unsqueeze(-1), 0.0)
        hand_emb = hand_emb.reshape(batch, hands, -1)

        if int(self.config.schema_version) >= 2:
            meta_input = hand_meta
            if not bool(self.config.use_amount_features):
                meta_input = hand_meta.clone()
                meta_input[..., 0] = 0.0
            meta = self.hand_meta_proj(meta_input) + self.hand_end_emb(hand_end)
            hand_emb = self.hand_meta_norm(hand_emb + meta)
        hand_emb = hand_emb.masked_fill(~hand_mask.unsqueeze(-1), 0.0)
        return hand_emb, action_states.reshape(batch, hands, actions, -1)

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        hand_emb, _ = self.encode_policy(**inputs)
        hand_mask = inputs["hand_mask"].bool()
        padding = ~hand_mask

        set_states = self.hand_encoder(hand_emb, src_key_padding_mask=padding)
        set_pool = self.chunk_pool(set_states, key_padding_mask=padding)

        hand_count = hand_emb.shape[1]
        position = torch.arange(hand_count, device=hand_emb.device).unsqueeze(0)
        temporal_input = hand_emb + self.hand_position_emb(position)
        temporal_input = temporal_input.masked_fill(~hand_mask.unsqueeze(-1), 0.0)
        temporal_states = self.temporal_encoder(temporal_input, src_key_padding_mask=padding)
        temporal_pool = self.temporal_pool(temporal_states, key_padding_mask=padding)
        edge_delta = self._masked_edge_delta(temporal_states, hand_mask)
        _, temporal_std = self._masked_moments(temporal_states, hand_mask)

        fused = torch.cat([set_pool, temporal_pool, edge_delta, temporal_std], dim=1)
        return self.head(self.fusion_norm(fused)).squeeze(-1)

    def masked_policy_logits(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return contextual token logits for masked public-benchmark pretraining."""
        _, action_states = self.encode_policy(**inputs)
        return {
            "action_type": self.masked_action_head(action_states),
            "street": self.masked_street_head(action_states),
            "amount_bucket": self.masked_amount_head(action_states),
        }
