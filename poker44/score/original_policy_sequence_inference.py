"""Inference runtime for the natural-unit hierarchical policy sequence model."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from poker44.score.chunk_sequence_model import SequenceModelConfig, encode_chunk
from poker44.score.chunk_view_aggregation import expand_chunk_views, reduce_view_scores
from poker44.score.original_policy_sequence_model import OriginalPolicySequenceNetwork


_FLOAT_KEYS = {"cont", "hand_meta"}
_BOOL_KEYS = {"action_mask", "hand_mask"}
_INPUT_KEYS = (
    "action_type",
    "street",
    "actor_role",
    "actor_alias",
    "amount_bucket",
    "pot_flow",
    "pot_frac",
    "street_pos",
    "first_in_street",
    "cont",
    "action_mask",
    "hand_mask",
    "hand_end",
    "hand_meta",
)


def build_network(component: dict[str, Any]) -> OriginalPolicySequenceNetwork:
    architecture = dict(component.get("architecture") or {})
    sequence_config = SequenceModelConfig(**dict(architecture.get("sequence_config") or {}))
    network = OriginalPolicySequenceNetwork(
        sequence_config,
        temporal_layers=int(architecture.get("temporal_layers", 1)),
        temporal_dropout=float(architecture.get("temporal_dropout", sequence_config.dropout)),
    )
    network.load_state_dict(component["state_dict"])
    network.eval()
    return network


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"{model_path} did not contain a dict bundle")
    components = list(bundle.get("components") or [])
    if not components:
        components = [bundle]
    bundle["_runtime_models"] = [build_network(component) for component in components]
    return bundle


def tensorize_chunks(
    chunks: Sequence[Sequence[dict[str, Any]]],
    config: SequenceModelConfig,
) -> dict[str, torch.Tensor]:
    encoded = [
        encode_chunk(
            list(chunk),
            max_hands_per_chunk=int(config.max_hands_per_chunk),
            max_actions_per_hand=int(config.max_actions_per_hand),
        )
        for chunk in chunks
    ]
    if not encoded:
        return {}
    tensors: dict[str, torch.Tensor] = {}
    for key in _INPUT_KEYS:
        values = np.stack([row[key] for row in encoded], axis=0)
        if key in _FLOAT_KEYS:
            tensors[key] = torch.from_numpy(values).float()
        elif key in _BOOL_KEYS:
            tensors[key] = torch.from_numpy(values).bool()
        else:
            tensors[key] = torch.from_numpy(values).long()
    return tensors


def score_view_logits(
    chunks: Sequence[Sequence[dict[str, Any]]],
    bundle: dict[str, Any],
    *,
    batch_size: int = 16,
) -> list[float]:
    if not chunks:
        return []
    networks = bundle.get("_runtime_models")
    if networks is None:
        components = list(bundle.get("components") or [])
        if not components:
            components = [bundle]
        networks = [build_network(component) for component in components]
        bundle["_runtime_models"] = networks
    if not networks:
        raise ValueError("policy sequence bundle has no model components")
    config = networks[0].config
    if any(network.config.to_dict() != config.to_dict() for network in networks[1:]):
        raise ValueError("policy sequence ensemble components use different tensor schemas")
    tensors = tensorize_chunks(chunks, config)
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(chunks), max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), len(chunks))
            inputs = {key: value[start:stop] for key, value in tensors.items()}
            component_logits = [network(**inputs) for network in networks]
            logits = torch.stack(component_logits, dim=0).mean(dim=0)
            scores.extend(float(value) for value in logits.detach().cpu().numpy())
    return scores


def score_chunks(
    chunks: Sequence[Sequence[dict[str, Any]]],
    bundle: dict[str, Any],
    *,
    batch_size: int = 16,
) -> list[float]:
    if not chunks:
        return []
    view_mode = str(bundle.get("view_mode") or "partition35_mean")
    views, owners = expand_chunk_views(chunks, view_mode)
    view_logits = score_view_logits(views, bundle, batch_size=batch_size)
    return reduce_view_scores(view_logits, owners, len(chunks), view_mode)


def score_from_file(
    chunks: Sequence[Sequence[dict[str, Any]]],
    model_path: str | os.PathLike[str],
) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path))
