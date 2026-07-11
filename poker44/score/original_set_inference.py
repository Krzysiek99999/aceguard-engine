"""Inference runtime for the original AceGuard hand-set network."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from poker44.score.original_behavior_features import hand_feature_rows
from poker44.score.original_set_model import OriginalHandSetNetwork


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"{model_path} did not contain a dict bundle")
    config = dict(bundle.get("network_config") or {})
    network = OriginalHandSetNetwork(**config)
    network.load_state_dict(bundle["state_dict"])
    network.eval()
    bundle["_runtime_model"] = network
    return bundle


def _sample_rows(rows: list[dict[str, float]], max_hands: int) -> list[dict[str, float]]:
    if len(rows) <= max_hands:
        return rows
    indices = np.linspace(0, len(rows) - 1, num=max_hands, dtype=int)
    return [rows[int(index)] for index in indices]


def _tensorize(
    chunks: Sequence[Any],
    *,
    keys: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    max_hands: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.zeros((len(chunks), max_hands, len(keys)), dtype=np.float32)
    valid = np.zeros((len(chunks), max_hands), dtype=bool)
    for chunk_index, chunk in enumerate(chunks):
        rows = _sample_rows(hand_feature_rows(list(chunk or [])), max_hands)
        if not rows:
            valid[chunk_index, 0] = True
            continue
        matrix = np.asarray(
            [[float(row.get(key, 0.0)) for key in keys] for row in rows],
            dtype=np.float32,
        )
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = np.clip((matrix - mean) / scale, -8.0, 8.0)
        x[chunk_index, : len(rows)] = matrix
        valid[chunk_index, : len(rows)] = True
    return torch.from_numpy(x), torch.from_numpy(valid)


def score_chunks(
    chunks: Sequence[Any],
    bundle: dict[str, Any],
    *,
    batch_size: int = 32,
) -> list[float]:
    if not chunks:
        return []
    network = bundle.get("_runtime_model")
    if network is None:
        config = dict(bundle.get("network_config") or {})
        network = OriginalHandSetNetwork(**config)
        network.load_state_dict(bundle["state_dict"])
        network.eval()
        bundle["_runtime_model"] = network
    keys = list(bundle.get("hand_keys") or [])
    if not keys:
        raise ValueError("original set bundle has no hand feature keys")
    mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
    scale = np.asarray(bundle["feature_scale"], dtype=np.float32)
    max_hands = int(bundle.get("max_hands", 100))
    hands, valid = _tensorize(
        chunks,
        keys=keys,
        mean=mean,
        scale=np.maximum(scale, 1e-6),
        max_hands=max_hands,
    )
    out: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(chunks), max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), len(chunks))
            logits = network(hands[start:stop], valid[start:stop])
            out.extend(float(value) for value in logits.detach().cpu().numpy())
    return out


def score_from_file(chunks: Sequence[Any], model_path: str | os.PathLike[str]) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path))
