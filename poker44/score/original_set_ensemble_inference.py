"""Inference for a rank-space ensemble of original hand-set networks."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from poker44.score.original_set_inference import score_chunks as score_component
from poker44.score.original_set_model import OriginalHandSetNetwork


def normalized_rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = np.arange(1, array.size + 1, dtype=float) / array.size
    return ranks


def _prepare_component(component: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(component)
    network = OriginalHandSetNetwork(**dict(runtime["network_config"]))
    network.load_state_dict(runtime["state_dict"])
    network.eval()
    runtime["_runtime_model"] = network
    return runtime


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"{model_path} did not contain a dict bundle")
    components = list(bundle.get("components") or [])
    weights = [float(value) for value in bundle.get("weights") or []]
    if not components or len(components) != len(weights) or sum(max(0.0, value) for value in weights) <= 0.0:
        raise ValueError("original set ensemble has invalid components/weights")
    bundle["_runtime_components"] = [_prepare_component(component) for component in components]
    return bundle


def score_chunks(
    chunks: Sequence[Any],
    bundle: dict[str, Any],
    *,
    batch_size: int = 32,
) -> list[float]:
    if not chunks:
        return []
    components = bundle.get("_runtime_components")
    if components is None:
        components = [_prepare_component(component) for component in list(bundle.get("components") or [])]
        bundle["_runtime_components"] = components
    weights = np.asarray([max(0.0, float(value)) for value in bundle.get("weights") or []], dtype=float)
    if len(components) != len(weights) or float(weights.sum()) <= 0.0:
        raise ValueError("original set ensemble has invalid runtime components/weights")
    columns = [
        normalized_rank(score_component(chunks, component, batch_size=batch_size))
        for component in components
    ]
    blended = sum(weight * column for weight, column in zip(weights, columns)) / float(weights.sum())
    return [float(value) for value in blended]


def score_from_file(chunks: Sequence[Any], model_path: str | os.PathLike[str]) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path))
