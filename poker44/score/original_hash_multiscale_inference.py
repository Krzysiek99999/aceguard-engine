"""Inference for the label-blind hash-multiscale tree surface."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from poker44.score.original_hash_multiscale_features import chunk_features
from poker44.score.original_tree_surface_inference import (
    BLEND_STRATEGY,
    blend_component_predictions,
    model_scores,
)


FEATURE_EXTRACTOR = (
    "poker44.score.original_hash_multiscale_features:chunk_features"
)
VIEW_MODE = "hash_multiscale_full"


def _validated_components(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(bundle, dict):
        raise TypeError("hash-multiscale bundle must be a dict")
    if str(bundle.get("feature_extractor") or "") != FEATURE_EXTRACTOR:
        raise ValueError("hash-multiscale feature extractor mismatch")
    if str(bundle.get("view_mode") or "") != VIEW_MODE:
        raise ValueError("hash-multiscale view mode mismatch")
    if str(bundle.get("blend_strategy") or "") != BLEND_STRATEGY:
        raise ValueError("hash-multiscale blend strategy mismatch")
    components = list(bundle.get("components") or [])
    if not components:
        raise ValueError("hash-multiscale bundle has no components")
    names: set[str] = set()
    for component in components:
        name = str(component.get("name") or "")
        keys = [str(key) for key in component.get("keys") or []]
        if not name or name in names or not keys or len(keys) != len(set(keys)):
            raise ValueError("hash-multiscale component schema is invalid")
        if "model" not in component:
            raise ValueError(f"component {name} has no model")
        if str(component.get("score_method") or "") not in {"predict", "predict_proba"}:
            raise ValueError(f"component {name} has invalid score method")
        names.add(name)
    return components


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        bundle = pickle.load(handle)
    _validated_components(bundle)
    return bundle


def score_chunks(
    chunks: Sequence[Sequence[dict[str, Any]]], bundle: dict[str, Any]
) -> list[float]:
    components = _validated_components(bundle)
    if not chunks:
        return []
    rows = [chunk_features(list(chunk or [])) for chunk in chunks]
    predictions: list[np.ndarray] = []
    for component in components:
        keys = [str(key) for key in component["keys"]]
        matrix = np.asarray(
            [[float(row.get(key, 0.0)) for key in keys] for row in rows],
            dtype=np.float32,
        )
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        predictions.append(
            model_scores(component["model"], matrix, str(component["score_method"]))
        )
    return [float(value) for value in blend_component_predictions(np.vstack(predictions))]


def score_from_file(
    chunks: Sequence[Sequence[dict[str, Any]]],
    model_path: str | os.PathLike[str],
) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path))
