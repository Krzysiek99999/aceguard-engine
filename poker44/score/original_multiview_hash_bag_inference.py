"""Inference for a source-augmented, balanced multi-view hash-bag model."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from poker44.score.balanced_hash_views import batch_view_lanes
from poker44.score.original_hash_bag_features import chunk_features
from poker44.score.original_tree_surface_inference import model_scores, percentile_rank


FEATURE_EXTRACTOR = "poker44.score.original_hash_bag_features:chunk_features"
VIEW_MODE = "balanced_hash_multiview_v1"
BLEND_STRATEGY = "lane_component_percentile_rank_mean"


def _validated_components(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(bundle, dict):
        raise TypeError("multi-view hash-bag bundle must be a dict")
    if str(bundle.get("feature_extractor") or "") != FEATURE_EXTRACTOR:
        raise ValueError("multi-view hash-bag feature extractor mismatch")
    if str(bundle.get("view_mode") or "") != VIEW_MODE:
        raise ValueError("multi-view hash-bag view mode mismatch")
    if str(bundle.get("blend_strategy") or "") != BLEND_STRATEGY:
        raise ValueError("multi-view hash-bag blend strategy mismatch")
    components = list(bundle.get("components") or [])
    if not components:
        raise ValueError("multi-view hash-bag bundle has no components")
    names: set[str] = set()
    for component in components:
        name = str(component.get("name") or "")
        keys = [str(key) for key in component.get("keys") or []]
        method = str(component.get("score_method") or "")
        if not name or name in names or not keys or len(keys) != len(set(keys)):
            raise ValueError("multi-view hash-bag component schema is invalid")
        if "model" not in component or method not in {"predict", "predict_proba"}:
            raise ValueError(f"component {name} has an invalid model contract")
        names.add(name)
    return components


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        bundle = pickle.load(handle)
    _validated_components(bundle)
    return bundle


def aggregate_component_lane_predictions(raw_lanes: np.ndarray) -> np.ndarray:
    """Convert (lanes, components, chunks) raw scores to stable component ranks."""
    values = np.asarray(raw_lanes, dtype=float)
    if values.ndim != 3 or min(values.shape) <= 0:
        raise ValueError("raw lane predictions must have shape (lanes, components, chunks)")
    ranked = np.empty_like(values, dtype=float)
    for lane in range(values.shape[0]):
        for component in range(values.shape[1]):
            ranked[lane, component] = percentile_rank(values[lane, component])
    return np.mean(ranked, axis=0)


def score_chunks_detailed(
    chunks: Sequence[Sequence[dict[str, Any]]], bundle: dict[str, Any]
) -> dict[str, Any]:
    components = _validated_components(bundle)
    if not chunks:
        return {"scores": [], "component_scores": [], "view_mode": "empty", "lanes": 0}
    lanes, mode = batch_view_lanes(chunks)
    raw_lanes: list[np.ndarray] = []
    for lane_chunks in lanes:
        rows = [chunk_features(list(chunk or [])) for chunk in lane_chunks]
        component_rows: list[np.ndarray] = []
        for component in components:
            keys = [str(key) for key in component["keys"]]
            matrix = np.asarray(
                [[float(row.get(key, 0.0)) for key in keys] for row in rows],
                dtype=np.float32,
            )
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            component_rows.append(
                model_scores(component["model"], matrix, str(component["score_method"]))
            )
        raw_lanes.append(np.vstack(component_rows))
    component_scores = aggregate_component_lane_predictions(np.stack(raw_lanes))
    scores = np.mean(component_scores, axis=0)
    return {
        "scores": [float(value) for value in scores],
        "component_scores": component_scores.tolist(),
        "view_mode": mode,
        "lanes": len(lanes),
    }


def score_chunks(
    chunks: Sequence[Sequence[dict[str, Any]]], bundle: dict[str, Any]
) -> list[float]:
    return list(score_chunks_detailed(chunks, bundle)["scores"])


def score_from_file(
    chunks: Sequence[Sequence[dict[str, Any]]], model_path: str | os.PathLike[str]
) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path))
