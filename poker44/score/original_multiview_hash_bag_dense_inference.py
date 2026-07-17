"""Inference for the balanced multi-view hash-bag plus dense complement."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from poker44.score.balanced_hash_views import batch_view_lanes
from poker44.score.original_hash_bag_features import chunk_features
from poker44.score.original_multiview_hash_bag_inference import (
    _validated_components,
    aggregate_component_lane_predictions,
)
from poker44.score.original_tree_surface_inference import model_scores, percentile_rank


FAMILY = "original_balanced_multiview_hash_bag_dense_ensemble"
DENSE_SOURCE_MODE = "full_natural_lane_only"
DENSE_LIVE_MODE = "all_balanced_40_hand_lanes"


def _validate(
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], list[Any], list[str], np.ndarray]:
    if not isinstance(bundle, dict) or str(bundle.get("family") or "") != FAMILY:
        raise ValueError("multi-view hash-bag/dense family mismatch")
    base = bundle.get("base_bundle")
    dense_models = list(bundle.get("dense_models") or [])
    dense_keys = [str(key) for key in bundle.get("dense_keys") or []]
    weights = np.asarray(bundle.get("view_weights") or [], dtype=float)
    if not isinstance(base, dict):
        raise ValueError("multi-view hash-bag/dense bundle has no base")
    _validated_components(base)
    if len(dense_models) < 2 or not dense_keys or len(dense_keys) != len(set(dense_keys)):
        raise ValueError("multi-view hash-bag/dense dense view is invalid")
    if any(not hasattr(model, "decision_function") for model in dense_models):
        raise ValueError("multi-view hash-bag/dense model lacks decision_function")
    if weights.shape != (2,) or not np.isfinite(weights).all():
        raise ValueError("multi-view hash-bag/dense weights are invalid")
    if np.any(weights < 0.0) or not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("multi-view hash-bag/dense weights must sum to one")
    if str(bundle.get("dense_source_mode") or "") != DENSE_SOURCE_MODE:
        raise ValueError("multi-view hash-bag/dense source mode mismatch")
    if str(bundle.get("dense_live_mode") or "") != DENSE_LIVE_MODE:
        raise ValueError("multi-view hash-bag/dense live mode mismatch")
    return base, dense_models, dense_keys, weights


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        bundle = pickle.load(handle)
    _validate(bundle)
    return bundle


def dense_lane_indices(view_mode: str, lane_count: int) -> tuple[int, ...]:
    if lane_count <= 0:
        return ()
    if view_mode == "source_full_plus_3_partial":
        return (0,)
    if view_mode == "live_5x40":
        return tuple(range(lane_count))
    raise ValueError(f"unsupported balanced view mode: {view_mode}")


def aggregate_dense_lane_predictions(raw_lanes: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_lanes, dtype=float)
    if values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all():
        raise ValueError("dense lane predictions must have shape (lanes, chunks)")
    return np.mean(np.vstack([percentile_rank(row) for row in values]), axis=0)


def score_prebuilt_lanes(
    lanes: Sequence[Sequence[Sequence[dict[str, Any]]]],
    view_mode: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    base, dense_models, dense_keys, weights = _validate(bundle)
    components = _validated_components(base)
    if not lanes:
        return {
            "scores": [],
            "base_scores": [],
            "dense_scores": [],
            "view_mode": "empty",
            "lanes": 0,
        }
    chunk_count = len(lanes[0])
    if chunk_count <= 0 or any(len(lane) != chunk_count for lane in lanes):
        raise ValueError("multi-view hash-bag/dense lane batches are not aligned")
    dense_indices = set(dense_lane_indices(view_mode, len(lanes)))
    base_raw_lanes: list[np.ndarray] = []
    dense_raw_lanes: list[np.ndarray] = []
    for lane_index, lane_chunks in enumerate(lanes):
        rows = [chunk_features(list(chunk or [])) for chunk in lane_chunks]
        base_rows: list[np.ndarray] = []
        for component in components:
            keys = [str(key) for key in component["keys"]]
            matrix = np.asarray(
                [[float(row.get(key, 0.0)) for key in keys] for row in rows],
                dtype=np.float32,
            )
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            base_rows.append(
                model_scores(component["model"], matrix, str(component["score_method"]))
            )
        base_raw_lanes.append(np.vstack(base_rows))
        if lane_index in dense_indices:
            dense_matrix = np.asarray(
                [[float(row.get(key, 0.0)) for key in dense_keys] for row in rows],
                dtype=np.float32,
            )
            dense_matrix = np.nan_to_num(
                dense_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )
            dense_raw_lanes.append(
                np.mean(
                    [
                        np.asarray(model.decision_function(dense_matrix), dtype=float)
                        for model in dense_models
                    ],
                    axis=0,
                )
            )
    base_components = aggregate_component_lane_predictions(np.stack(base_raw_lanes))
    base_scores = np.mean(base_components, axis=0)
    dense_scores = aggregate_dense_lane_predictions(np.vstack(dense_raw_lanes))
    combined = float(weights[0]) * base_scores + float(weights[1]) * dense_scores
    expected = (chunk_count,)
    if (
        base_scores.shape != expected
        or dense_scores.shape != expected
        or combined.shape != expected
        or not np.isfinite(combined).all()
    ):
        raise ValueError("multi-view hash-bag/dense output is invalid")
    return {
        "scores": [float(value) for value in combined],
        "base_scores": [float(value) for value in base_scores],
        "dense_scores": [float(value) for value in dense_scores],
        "base_component_scores": base_components.tolist(),
        "view_mode": view_mode,
        "lanes": len(lanes),
        "dense_lanes": len(dense_raw_lanes),
    }


def score_chunks_detailed(
    chunks: Sequence[Sequence[dict[str, Any]]], bundle: dict[str, Any]
) -> dict[str, Any]:
    if not chunks:
        return {
            "scores": [],
            "base_scores": [],
            "dense_scores": [],
            "view_mode": "empty",
            "lanes": 0,
        }
    lanes, mode = batch_view_lanes(chunks)
    return score_prebuilt_lanes(lanes, mode, bundle)


def score_chunks(
    chunks: Sequence[Sequence[dict[str, Any]]], bundle: dict[str, Any]
) -> list[float]:
    return list(score_chunks_detailed(chunks, bundle)["scores"])


def score_from_file(
    chunks: Sequence[Sequence[dict[str, Any]]], model_path: str | os.PathLike[str]
) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path))
