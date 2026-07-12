"""Inference for the natural-unit original tree-surface challenger."""
from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from poker44.score.chunk_view_aggregation import expand_chunk_views, reduce_view_scores
from poker44.score.original_tree_surface_features import chunk_features


FEATURE_EXTRACTOR = "poker44.score.original_tree_surface_features:chunk_features"
VIEW_MODE = "partition35_mean"
BLEND_STRATEGY = "component_percentile_rank_mean"


def percentile_rank(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return permutation-equivariant percentile ranks with average ranks for ties."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("percentile_rank expects one-dimensional values")
    if not np.isfinite(array).all():
        raise ValueError("percentile_rank received non-finite values")
    if array.size == 0:
        return np.asarray([], dtype=float)
    if array.size == 1:
        return np.asarray([0.5], dtype=float)

    order = np.argsort(array, kind="mergesort")
    ordered = array[order]
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and ordered[stop] == ordered[start]:
            stop += 1
        average_rank = 0.5 * float(start + stop - 1)
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks / float(array.size - 1)


def blend_component_predictions(component_predictions: np.ndarray) -> np.ndarray:
    """Percentile-rank each component inside the current batch, then mean."""
    values = np.asarray(component_predictions, dtype=float)
    if values.ndim != 2:
        raise ValueError("component predictions must have shape (components, batch)")
    if values.shape[0] == 0:
        raise ValueError("at least one component prediction is required")
    if values.shape[1] == 0:
        return np.asarray([], dtype=float)
    ranked = np.vstack([percentile_rank(row) for row in values])
    return np.mean(ranked, axis=0)


def model_scores(model: Any, matrix: np.ndarray, score_method: str) -> np.ndarray:
    """Read a positive-class score from a serialized surface model."""
    if score_method == "predict":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, but LGBMRanker was fitted "
                    "with feature names"
                ),
                category=UserWarning,
            )
            scores = np.asarray(model.predict(matrix), dtype=float).reshape(-1)
    elif score_method == "predict_proba":
        probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
        if probabilities.ndim != 2:
            raise ValueError("predict_proba did not return a two-dimensional matrix")
        classes = np.asarray(getattr(model, "classes_", []))
        positive = np.flatnonzero(classes == 1)
        if positive.size != 1 or int(positive[0]) >= probabilities.shape[1]:
            raise ValueError("classifier bundle has no unique positive class")
        scores = probabilities[:, int(positive[0])].reshape(-1)
    else:
        raise ValueError(f"unsupported component score method: {score_method}")
    if scores.shape != (len(matrix),):
        raise ValueError("component model returned the wrong number of scores")
    if not np.isfinite(scores).all():
        raise ValueError("component model returned non-finite scores")
    return scores


def _validated_components(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(bundle, dict):
        raise TypeError("tree-surface bundle must be a dict")
    extractor = str(bundle.get("feature_extractor") or FEATURE_EXTRACTOR)
    if extractor != FEATURE_EXTRACTOR:
        raise ValueError(f"unsupported tree-surface feature extractor: {extractor}")
    view_mode = str(bundle.get("view_mode") or VIEW_MODE)
    if view_mode != VIEW_MODE:
        raise ValueError(f"tree-surface inference requires {VIEW_MODE}, got {view_mode}")
    strategy = str(bundle.get("blend_strategy") or BLEND_STRATEGY)
    if strategy != BLEND_STRATEGY:
        raise ValueError(f"unsupported tree-surface blend strategy: {strategy}")

    components = list(bundle.get("components") or [])
    if not components:
        raise ValueError("tree-surface bundle has no components")
    names: set[str] = set()
    for component in components:
        if not isinstance(component, dict):
            raise TypeError("tree-surface component must be a dict")
        name = str(component.get("name") or "")
        keys = [str(key) for key in component.get("keys") or []]
        if not name or name in names:
            raise ValueError("tree-surface component names must be non-empty and unique")
        if not keys or len(keys) != len(set(keys)):
            raise ValueError(f"tree-surface component {name} has invalid feature keys")
        if "model" not in component:
            raise ValueError(f"tree-surface component {name} has no model")
        if str(component.get("score_method") or "") not in {"predict", "predict_proba"}:
            raise ValueError(f"tree-surface component {name} has invalid score_method")
        names.add(name)
    return components


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        bundle = pickle.load(handle)
    _validated_components(bundle)
    return bundle


def component_raw_scores(
    chunks: Sequence[Sequence[dict[str, Any]]],
    bundle: dict[str, Any],
) -> np.ndarray:
    """Score views and reduce each component before any cross-component rank blend."""
    components = _validated_components(bundle)
    if not chunks:
        return np.empty((len(components), 0), dtype=float)

    views, owners = expand_chunk_views([list(chunk or []) for chunk in chunks], VIEW_MODE)
    feature_rows = [chunk_features(view) for view in views]
    reduced_components: list[np.ndarray] = []
    for component in components:
        keys = [str(key) for key in component["keys"]]
        matrix = np.asarray(
            [[float(row.get(key, 0.0)) for key in keys] for row in feature_rows],
            dtype=np.float32,
        )
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        view_scores = model_scores(
            component["model"], matrix, str(component["score_method"])
        )
        reduced = reduce_view_scores(view_scores, owners, len(chunks), VIEW_MODE)
        reduced_components.append(np.asarray(reduced, dtype=float))
    return np.vstack(reduced_components)


def score_chunks(
    chunks: Sequence[Sequence[dict[str, Any]]],
    bundle: dict[str, Any],
) -> list[float]:
    """Score one live batch using per-component ranks over its original chunks."""
    components = _validated_components(bundle)
    if not chunks:
        return []
    raw = component_raw_scores(chunks, bundle)
    if raw.shape[0] != len(components):
        raise RuntimeError("tree-surface component count changed during inference")
    return [float(value) for value in blend_component_predictions(raw)]


def score_from_file(
    chunks: Sequence[Sequence[dict[str, Any]]],
    model_path: str | os.PathLike[str],
) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path))
