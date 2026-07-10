"""Inference for independently implemented AceGuard LambdaMART bundles."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

from poker44.score.original_behavior_features import chunk_features


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"{model_path} did not contain a dict bundle")
    return bundle


def _feature_rows(chunks: Sequence[Any]) -> list[dict[str, float]]:
    return [chunk_features(list(chunk or [])) for chunk in chunks]


def _matrix(rows: list[dict[str, float]], keys: Sequence[str]) -> np.ndarray:
    x = np.asarray(
        [[float(row.get(key, 0.0)) for key in keys] for row in rows],
        dtype=float,
    )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def score_chunks(chunks: Sequence[Any], bundle: dict[str, Any]) -> list[float]:
    if not chunks:
        return []
    rows = _feature_rows(chunks)
    children = list(bundle.get("seed_children") or [])
    if children:
        ranked: list[np.ndarray] = []
        weights: list[float] = []
        for child in children:
            keys = list(child.get("keys") or [])
            model = child.get("model")
            weight = float(child.get("weight", 0.0))
            if not keys or model is None or weight <= 0.0:
                continue
            raw = np.asarray(model.predict(_matrix(rows, keys)), dtype=float)
            ranked.append(rankdata(raw, method="average") / max(len(raw), 1))
            weights.append(weight)
        if not ranked or float(sum(weights)) <= 1e-12:
            raise ValueError("original ensemble has no usable positive-weight children")
        normalized = np.asarray(weights, dtype=float) / float(sum(weights))
        scores = np.column_stack(ranked) @ normalized
        return [float(value) for value in scores]

    keys = list(bundle.get("keys") or [])
    model = bundle.get("model")
    if not keys or model is None:
        raise ValueError("original LambdaMART bundle has no model/keys or seed children")
    return [float(value) for value in np.asarray(model.predict(_matrix(rows, keys)), dtype=float)]


def score_from_file(chunks: Sequence[Any], model_path: str | os.PathLike[str]) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path))
