"""Serve-side inference for the AceGuard supervised schema model.

The model consumes miner-visible Poker44 hand chunks and returns one raw risk
score per chunk. Deployment code applies a rank cap such as top1, top2, or top3
depending on the miner slot.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

from poker44.score.robust_schema.features import chunk_features as schema_features
from poker44.score.statistical_v25 import compute_features as v25_features

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def load_bundle(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = str(path)
    mtime = os.path.getmtime(p)
    cached = _CACHE.get(p)
    if cached is None or cached[0] != mtime:
        with open(p, "rb") as handle:
            _CACHE[p] = (mtime, pickle.load(handle))
    return _CACHE[p][1]


def _unwrap(chunks: Sequence[Any]) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            chunk = chunk.get("hands", chunk.get("chunks", chunk))
        out.append(chunk if isinstance(chunk, list) else [])
    return out


def _feature_dict(chunk: list[dict[str, Any]], feature_set: str) -> dict[str, float]:
    if feature_set == "v25":
        return {f"v25__{k}": float(v) for k, v in v25_features(chunk).items()}
    if feature_set in {"schema", "robust_schema"}:
        return {f"schema__{k}": float(v) for k, v in schema_features(chunk).items()}
    if feature_set == "super":
        out = {f"schema__{k}": float(v) for k, v in schema_features(chunk).items()}
        out.update({f"v25__{k}": float(v) for k, v in v25_features(chunk).items()})
        return out
    raise ValueError(f"unknown feature_set={feature_set}")


def _mat(feats: list[dict[str, float]], keys: list[str]) -> np.ndarray:
    arr = np.array([[feat.get(key, np.nan) for key in keys] for feat in feats], dtype=float)
    if arr.size == 0:
        return arr
    col_means = np.nanmean(np.where(np.isnan(arr), np.nan, arr), axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    idx = np.where(np.isnan(arr))
    arr[idx] = np.take(col_means, idx[1])
    return arr


def _batch_z(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (arr - mu) / sd


def _build_x(chunks: Sequence[Any], bundle: dict[str, Any]) -> np.ndarray:
    feature_set = str(bundle.get("feature_set") or "super")
    keys = list(bundle["keys"])
    mask = np.asarray(bundle["abs_stable_mask"], dtype=bool)
    hands = _unwrap(chunks)
    feats = [_feature_dict(chunk, feature_set) for chunk in hands]
    arr = _mat(feats, keys)
    return np.hstack([arr[:, mask], _batch_z(arr)])


def score_chunks(
    chunks: Sequence[Any],
    bundle: dict[str, Any],
    strategy: str = "rank_mean",
) -> list[float]:
    if not chunks:
        return []
    X = _build_x(chunks, bundle)
    models = bundle["models"]
    strategy = (strategy or "rank_mean").lower()

    if strategy in models:
        pred = models[strategy].predict_proba(X)[:, 1]
        return [float(np.clip(v, 0.0, 1.0)) for v in pred]

    names = list(bundle.get("stack_model_names") or sorted(models))
    base = np.column_stack([models[name].predict_proba(X)[:, 1] for name in names])

    if strategy == "stack":
        pred = bundle["meta"].predict_proba(base)[:, 1]
        return [float(np.clip(v, 0.0, 1.0)) for v in pred]

    if strategy == "avg":
        pred = np.mean(base, axis=1)
        return [float(np.clip(v, 0.0, 1.0)) for v in pred]

    if strategy == "rank_mean":
        ranked = [rankdata(base[:, i]) / max(len(base), 1) for i in range(base.shape[1])]
        pred = np.mean(ranked, axis=0)
        return [float(np.clip(v, 0.0, 1.0)) for v in pred]

    raise ValueError(f"unknown v112_super strategy={strategy}")


def score_from_file(
    chunks: Sequence[Any],
    model_path: str | os.PathLike[str],
    strategy: str = "rank_mean",
) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path), strategy=strategy)


def is_available(model_path: str | os.PathLike[str] | None = None) -> bool:
    if model_path is None:
        model_path = Path(__file__).resolve().parents[2] / "data" / "models" / "v112_super" / "model.pkl"
    return Path(model_path).exists()
