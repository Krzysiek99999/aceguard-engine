"""Serve-side inference for the AceGuard supervised schema model.

The model consumes miner-visible Poker44 hand chunks and returns one raw risk
score per chunk. Deployment code applies a rank cap such as top1, top2, or top3
depending on the miner slot.
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

from poker44.score.robust_schema.features import chunk_features as schema_features
from poker44.score.sequence_schema import chunk_features as sequence_features
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
    if feature_set == "seq":
        return {f"seq__{k}": float(v) for k, v in sequence_features(chunk).items()}
    if feature_set in {"super_seq", "v115"}:
        out = {f"schema__{k}": float(v) for k, v in schema_features(chunk).items()}
        out.update({f"v25__{k}": float(v) for k, v in v25_features(chunk).items()})
        out.update({f"seq__{k}": float(v) for k, v in sequence_features(chunk).items()})
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
    feature_mode = str(bundle.get("feature_mode") or "abs_batch")
    keys = list(bundle["keys"])
    mask = np.asarray(bundle["abs_stable_mask"], dtype=bool)
    hands = _unwrap(chunks)
    feats = [_feature_dict(chunk, feature_set) for chunk in hands]
    arr = _mat(feats, keys)
    pieces: list[np.ndarray] = []
    if feature_mode in {"abs_batch", "abs_only"}:
        pieces.append(arr[:, mask])
    if feature_mode in {"abs_batch", "batch_only"}:
        pieces.append(_batch_z(arr))
    if not pieces:
        X = np.hstack([arr[:, mask], _batch_z(arr)])
    else:
        X = np.hstack(pieces)
    feature_indices = bundle.get("feature_indices")
    if feature_indices is not None:
        idx = np.asarray(feature_indices, dtype=int)
        if idx.size:
            X = X[:, idx]
    return X


def _predict_model_scores(model: Any, X: np.ndarray, chunks: list[list[dict[str, Any]]]) -> np.ndarray:
    if hasattr(model, "predict_chunk_scores"):
        return np.asarray(model.predict_chunk_scores(chunks), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
    return np.asarray(model.predict(X), dtype=float)


def _base_score_matrix(
    models: dict[str, Any],
    names: list[str],
    X: np.ndarray,
    chunks: list[list[dict[str, Any]]],
) -> np.ndarray:
    return np.column_stack(
        [np.clip(_predict_model_scores(models[name], X, chunks), 0.0, 1.0) for name in names]
    )


def _split_chunk(chunk: list[dict[str, Any]], segment_size: int) -> list[list[dict[str, Any]]]:
    if segment_size <= 0 or len(chunk) <= segment_size:
        return [chunk]
    n_segments = max(1, int(round(len(chunk) / float(segment_size))))
    boundaries = np.linspace(0, len(chunk), n_segments + 1)
    segments = [
        list(chunk[int(round(boundaries[idx])) : int(round(boundaries[idx + 1]))])
        for idx in range(n_segments)
    ]
    segments = [segment for segment in segments if segment]
    return segments


def _aggregate_segment_scores(values: list[float], mode: str) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.5
    if mode == "max":
        return float(np.max(arr))
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "q75":
        return float(np.quantile(arr, 0.75))
    if mode == "top2mean":
        return float(np.mean(np.sort(arr)[-min(2, len(arr)) :]))
    raise ValueError(f"unknown segment aggregation mode={mode}")


def _parse_segment_strategy(strategy: str) -> tuple[int, str, str] | None:
    match = re.fullmatch(r"seg(\d+)_(max|mean|q75|top2mean)_(.+)", strategy)
    if not match:
        return None
    return int(match.group(1)), match.group(2), match.group(3)


def score_chunks(
    chunks: Sequence[Any],
    bundle: dict[str, Any],
    strategy: str = "rank_mean",
) -> list[float]:
    if not chunks:
        return []
    strategy = (strategy or "rank_mean").lower()
    segment = _parse_segment_strategy(strategy)
    if segment is not None:
        segment_size, aggregate_mode, inner_strategy = segment
        original_chunks = _unwrap(chunks)
        expanded: list[list[dict[str, Any]]] = []
        owners: list[int] = []
        for owner, chunk in enumerate(original_chunks):
            segments = _split_chunk(chunk, segment_size)
            expanded.extend(segments)
            owners.extend([owner] * len(segments))
        segment_scores = score_chunks(expanded, bundle, strategy=inner_strategy)
        grouped: list[list[float]] = [[] for _ in original_chunks]
        if len(segment_scores) != len(owners):
            raise RuntimeError(
                f"segment scorer returned {len(segment_scores)} scores for {len(owners)} segments"
            )
        for owner, score in zip(owners, segment_scores, strict=True):
            grouped[owner].append(float(score))
        fallback: list[float] | None = None
        out: list[float] = []
        for idx, values in enumerate(grouped):
            if values:
                score = _aggregate_segment_scores(values, aggregate_mode)
            else:
                if fallback is None:
                    fallback = score_chunks(original_chunks, bundle, strategy=inner_strategy)
                score = fallback[idx]
            out.append(float(np.clip(score, 0.0, 1.0)))
        return out

    original_chunks = _unwrap(chunks)
    X = _build_x(original_chunks, bundle)
    models = bundle["models"]

    if strategy in models:
        pred = _predict_model_scores(models[strategy], X, original_chunks)
        return [float(np.clip(v, 0.0, 1.0)) for v in pred]

    names = list(bundle.get("stack_model_names") or sorted(models))
    base = _base_score_matrix(models, names, X, original_chunks)

    blend_weights = bundle.get("blend_weights_by_strategy") or {}
    if strategy in blend_weights:
        weights_by_name = blend_weights[strategy] or {}
        weights = np.asarray([float(weights_by_name.get(name, 0.0)) for name in names], dtype=float)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            raise ValueError(f"empty blend weights for strategy={strategy}")
        pred = base @ (weights / weight_sum)
        return [float(np.clip(v, 0.0, 1.0)) for v in pred]

    if strategy.startswith("avg_no_"):
        dropped = strategy.removeprefix("avg_no_")
        keep = [idx for idx, name in enumerate(names) if name != dropped]
        if not keep:
            raise ValueError(f"avg_no strategy removed all model heads: {strategy}")
        pred = np.mean(base[:, keep], axis=1)
        return [float(np.clip(v, 0.0, 1.0)) for v in pred]

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
