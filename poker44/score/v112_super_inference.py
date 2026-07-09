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


def _cap_actions_per_hand(
    chunks: list[list[dict[str, Any]]],
    max_actions: int,
) -> list[list[dict[str, Any]]]:
    if max_actions <= 0:
        return chunks
    capped_chunks: list[list[dict[str, Any]]] = []
    for chunk in chunks:
        capped_chunk: list[dict[str, Any]] = []
        for hand in chunk:
            if not isinstance(hand, dict):
                continue
            actions = hand.get("actions")
            if isinstance(actions, list) and len(actions) > max_actions:
                capped = dict(hand)
                capped["actions"] = list(actions[:max_actions])
                capped_chunk.append(capped)
            else:
                capped_chunk.append(hand)
        capped_chunks.append(capped_chunk)
    return capped_chunks


def _v11_feature_dict(chunk: list[dict[str, Any]]) -> dict[str, float]:
    from poker44.score.ensemble_v11 import score_chunk_v11

    score, telemetry, chunk_type = score_chunk_v11(chunk)
    out = {"v11__score": float(score)}
    for key, value in telemetry.items():
        out[f"v11__{key}"] = float(value)
    for name in ("short", "mixed", "long"):
        out[f"v11__type_{name}"] = 1.0 if str(chunk_type) == name else 0.0
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
    if feature_set in {
        "behav_mix",
        "v131",
        "behav_ngram",
        "v132",
        "behav_ngram_response",
        "v157",
        "behav_mix_v11",
        "v234",
    }:
        from poker44.score.enterprise_features import compute_enterprise_features
        from poker44.score.extended_features import compute_extended_features
        from poker44.score.features_pot_geometry import extract_pot_geometry_features
        from poker44.score.features_response_curves import extract_response_curve_features
        from poker44.score.features_v13_safe import chunk_features_v13

        out = {f"schema__{k}": float(v) for k, v in schema_features(chunk).items()}
        out.update({f"v25__{k}": float(v) for k, v in v25_features(chunk).items()})
        out.update({f"seq__{k}": float(v) for k, v in sequence_features(chunk).items()})
        out.update(
            {f"geo__{k}": float(v) for k, v in extract_pot_geometry_features(chunk).items()}
        )
        out.update({f"v13__{k}": float(v) for k, v in chunk_features_v13(chunk).items()})
        out.update({f"ext__{k}": float(v) for k, v in compute_extended_features(chunk).items()})
        out.update({f"ent__{k}": float(v) for k, v in compute_enterprise_features(chunk).items()})
        if feature_set in {"behav_ngram_response", "v157"}:
            out.update(
                {f"resp__{k}": float(v) for k, v in extract_response_curve_features(chunk).items()}
            )
        if feature_set in {"behav_mix_v11", "v234"}:
            out.update(_v11_feature_dict(chunk))
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
    hands = _cap_actions_per_hand(hands, int(bundle.get("serve_max_actions_per_hand") or 0))
    feats = [_feature_dict(chunk, feature_set) for chunk in hands]
    hand_feature_model = bundle.get("hand_feature_model")
    if hand_feature_model is not None:
        hand_feature_rows = hand_feature_model.chunk_features_many(hands)
        for feat, hand_feat in zip(feats, hand_feature_rows, strict=False):
            feat.update({str(key): float(value) for key, value in hand_feat.items()})
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


def _rank01(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    return rankdata(arr, method="average") / max(len(arr), 1)


def _shape_scores(values: Sequence[float], config: dict[str, Any]) -> np.ndarray:
    """Monotone score head that moves the fixed 0.5 cutoff."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    threshold = float(config.get("threshold", config.get("t_star", 0.5)))
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    sharpness = float(config.get("sharpness", 14.0))
    clipped = np.clip(arr, 0.0, 1.0)
    core = 1.0 / (1.0 + np.exp(-np.clip(sharpness * (clipped - threshold), -60.0, 60.0)))
    shaped = 0.998 * core + 0.002 * clipped
    return np.clip(shaped, 0.0, 1.0)


def _rank_ladder_scores(values: Sequence[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    order = np.argsort(-arr, kind="mergesort")
    out = np.zeros(arr.size, dtype=float)
    if arr.size == 1:
        out[int(order[0])] = 0.75
        return [float(v) for v in out]
    for rank, idx in enumerate(order):
        out[int(idx)] = 0.95 - (rank / (arr.size - 1)) * 0.90
    return [float(np.clip(v, 0.0, 1.0)) for v in out]


def _score_child(
    chunks: list[list[dict[str, Any]]],
    child: dict[str, Any],
) -> np.ndarray:
    callable_name = str(child.get("callable") or "").lower()
    if callable_name == "v11":
        from poker44.score.ensemble_v11 import score_chunks_v11

        raw, _telemetry, _types = score_chunks_v11(chunks)
        return np.asarray(raw, dtype=float)

    child_bundle = child.get("bundle")
    if child_bundle is None and child.get("model_path"):
        child_bundle = load_bundle(child["model_path"])
    if child_bundle is None:
        raise ValueError("child is missing embedded bundle/model_path/callable")
    strategy = str(child.get("strategy") or "rank_mean")
    return np.asarray(score_chunks(chunks, child_bundle, strategy=strategy), dtype=float)


def _score_blend_children(
    chunks: list[list[dict[str, Any]]],
    bundle: dict[str, Any],
) -> list[float]:
    children = list(bundle.get("blend_children") or [])
    if not children:
        raise ValueError("blend bundle has no children")
    mode = str(bundle.get("blend_mode") or "rank_mean").lower()
    weighted: list[np.ndarray] = []
    weights: list[float] = []
    for child in children:
        scores = _score_child(chunks, child)
        if mode in {"rank_mean", "rank_space", "rank"}:
            scores = _rank01(scores)
        weighted.append(scores)
        weights.append(float(child.get("weight", 1.0)))
    if not weighted:
        return [0.5 for _ in chunks]
    matrix = np.column_stack(weighted)
    weight_arr = np.asarray(weights, dtype=float)
    if float(np.sum(weight_arr)) <= 1e-12:
        weight_arr = np.ones_like(weight_arr)
    pred = matrix @ (weight_arr / float(np.sum(weight_arr)))
    return [float(np.clip(v, 0.0, 1.0)) for v in pred]


def _score_toplock_children(
    chunks: list[list[dict[str, Any]]],
    bundle: dict[str, Any],
) -> list[float]:
    config = dict(bundle.get("top_lock") or {})
    anchor = dict(config.get("anchor") or {"callable": "v11"})
    lock_sequence = list(config.get("lock_sequence") or [])
    rest_children = list(config.get("rest_children") or [])
    if not rest_children:
        raise ValueError("top_lock bundle has no rest_children")
    lock_n = max(0, int(config.get("lock_n", 2)))

    if lock_sequence:
        first_child = dict(lock_sequence[0].get("child") or lock_sequence[0])
        n = int(_score_child(chunks, first_child).size)
    else:
        anchor_scores = _score_child(chunks, anchor)
        n = int(anchor_scores.size)
    if n == 0:
        return []

    rest_signal = np.zeros(n, dtype=float)
    total_weight = 0.0
    for child in rest_children:
        child_scores = _score_child(chunks, child)
        rest_signal += float(child.get("weight", 1.0)) * _rank01(child_scores)
        total_weight += float(child.get("weight", 1.0))
    if total_weight > 1e-12:
        rest_signal /= total_weight

    locked: list[int] = []
    locked_set: set[int] = set()
    if lock_sequence:
        for step in lock_sequence:
            if not isinstance(step, dict):
                continue
            child = dict(step.get("child") or step)
            step_lock_n = max(0, int(step.get("lock_n", 1)))
            child_scores = _score_child(chunks, child)
            if int(child_scores.size) != n:
                raise RuntimeError(
                    f"top_lock lock_sequence child returned {child_scores.size} scores for {n} chunks"
                )
            selected = 0
            for idx in np.argsort(-child_scores, kind="mergesort"):
                int_idx = int(idx)
                if int_idx in locked_set:
                    continue
                locked.append(int_idx)
                locked_set.add(int_idx)
                selected += 1
                if selected >= step_lock_n or len(locked) >= n:
                    break
    else:
        anchor_order = list(np.argsort(-anchor_scores, kind="mergesort"))
        locked = [int(idx) for idx in anchor_order[: min(lock_n, n)]]
        locked_set = set(locked)
    rest_order = [
        int(idx)
        for idx in np.argsort(-rest_signal, kind="mergesort")
        if int(idx) not in locked_set
    ]
    order = locked + rest_order
    out = np.zeros(n, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = 1.0 - (rank / max(n - 1, 1))
    return [float(np.clip(v, 0.0, 1.0)) for v in out]


def _score_strategy_blend(
    chunks: list[list[dict[str, Any]]],
    bundle: dict[str, Any],
    strategy: str,
) -> list[float]:
    blend_specs = bundle.get("strategy_blend_weights") or {}
    weights_by_strategy = blend_specs.get(strategy)
    if not isinstance(weights_by_strategy, dict) or not weights_by_strategy:
        raise ValueError(f"unknown strategy blend={strategy}")
    pieces: list[np.ndarray] = []
    weights: list[float] = []
    for child_strategy, weight in sorted(weights_by_strategy.items()):
        child_strategy = str(child_strategy)
        if child_strategy == strategy:
            raise ValueError(f"strategy blend {strategy} cannot reference itself")
        weight = float(weight)
        if weight <= 0.0:
            continue
        child_scores = np.asarray(score_chunks(chunks, bundle, strategy=child_strategy), dtype=float)
        pieces.append(_rank01(child_scores))
        weights.append(weight)
    if not pieces:
        return [0.5 for _ in chunks]
    matrix = np.column_stack(pieces)
    weight_arr = np.asarray(weights, dtype=float)
    pred = matrix @ (weight_arr / float(np.sum(weight_arr)))
    return [float(np.clip(v, 0.0, 1.0)) for v in pred]


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
    if strategy.startswith("ladder_"):
        base_strategy = strategy.removeprefix("ladder_") or "rank_mean"
        raw = score_chunks(chunks, bundle, strategy=base_strategy)
        return _rank_ladder_scores(raw)
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
    original_chunks = _cap_actions_per_hand(
        original_chunks,
        int(bundle.get("serve_max_actions_per_hand") or 0),
    )
    if strategy in (bundle.get("strategy_blend_weights") or {}):
        return _score_strategy_blend(original_chunks, bundle, strategy)
    if bundle.get("top_lock"):
        return _score_toplock_children(original_chunks, bundle)
    if bundle.get("blend_children"):
        return _score_blend_children(original_chunks, bundle)
    score_shapes = bundle.get("score_shapes_by_strategy") or {}
    if strategy in score_shapes:
        config = dict(score_shapes[strategy] or {})
        base_strategy = str(config.get("base_strategy") or "").lower()
        if not base_strategy or base_strategy == strategy:
            raise ValueError(f"invalid shaped strategy base for {strategy}")
        raw = score_chunks(original_chunks, bundle, strategy=base_strategy)
        shaped = _shape_scores(raw, config)
        return [float(v) for v in shaped]

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
