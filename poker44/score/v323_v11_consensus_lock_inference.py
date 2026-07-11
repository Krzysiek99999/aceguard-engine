"""Inference for the confidence-aware v11/v321 consensus-lock hybrid."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

from poker44.score.ensemble_v11 import score_chunks_v11
from poker44.score.original_set_ensemble_inference import (
    load_bundle as load_challenger_bundle,
    score_chunks as score_challenger,
)


FAMILY = "v323_v11_consensus_lock8_v321_top10"
COMPONENTS = ("v5", "v6", "v8_markov", "pot_geo", "response_curves")


def rank01(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    return (rankdata(array, method="average") - 1.0) / max(array.size - 1, 1)


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    bundle = load_challenger_bundle(model_path)
    if bundle.get("family") != FAMILY:
        raise ValueError(f"unexpected v323 family: {bundle.get('family')!r}")
    if int(bundle.get("head_n", -1)) != 10 or int(bundle.get("lock_n", -1)) != 8:
        raise ValueError("v323 requires head_n=10 and lock_n=8")
    if tuple(bundle.get("consensus_components") or ()) != COMPONENTS:
        raise ValueError("v323 consensus component contract mismatch")
    if float(bundle.get("v11_score_weight", -1.0)) != 0.0:
        raise ValueError("v323 frozen v11 score weight must be zero")
    return bundle


def consensus_lock_rank(
    v11_scores: Sequence[float],
    challenger_scores: Sequence[float],
    telemetry: Sequence[dict[str, float]],
    *,
    head_n: int,
    lock_n: int,
) -> np.ndarray:
    anchor = np.asarray(v11_scores, dtype=float)
    challenger = np.asarray(challenger_scores, dtype=float)
    components = np.asarray(
        [[float(row.get(name, 0.0)) for name in COMPONENTS] for row in telemetry],
        dtype=float,
    )
    if anchor.shape != challenger.shape or components.shape != (anchor.size, len(COMPONENTS)):
        raise ValueError("v323 score/component shape mismatch")
    if anchor.size == 0:
        return anchor
    if not np.isfinite(anchor).all() or not np.isfinite(challenger).all() or not np.isfinite(components).all():
        raise ValueError("v323 inputs must be finite")

    consensus = np.mean(
        np.column_stack([rank01(components[:, column]) for column in range(components.shape[1])]),
        axis=1,
    )
    anchor_head = np.argsort(-anchor, kind="mergesort")[: max(0, min(int(head_n), anchor.size))]
    locked_count = max(0, min(int(lock_n), len(anchor_head)))
    locked = set(
        int(index)
        for index in anchor_head[
            np.argsort(-consensus[anchor_head], kind="mergesort")[:locked_count]
        ]
    )
    challenger_order = [int(index) for index in np.argsort(-challenger, kind="mergesort")]
    order = [index for index in challenger_order if index in locked]
    order.extend(index for index in challenger_order if index not in locked)
    output = np.empty(anchor.size, dtype=float)
    output[np.asarray(order, dtype=int)] = np.linspace(1.0, 0.0, anchor.size, dtype=float)
    return output


def score_chunks(
    chunks: Sequence[Any],
    bundle: dict[str, Any],
    *,
    batch_size: int = 32,
) -> list[float]:
    if not chunks:
        return []
    anchor, telemetry, _types = score_chunks_v11(list(chunks))
    challenger = score_challenger(chunks, bundle, batch_size=batch_size)
    scores = consensus_lock_rank(
        anchor,
        challenger,
        telemetry,
        head_n=int(bundle["head_n"]),
        lock_n=int(bundle["lock_n"]),
    )
    return [float(value) for value in scores]


def score_from_file(
    chunks: Sequence[Any],
    model_path: str | os.PathLike[str],
    *,
    batch_size: int = 32,
) -> list[float]:
    return score_chunks(chunks, load_bundle(Path(model_path)), batch_size=batch_size)
