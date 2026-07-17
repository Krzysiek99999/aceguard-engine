"""Efficient inference for natural-order multisalt sequence view bags."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Sequence

import numpy as np

from poker44.score.natural_order_balanced_views import (
    index_natural_hands,
    salted_natural_order_windows_from_indexed,
)
from poker44.score.original_tree_surface_inference import percentile_rank


VIEWS_PER_SALT = 5
DEFAULT_SALTS = tuple(
    f"aceguard-v394-multisalt-probe-b0-s{index}\0".encode() for index in range(16)
)


def iter_lanes(
    chunks: Sequence[Sequence[dict[str, Any]]],
    salts: Sequence[bytes],
) -> Iterator[list[list[dict[str, Any]]]]:
    indexed_chunks = [index_natural_hands(chunk) for chunk in chunks]
    for salt in salts:
        per_chunk = [
            salted_natural_order_windows_from_indexed(indexed, salt=salt)
            for indexed in indexed_chunks
        ]
        for lane in range(VIEWS_PER_SALT):
            yield [per_chunk[index][lane] for index in range(len(chunks))]


def flatten_lanes(
    lanes: Sequence[Sequence[Sequence[dict[str, Any]]]],
) -> tuple[list[list[dict[str, Any]]], int]:
    if not lanes:
        return [], 0
    chunk_count = len(lanes[0])
    if chunk_count <= 0 or any(len(lane) != chunk_count for lane in lanes):
        raise ValueError("natural multisalt lanes are not batch-aligned")
    return [list(chunk) for lane in lanes for chunk in lane], chunk_count


def unflatten_logits(
    logits: Sequence[float], *, lane_count: int, chunk_count: int
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (int(lane_count) * int(chunk_count),):
        raise ValueError("flattened natural multisalt logits have invalid shape")
    return values.reshape(int(lane_count), int(chunk_count))


def aggregate_lane_ranks(raw_lanes: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(raw_lanes, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all():
        raise ValueError("natural multisalt logits must have shape (lanes, chunks)")
    return np.mean(np.vstack([percentile_rank(row) for row in values]), axis=0)


def score_chunks_detailed(
    chunks: Sequence[Sequence[dict[str, Any]]],
    bundle: dict[str, Any],
    *,
    salts: Sequence[bytes] = DEFAULT_SALTS,
    lanes_per_block: int = 8,
    model_batch_size: int = 64,
    include_raw_lanes: bool = False,
) -> dict[str, Any]:
    if not chunks:
        return {"scores": [], "raw_lanes": [], "lanes": 0}
    if not salts:
        raise ValueError("natural multisalt inference needs at least one salt")
    block_size = max(1, int(lanes_per_block))
    from poker44.score.original_policy_sequence_inference import score_view_logits

    raw_blocks: list[np.ndarray] = []
    pending: list[list[list[dict[str, Any]]]] = []
    for lane in iter_lanes(chunks, salts):
        pending.append(lane)
        if len(pending) < block_size:
            continue
        flat, chunk_count = flatten_lanes(pending)
        logits = score_view_logits(flat, bundle, batch_size=int(model_batch_size))
        raw_blocks.append(
            unflatten_logits(
                logits, lane_count=len(pending), chunk_count=chunk_count
            )
        )
        pending = []
    if pending:
        flat, chunk_count = flatten_lanes(pending)
        logits = score_view_logits(flat, bundle, batch_size=int(model_batch_size))
        raw_blocks.append(
            unflatten_logits(
                logits, lane_count=len(pending), chunk_count=chunk_count
            )
        )
    raw = np.vstack(raw_blocks)
    scores = aggregate_lane_ranks(raw)
    result = {
        "scores": [float(value) for value in scores],
        "lanes": int(raw.shape[0]),
        "chunks": int(raw.shape[1]),
        "lanes_per_block": block_size,
        "model_batch_size": int(model_batch_size),
    }
    if include_raw_lanes:
        result["raw_lanes"] = raw.tolist()
    return result


def score_chunks(
    chunks: Sequence[Sequence[dict[str, Any]]],
    bundle: dict[str, Any],
    **kwargs: Any,
) -> list[float]:
    return list(score_chunks_detailed(chunks, bundle, **kwargs)["scores"])
