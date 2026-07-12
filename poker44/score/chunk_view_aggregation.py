"""Deterministic same-player chunk views for source-sized tabular models."""
from __future__ import annotations

from statistics import median
from typing import Any, Sequence


VALID_VIEW_MODES = (
    "full",
    "sliding40_mean3",
    "sliding40_mean5",
    "sliding40_median5",
    "partition35_mean",
)


def chunk_views(chunk: Sequence[dict[str, Any]], mode: str) -> list[list[dict[str, Any]]]:
    rows = list(chunk)
    if mode not in VALID_VIEW_MODES:
        raise ValueError(f"unsupported chunk view mode: {mode}")
    if mode == "full" or len(rows) <= 40:
        return [rows]
    if mode.startswith("sliding40_"):
        count = 3 if mode.endswith("3") else 5
        span = len(rows) - 40
        starts = sorted({int(round(index * span / max(count - 1, 1))) for index in range(count)})
        return [rows[start : start + 40] for start in starts]
    if mode == "partition35_mean":
        count = max(1, int(round(len(rows) / 35.0)))
        boundaries = [int(round(index * len(rows) / count)) for index in range(count + 1)]
        return [rows[boundaries[index] : boundaries[index + 1]] for index in range(count)]
    raise ValueError(f"unsupported chunk view mode: {mode}")


def expand_chunk_views(
    chunks: Sequence[Sequence[dict[str, Any]]],
    mode: str,
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    views: list[list[dict[str, Any]]] = []
    owners: list[int] = []
    for owner, chunk in enumerate(chunks):
        current = chunk_views(chunk, mode)
        views.extend(current)
        owners.extend([owner] * len(current))
    return views, owners


def reduce_view_scores(
    scores: Sequence[float],
    owners: Sequence[int],
    chunk_count: int,
    mode: str,
) -> list[float]:
    if len(scores) != len(owners):
        raise ValueError("view scores and owners have different lengths")
    grouped: list[list[float]] = [[] for _ in range(chunk_count)]
    for score, owner in zip(scores, owners, strict=True):
        if owner < 0 or owner >= chunk_count:
            raise ValueError(f"view owner is outside chunk range: {owner}")
        grouped[owner].append(float(score))
    if any(not values for values in grouped):
        raise ValueError("at least one chunk has no scored views")
    if mode == "sliding40_median5":
        return [float(median(values)) for values in grouped]
    return [float(sum(values) / len(values)) for values in grouped]
