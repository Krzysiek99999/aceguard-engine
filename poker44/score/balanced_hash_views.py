"""Order-invariant, coverage-balanced views of one player's hand bag."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from statistics import median
from typing import Any, Sequence

from poker44.score.model_view_hand_hash import canonical_hand_bag, model_view_hand_hash


SOURCE_VIEW_COUNT = 4
SOURCE_PARTIAL_COUNT = SOURCE_VIEW_COUNT - 1
LIVE_VIEW_COUNT = 5
LIVE_VIEW_HANDS = 40
LIVE_MODE_MIN_MEDIAN_HANDS = 60
_RING_SALT = b"aceguard-balanced-hash-views-v1\0"


def _ring_order(chunk: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: list[tuple[str, str, dict[str, Any]]] = []
    for hand in chunk:
        if not isinstance(hand, dict):
            continue
        visible_hash = model_view_hand_hash(hand)
        ring_hash = hashlib.sha256(_RING_SALT + visible_hash.encode()).hexdigest()
        keyed.append((ring_hash, visible_hash, hand))
    return [row[2] for row in sorted(keyed, key=lambda row: (row[0], row[1]))]


def balanced_hash_windows(
    chunk: Sequence[dict[str, Any]], *, target_hands: int, view_count: int
) -> list[list[dict[str, Any]]]:
    """Return circular hash-ring windows with complete, near-uniform coverage."""
    rows = _ring_order(chunk)
    if not rows:
        return [[] for _ in range(max(1, int(view_count)))]
    count = max(1, int(view_count))
    target = max(1, min(int(target_hands), len(rows)))
    views: list[list[dict[str, Any]]] = []
    for index in range(count):
        start = (index * len(rows)) // count
        selected = [rows[(start + offset) % len(rows)] for offset in range(target)]
        views.append(canonical_hand_bag(selected))
    return views


def source_unit_views(chunk: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Full natural unit plus three partial views used for source-safe training."""
    full = canonical_hand_bag(list(chunk))
    if len(full) <= 1:
        return [full for _ in range(SOURCE_VIEW_COUNT)]
    target = min(len(full) - 1, max(24, int(math.ceil(0.75 * len(full)))))
    partial = balanced_hash_windows(
        full,
        target_hands=target,
        view_count=SOURCE_PARTIAL_COUNT,
    )
    return [full, *partial]


def live_chunk_views(chunk: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Five balanced 40-hand views; every hand is covered on 80-100 hand live chunks."""
    return balanced_hash_windows(
        chunk,
        target_hands=min(LIVE_VIEW_HANDS, max(1, len(chunk))),
        view_count=LIVE_VIEW_COUNT,
    )


def batch_view_lanes(
    chunks: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[list[list[dict[str, Any]]]], str]:
    """Transpose per-chunk views into batch-aligned lanes for percentile ranking."""
    if not chunks:
        return [], "empty"
    lengths = [len(chunk) for chunk in chunks]
    live_mode = median(lengths) >= LIVE_MODE_MIN_MEDIAN_HANDS
    mode = "live_5x40" if live_mode else "source_full_plus_3_partial"
    per_chunk = [
        live_chunk_views(chunk) if live_mode else source_unit_views(chunk)
        for chunk in chunks
    ]
    lane_count = LIVE_VIEW_COUNT if live_mode else SOURCE_VIEW_COUNT
    if any(len(views) != lane_count for views in per_chunk):
        raise RuntimeError("balanced hash view count changed inside one batch")
    return [
        [per_chunk[chunk_index][lane] for chunk_index in range(len(chunks))]
        for lane in range(lane_count)
    ], mode


def coverage_counts(
    chunk: Sequence[dict[str, Any]], views: Sequence[Sequence[dict[str, Any]]]
) -> dict[str, int]:
    """Count view inclusion per miner-visible hand hash for diagnostics and tests."""
    expected = Counter(model_view_hand_hash(hand) for hand in chunk if isinstance(hand, dict))
    observed = Counter(
        model_view_hand_hash(hand)
        for view in views
        for hand in view
        if isinstance(hand, dict)
    )
    return {key: observed.get(key, 0) for key in expected}
