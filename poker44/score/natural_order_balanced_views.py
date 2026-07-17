"""Deterministic hash-balanced subsets rendered in natural hand order."""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

from poker44.score.model_view_hand_hash import model_view_hand_hash


IndexedNaturalHand = tuple[int, str, dict[str, Any]]


def index_natural_hands(
    chunk: Sequence[dict[str, Any]],
) -> list[IndexedNaturalHand]:
    return [
        (index, model_view_hand_hash(hand), hand)
        for index, hand in enumerate(chunk)
        if isinstance(hand, dict)
    ]


def salted_natural_order_windows_from_indexed(
    indexed: Sequence[IndexedNaturalHand],
    *,
    salt: bytes,
    target_hands: int = 40,
    view_count: int = 5,
) -> list[list[dict[str, Any]]]:
    """Build salted views from hashes cached once for the whole request."""
    count = max(1, int(view_count))
    if not indexed:
        return [[] for _ in range(count)]
    ring = sorted(
        indexed,
        key=lambda row: (
            hashlib.sha256(salt + row[1].encode()).hexdigest(),
            row[1],
            row[0],
        ),
    )
    target = max(1, min(int(target_hands), len(ring)))
    views: list[list[dict[str, Any]]] = []
    for lane in range(count):
        start = (lane * len(ring)) // count
        selected = [ring[(start + offset) % len(ring)] for offset in range(target)]
        views.append([row[2] for row in sorted(selected, key=lambda row: row[0])])
    return views


def salted_natural_order_windows(
    chunk: Sequence[dict[str, Any]],
    *,
    salt: bytes,
    target_hands: int = 40,
    view_count: int = 5,
) -> list[list[dict[str, Any]]]:
    """Select balanced hash-ring windows without destroying temporal order."""
    return salted_natural_order_windows_from_indexed(
        index_natural_hands(chunk),
        salt=salt,
        target_hands=target_hands,
        view_count=view_count,
    )
