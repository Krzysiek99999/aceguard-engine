"""Order-robust full-chunk and multi-subset behavioral surface."""
from __future__ import annotations

from typing import Any

import numpy as np

from poker44.score.model_view_hand_hash import canonical_hand_bag
from poker44.score.original_redundancy_features import chunk_features as redundancy_features
from poker44.score.original_tree_surface_features import chunk_features as wide_features


FULL_HAND_CAP = 48
BUCKETS = 3
BUCKET_HAND_CAP = 16
STATISTICS = ("mean", "std", "min", "max", "range")


def _base_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    output = {
        key: float(value)
        for key, value in wide_features(chunk).items()
        if not key.startswith("temporal__")
    }
    output.update(
        {f"redundancy__{key}": float(value) for key, value in redundancy_features(chunk).items()}
    )
    return output


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize one natural player chunk without using source hand order."""
    ordered = canonical_hand_bag(list(chunk or []))
    full = _base_features(ordered[:FULL_HAND_CAP])
    subsets = [
        _base_features(ordered[index::BUCKETS][:BUCKET_HAND_CAP])
        for index in range(BUCKETS)
    ]
    keys = sorted(set(full).union(*(set(row) for row in subsets)))
    output = {f"full__{key}": float(full.get(key, 0.0)) for key in keys}
    for key in keys:
        values = np.asarray([row.get(key, 0.0) for row in subsets], dtype=float)
        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
        }
        for name in STATISTICS:
            output[f"bucket__{name}__{key}"] = stats[name]
    return output
