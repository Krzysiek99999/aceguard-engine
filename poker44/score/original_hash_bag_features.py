"""Single full canonical hand-bag behavioral surface."""
from __future__ import annotations

from typing import Any

from poker44.score.model_view_hand_hash import canonical_hand_bag
from poker44.score.original_redundancy_features import chunk_features as redundancy_features
from poker44.score.original_tree_surface_features import chunk_features as wide_features


HAND_CAP = 60


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    ordered = canonical_hand_bag(list(chunk or []), max_hands=HAND_CAP)
    output = {
        key: float(value)
        for key, value in wide_features(ordered).items()
        if not key.startswith("temporal__")
    }
    output.update(
        {f"redundancy__{key}": float(value) for key, value in redundancy_features(ordered).items()}
    )
    return output
