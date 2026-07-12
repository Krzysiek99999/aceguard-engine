"""Frozen base tree surface plus an independent HERO-only policy surface."""
from __future__ import annotations

from typing import Any

from poker44.score.original_hero_policy_surface_features import (
    chunk_features as hero_policy_features,
)
from poker44.score.original_tree_surface_features import chunk_features as base_features


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    out = dict(base_features(chunk))
    out.update(
        {
            f"hero__{key}": float(value)
            for key, value in hero_policy_features(chunk).items()
        }
    )
    return out


__all__ = ["chunk_features"]
