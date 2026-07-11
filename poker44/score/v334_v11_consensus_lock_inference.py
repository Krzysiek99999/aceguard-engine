"""Inference for the v11 lock7 hybrid with a dual-generation hand-set challenger."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from poker44.score.ensemble_v11 import score_chunks_v11
from poker44.score.original_set_ensemble_inference import (
    load_bundle as load_challenger_bundle,
    score_chunks as score_challenger,
)
from poker44.score.v323_v11_consensus_lock_inference import COMPONENTS, consensus_lock_rank


FAMILY = "v334_v11_consensus_lock7_dual25_top8"


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    bundle = load_challenger_bundle(model_path)
    if bundle.get("family") != FAMILY:
        raise ValueError(f"unexpected v334 family: {bundle.get('family')!r}")
    contract = (
        int(bundle.get("top_n", -1)),
        int(bundle.get("head_n", -1)),
        int(bundle.get("lock_n", -1)),
    )
    if contract != (8, 8, 7):
        raise ValueError("v334 requires top_n=8, head_n=8 and lock_n=7")
    if tuple(bundle.get("consensus_components") or ()) != COMPONENTS:
        raise ValueError("v334 consensus component contract mismatch")
    if float(bundle.get("refresh_weight", -1.0)) != 0.25:
        raise ValueError("v334 requires refresh_weight=0.25")
    if len(bundle.get("components") or []) != 5:
        raise ValueError("v334 requires five positive-weight challenger components")
    return bundle


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
