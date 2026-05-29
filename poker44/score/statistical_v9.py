"""V9 — type-calibrated detector (codex iter 13/16 #1 priority).

Live payload alternates two variants (deterministic chunk → fixed window):
  - LONG  : ~12 actions/hand (single-action source, fixed window=12)
  - SHORT : 3-8 actions/hand variable (multi-action source, window 6-10)

V5/V6/V8 work best on LONG (more action transitions, more sizing data, stronger
repetition signal). On SHORT they get noisier — same features but less data.

V9 detects type from `actions_per_hand` distribution, then applies different
feature weights + score floor + max_n per type. Designed to match payload bifurcation
the way UID 211 likely does (R5=R7 spike pattern observed).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from poker44.score.statistical_v5 import score_chunk_v5
from poker44.score.statistical_v6 import score_chunk_v6
from poker44.score.sequence_v8 import score_chunk_v8
from poker44.score.features_pot_geometry import score_chunk_pot_geometry
from poker44.score.features_response_curves import score_chunk_response_curves


def detect_chunk_type(hands: List[dict]) -> str:
    """Classify chunk as 'long' (~12 fixed actions) or 'short' (3-8 variable).

    Returns: 'long' | 'short' | 'mixed' (fallback if ambiguous).
    """
    if not hands:
        return "mixed"
    counts = [len((h.get("actions") or [])) for h in hands]
    if not counts:
        return "mixed"
    median = int(np.median(counts))
    # Long variant: most hands have exactly 12 (fixed window for single-action chunks)
    pct_12 = sum(1 for c in counts if c == 12) / len(counts)
    # Short variant: median in 3-8 range
    if median >= 10 and pct_12 >= 0.5:
        return "long"
    if median <= 8:
        return "short"
    return "mixed"


def score_chunk_v9(hands: List[dict]) -> Tuple[float, str]:
    """V9 type-aware score [0,1] + detected type. Higher = more bot-like.

    LONG strategy: v5 + v8 weighted (repetition/sizing strong, sequence helps)
    SHORT strategy: v5 + v6 weighted (per-seat consistency relatively stronger)
    """
    chunk_type = detect_chunk_type(hands)

    # Use ORTHOGONAL scorers (codex strategy iter — earlier v9 had v5_vs_v9 corr 0.99).
    # response_curves has best signal separation (std 0.18 on live).
    # pot_geometry orthogonal to v5 (corr 0.35).
    v5 = score_chunk_v5(hands)
    rc = score_chunk_response_curves(hands)
    pg = score_chunk_pot_geometry(hands)
    v8 = score_chunk_v8(hands)

    if chunk_type == "long":
        # LONG: rich action data. Mix of all 4 axes.
        # response_curves and pot_geo are strongest separators on live.
        score = 0.35 * rc + 0.25 * pg + 0.25 * v5 + 0.15 * v8
    elif chunk_type == "short":
        # SHORT: less per-hand data — response_curves and v5 most reliable.
        score = 0.40 * rc + 0.25 * v5 + 0.20 * pg + 0.15 * v8
    else:
        score = 0.30 * rc + 0.25 * pg + 0.25 * v5 + 0.20 * v8

    return max(0.0, min(1.0, score)), chunk_type


def score_chunks_v9(chunks: List[List[dict]]) -> Tuple[List[float], List[str]]:
    """Returns (scores, chunk_types) — chunk_types useful for diag logging."""
    results = [score_chunk_v9(c) for c in chunks]
    scores = [r[0] for r in results]
    types = [r[1] for r in results]
    return scores, types
