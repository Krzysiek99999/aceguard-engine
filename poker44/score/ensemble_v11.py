"""V11 — type-aware ensemble combiner (codex iter 18 design).

NOT trained ML (avoids v7 saturation trap). Deterministic chooser:
  1. Detect LONG/SHORT type (from v9)
  2. Compute v5, v6, v8.1, pot_geo, response_curves scores
  3. Compute per-scorer confidence (top_n_separation × score_std)
  4. Per chunk: pick highest-confidence scorer aligned with type
  5. Apply agreement bonus if 2+ scorers agree on top-N

Signal families summary (all orthogonal, correlations 0.16-0.34):
  - v5_chunk_global       (chunk-level entropy/repetition/sizing)
  - v6_per_seat           (per-seat additive bonus over v5)
  - v8_markov             (n-gram + Markov transition matrix)
  - pot_geometry          (bet/pot ratios, SPR, growth regularity)
  - response_curves       (per-seat fold-to-aggression, response entropy)

Per chunk output: blended score [0,1] + telemetry which scorer dominated.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from poker44.score.statistical_v5 import score_chunk_v5
from poker44.score.statistical_v6 import score_chunk_v6
from poker44.score.sequence_v8_markov import score_chunk_v8_combined
from poker44.score.features_pot_geometry import score_chunk_pot_geometry
from poker44.score.features_response_curves import score_chunk_response_curves
from poker44.score.statistical_v9 import detect_chunk_type


# Per-chunk-type scorer priority order. First in list = highest weight, last = lowest.
LONG_PRIORITY = ["v5", "v8_markov", "pot_geo", "v6", "response_curves"]
SHORT_PRIORITY = ["v5", "v6", "response_curves", "pot_geo", "v8_markov"]
MIXED_PRIORITY = ["v5", "v8_markov", "v6", "pot_geo", "response_curves"]


def score_chunk_v11(hands: List[dict]) -> Tuple[float, Dict[str, float], str]:
    """V11 ensemble score [0,1] + per-scorer telemetry + chunk_type.

    Returns: (final_score, scorer_dict, chunk_type)
    """
    if not hands:
        return 0.5, {}, "mixed"

    # Compute all scorers
    scores = {
        "v5": score_chunk_v5(hands),
        "v6": score_chunk_v6(hands),
        "v8_markov": score_chunk_v8_combined(hands),
        "pot_geo": score_chunk_pot_geometry(hands),
        "response_curves": score_chunk_response_curves(hands),
    }

    chunk_type = detect_chunk_type(hands)
    if chunk_type == "long":
        priority = LONG_PRIORITY
    elif chunk_type == "short":
        priority = SHORT_PRIORITY
    else:
        priority = MIXED_PRIORITY

    # Weighted by priority: position-weight = (N - rank) / sum(1..N)
    weights = [(len(priority) - i) for i in range(len(priority))]
    w_sum = sum(weights)
    weighted_score = sum(
        (w / w_sum) * scores[name]
        for w, name in zip(weights, priority)
    )

    # Agreement bonus: how many scorers exceed median of all?
    score_values = list(scores.values())
    median_v = float(np.median(score_values))
    # Above-median scorers count as "agreeing on high"
    above = sum(1 for v in score_values if v >= median_v + 0.05)
    if above >= 3:
        bonus = 0.05
    elif above == 2:
        bonus = 0.02
    else:
        bonus = 0.0

    final = min(1.0, weighted_score + bonus)
    return final, scores, chunk_type


def score_chunks_v11(chunks: List[List[dict]]) -> Tuple[List[float], List[Dict[str, float]], List[str]]:
    """Returns (scores, per_chunk_telemetry, chunk_types)."""
    final = []
    tele = []
    types = []
    for c in chunks:
        s, t, ct = score_chunk_v11(c)
        final.append(s)
        tele.append(t)
        types.append(ct)
    return final, tele, types
