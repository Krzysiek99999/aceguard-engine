"""V7 feature extractor — minimal stable feature set.

Pure behavioral features only:
  - 5 v5 sub-signals (chunk-global)
  - 4 v6 sub-signals (per-seat)

Total 9 features. Designed for benchmark-supervised LightGBM training while avoiding
the v2 saturation trap (LightGBM finding shortcut on other_ratio which doesn't exist
on live).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

import numpy as np

from poker44.score.statistical_v5 import (
    _entropy,
    _bet_size_quantization,
    _per_hand_signature,
)
from poker44.score.statistical_v6 import (
    _per_seat_stats,
    _per_seat_entropy_score,
    _per_seat_sizing_entropy_score,
    _per_seat_vpip_consistency_score,
    _per_seat_open_size_consistency_score,
)


V7_FEATURE_NAMES: List[str] = [
    # v5 chunk-global
    "v5_repetition",
    "v5_unique_lo",
    "v5_quant",
    "v5_action_entropy_lo",
    "v5_regularity",
    # v6 per-seat
    "v6_seat_action_entropy",
    "v6_seat_sizing_entropy",
    "v6_seat_vpip_consistency",
    "v6_seat_open_size_consistency",
]


def extract_v7_features(hands: List[dict]) -> np.ndarray:
    """Extract 9-dim feature vector for one chunk. Order matches V7_FEATURE_NAMES."""
    n_hands = len(hands)
    if n_hands == 0:
        return np.zeros(len(V7_FEATURE_NAMES), dtype=np.float64)

    # v5 sub-features
    all_action_types: List[str] = []
    per_hand_action_counts: List[int] = []
    aggressive_amounts: List[float] = []
    signatures: List[str] = []

    for hand in hands:
        actions = hand.get("actions") or []
        per_hand_action_counts.append(len(actions))
        for a in actions:
            t = a.get("action_type", "?")
            all_action_types.append(t)
            if t in ("bet", "raise"):
                amt = a.get("normalized_amount_bb") or 0.0
                if amt > 0:
                    aggressive_amounts.append(float(amt))
        signatures.append(_per_hand_signature(hand))

    counts = Counter(all_action_types)
    H_action = _entropy(list(counts.values()))
    H_action_max = math.log2(max(len(counts), 1)) if counts else 1.0
    action_entropy_norm = H_action / max(H_action_max, 1e-6)
    action_entropy_lo = 1.0 - action_entropy_norm

    quant = _bet_size_quantization(aggressive_amounts)

    sig_counts = Counter(signatures)
    n_repeated = sum(c for c in sig_counts.values() if c > 1)
    repetition = n_repeated / max(n_hands, 1)

    if len(per_hand_action_counts) > 1:
        mean = sum(per_hand_action_counts) / len(per_hand_action_counts)
        var = sum((x - mean) ** 2 for x in per_hand_action_counts) / len(per_hand_action_counts)
        std = math.sqrt(var)
        cv = std / max(mean, 1e-6)
        regularity = max(0.0, 1.0 - min(cv, 1.0))
    else:
        regularity = 0.5

    n_unique = len(sig_counts)
    unique_lo = 1.0 - (n_unique / max(n_hands, 1))

    # v6 per-seat
    seats = _per_seat_stats(hands)
    seat_action_entropy = _per_seat_entropy_score(seats, n_hands)
    seat_sizing_entropy = _per_seat_sizing_entropy_score(seats)
    seat_vpip = _per_seat_vpip_consistency_score(seats)
    seat_open_size = _per_seat_open_size_consistency_score(seats)

    return np.array(
        [
            repetition,
            unique_lo,
            quant,
            action_entropy_lo,
            regularity,
            seat_action_entropy,
            seat_sizing_entropy,
            seat_vpip,
            seat_open_size,
        ],
        dtype=np.float64,
    )


def chunks_to_matrix_v7(chunks: List[dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build (X, y, names) for a list of {hands, is_bot} dicts. Mirrors features_v1.chunks_to_matrix."""
    X = np.zeros((len(chunks), len(V7_FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(len(chunks), dtype=np.int32)
    for i, c in enumerate(chunks):
        hands = c.get("hands", []) if isinstance(c, dict) else c
        X[i] = extract_v7_features(hands)
        y[i] = int(c.get("is_bot", 0)) if isinstance(c, dict) else 0
    return X, y, list(V7_FEATURE_NAMES)
