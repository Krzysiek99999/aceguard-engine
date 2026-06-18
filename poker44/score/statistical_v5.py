"""Statistical detector V5.

This detector ranks chunks by behavioral anomaly score using features
that have natural variance on live:
  - Bet size entropy / quantization
  - Action count regularity
  - Action sequence repetition
  - Bet sizing dispersion

The score is a unitless anomaly index; downstream rank calibration maps it to
validator-facing risk scores.

Usage:
  from poker44.score.statistical_v5 import score_chunk_v5
  raw_score = score_chunk_v5(chunk_hands)  # → float in [0, 1]
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence

import math


def _entropy(counts: Sequence[int]) -> float:
    """Shannon entropy of a discrete distribution (counts). Bits."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    p = [c / total for c in counts if c > 0]
    return -sum(pi * math.log2(pi) for pi in p)


def _bet_size_quantization(amounts_bb: List[float]) -> float:
    """Fraction of bet sizes that are near "round" pot fractions.

    Bots often use templated sizings (0.25x, 0.5x, 0.75x, 1.0x pot).
    Higher score = more quantized = more bot-like.

    Returns: ratio in [0, 1].
    """
    if not amounts_bb:
        return 0.0
    targets = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    n_near = 0
    for a in amounts_bb:
        if a <= 0:
            continue
        rel = a  # already normalized_amount_bb relative-ish
        for t in targets:
            if abs(rel - t) / max(t, 1e-6) < 0.05:
                n_near += 1
                break
    return n_near / max(len(amounts_bb), 1)


def _per_hand_signature(hand: dict) -> str:
    """Compact deterministic signature of a hand's action sequence + sizing bucket."""
    actions = hand.get("actions") or []
    sig_parts = []
    for a in actions:
        at = a.get("action_type", "?")
        amt = a.get("normalized_amount_bb") or 0.0
        # bucket bet sizes coarsely (4 bins)
        if at in ("bet", "raise"):
            if amt < 1.0:
                bucket = "s"
            elif amt < 3.0:
                bucket = "m"
            elif amt < 8.0:
                bucket = "l"
            else:
                bucket = "x"
            sig_parts.append(f"{at[0]}{bucket}")
        else:
            sig_parts.append(at[0])  # f/c/k/o
    return "-".join(sig_parts)


def score_chunk_v5(hands: List[dict]) -> float:
    """Anomaly score for a chunk of hands. Range ~[0, 1]; higher = more bot-like.

    Combines 5 sub-scores into a weighted sum:
    1. action_entropy_lo: low entropy in action type distribution → bot
    2. sizing_quantization: bet amounts near round fractions → bot
    3. sequence_repetition: same hand-signature repeated in chunk → bot
    4. action_count_regularity: low std in actions per hand → bot
    5. unique_sigs_low: small fraction of unique signatures → bot

    Weights are fixed and deterministic.
    """
    if not hands:
        return 0.5

    n_hands = len(hands)

    # 1. Action type entropy across all actions in chunk
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

    # Action type entropy (max ~2.32 bits for 5 types — normalize)
    counts = Counter(all_action_types)
    H_action = _entropy(list(counts.values()))
    H_action_max = math.log2(max(len(counts), 1)) if counts else 1.0
    action_entropy_norm = H_action / max(H_action_max, 1e-6)  # 0..1
    action_entropy_lo = 1.0 - action_entropy_norm  # high → bot

    # 2. Bet sizing quantization
    quant = _bet_size_quantization(aggressive_amounts)

    # 3. Sequence repetition: fraction of hands whose signature appears > 1 time
    sig_counts = Counter(signatures)
    n_repeated = sum(c for c in sig_counts.values() if c > 1)
    repetition = n_repeated / max(n_hands, 1)

    # 4. Action count regularity: low std → bot
    if len(per_hand_action_counts) > 1:
        mean = sum(per_hand_action_counts) / len(per_hand_action_counts)
        var = sum((x - mean) ** 2 for x in per_hand_action_counts) / len(per_hand_action_counts)
        std = math.sqrt(var)
        cv = std / max(mean, 1e-6)
        regularity = max(0.0, 1.0 - min(cv, 1.0))  # low cv → high regularity → bot
    else:
        regularity = 0.5

    # 5. Unique signatures ratio (low → bot)
    n_unique = len(sig_counts)
    unique_lo = 1.0 - (n_unique / max(n_hands, 1))

    # Weighted combination emphasizing repetition and uniqueness.
    score = (
        0.25 * repetition
        + 0.25 * unique_lo
        + 0.20 * quant
        + 0.15 * action_entropy_lo
        + 0.15 * regularity
    )
    return max(0.0, min(1.0, score))


def score_chunks_v5(chunks: List[List[dict]]) -> List[float]:
    """Score a list of chunks. Returns list of floats in [0,1]."""
    return [score_chunk_v5(c) for c in chunks]
