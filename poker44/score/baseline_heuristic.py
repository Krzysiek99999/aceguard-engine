"""Baseline heuristic — replica oficjalnego neurons/miner.py.

7-feature linear weighted scoring per hand, average per chunk.
Re-implementacja w nowym module — NIE importuje obcego kodu.

Polarity: high score = bot. Range [0, 1].
"""
from __future__ import annotations
from collections import Counter
from typing import Sequence


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_hand_baseline(hand: dict) -> float:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}

    action_counts = Counter((a.get("action_type") or "").lower() for a in actions)
    meaningful = max(1, sum(action_counts.get(k, 0)
                            for k in ("call", "check", "bet", "raise", "fold")))

    call_r = action_counts.get("call", 0) / meaningful
    check_r = action_counts.get("check", 0) / meaningful
    fold_r = action_counts.get("fold", 0) / meaningful
    raise_r = action_counts.get("raise", 0) / meaningful
    street_d = len(streets) / 3.0
    showdown = 1.0 if outcome.get("showdown") else 0.0
    pcs = (6 - min(len(players), 6)) / 4.0 if players else 0.0

    score = 0.0
    score += 0.32 * street_d
    score += 0.22 * showdown
    score += 0.18 * _clamp01(call_r / 0.35)
    score += 0.12 * _clamp01(check_r / 0.30)
    score += 0.08 * _clamp01(pcs)
    score -= 0.18 * _clamp01(fold_r / 0.55)
    score -= 0.10 * _clamp01(raise_r / 0.20)
    return _clamp01(score)


def score_chunk_baseline_heuristic(hands: Sequence[dict]) -> float:
    if not hands:
        return 0.5
    return round(_clamp01(sum(score_hand_baseline(h) for h in hands) / len(hands)), 6)


def score_chunks_baseline_heuristic(chunks: Sequence[Sequence[dict]]) -> list[float]:
    return [score_chunk_baseline_heuristic(c) for c in chunks]
