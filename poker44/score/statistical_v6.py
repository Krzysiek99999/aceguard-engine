"""Statistical detector V6 — V5 + per-seat behavioral consistency features.

Why v5 wasn't enough: v5 looks chunk-globally (entropy/repetition over all hands).
Misses signal where ONE seat plays like a bot while others are human.

V6 adds 4 per-seat features (codex iter 5 #11-17):
  - seat_action_entropy_lo: across hands per seat, entropy of action_types low → bot
  - seat_sizing_entropy_lo: bet-size entropy per seat low → bot
  - seat_vpip_consistency:  stddev of VPIP across hands per seat low → bot (stationary)
  - seat_open_size_consistency: stddev of first-aggressive size per seat low → bot

Each per-seat sub-score is averaged across seats (weighted by hand count per seat
so that low-frequency seats don't dominate).

Hypothesis: UID 211 likely exploits per-seat patterns (a bot ID == a seat across hands).
Per-seat features should add +0.02-0.05 composite over chunk-global v5.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List

# Reuse v5 helpers and primary signal
from poker44.score.statistical_v5 import (
    _entropy,
    _bet_size_quantization,
    _per_hand_signature,
    score_chunk_v5,  # fallback / sanity
)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 1e-12 else default


def _safe_stddev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _per_seat_stats(hands: List[dict]) -> Dict[int, dict]:
    """Group actions by seat and compute per-seat aggregates across the chunk.

    Returns: dict[seat -> {action_types, bet_sizes, hands_seen, vpip_per_hand, open_size_per_hand}]
    """
    seats: Dict[int, dict] = defaultdict(
        lambda: {
            "action_types": [],
            "bet_sizes": [],
            "hands_seen": 0,
            "vpip_per_hand": [],
            "open_size_per_hand": [],
        }
    )
    for hand in hands:
        actions = hand.get("actions") or []
        # Per-hand per-seat tracking
        seen_seats = set()
        per_seat_actions: Dict[int, list] = defaultdict(list)
        for a in actions:
            seat = a.get("actor_seat")
            if seat is None:
                continue
            per_seat_actions[int(seat)].append(a)

        for seat, acts in per_seat_actions.items():
            seats[seat]["hands_seen"] += 1
            seen_seats.add(seat)
            # All action types this seat made in this hand → contribute to global per-seat
            for a in acts:
                t = a.get("action_type", "?")
                seats[seat]["action_types"].append(t)
                if t in ("bet", "raise"):
                    amt = a.get("normalized_amount_bb") or 0.0
                    if amt > 0:
                        seats[seat]["bet_sizes"].append(float(amt))
            # Per-hand VPIP: did this seat voluntarily put money in?
            voluntary = any(
                a.get("action_type") in ("call", "bet", "raise")
                for a in acts
                if a.get("street") == "preflop"
            )
            seats[seat]["vpip_per_hand"].append(1.0 if voluntary else 0.0)
            # Per-hand first aggressive size (preflop opens)
            for a in acts:
                if a.get("street") == "preflop" and a.get("action_type") in ("bet", "raise"):
                    amt = a.get("normalized_amount_bb") or 0.0
                    if amt > 0:
                        seats[seat]["open_size_per_hand"].append(float(amt))
                        break
    return dict(seats)


def _per_seat_entropy_score(seats: Dict[int, dict], n_hands: int) -> float:
    """Average (weighted by hands_seen) of (1 - normalized action-type entropy) per seat.

    High = each seat plays narrow action mix = bot-like.
    """
    if not seats or n_hands == 0:
        return 0.0
    weighted_sum = 0.0
    total_w = 0.0
    for seat, s in seats.items():
        types = s["action_types"]
        if not types:
            continue
        cnt = Counter(types)
        H = _entropy(list(cnt.values()))
        H_max = math.log2(max(len(cnt), 1)) if cnt else 1.0
        ent_norm = H / max(H_max, 1e-6)
        score = 1.0 - ent_norm
        w = s["hands_seen"]
        weighted_sum += w * score
        total_w += w
    return _safe_div(weighted_sum, total_w, 0.0)


def _per_seat_sizing_entropy_score(seats: Dict[int, dict]) -> float:
    """Per-seat bet-size entropy (low → bot). Bucket sizes coarsely, weighted by hands."""
    if not seats:
        return 0.0
    weighted_sum = 0.0
    total_w = 0.0
    # bucket boundaries (BB units) — coarse
    buckets = [0.5, 1.5, 3.0, 5.0, 10.0]
    for seat, s in seats.items():
        sizes = s["bet_sizes"]
        if len(sizes) < 2:
            continue
        binned = []
        for amt in sizes:
            b = 0
            for i, top in enumerate(buckets):
                if amt < top:
                    b = i
                    break
            else:
                b = len(buckets)
            binned.append(b)
        cnt = Counter(binned)
        H = _entropy(list(cnt.values()))
        H_max = math.log2(max(len(cnt), 1)) if cnt else 1.0
        ent_norm = H / max(H_max, 1e-6)
        score = 1.0 - ent_norm
        w = s["hands_seen"]
        weighted_sum += w * score
        total_w += w
    return _safe_div(weighted_sum, total_w, 0.0)


def _per_seat_vpip_consistency_score(seats: Dict[int, dict]) -> float:
    """Per-seat stddev of VPIP across hands. Low stddev = bot (stationary policy).

    Returns 1 - normalized_stddev so that high = more bot-like.
    """
    if not seats:
        return 0.0
    # Average per-seat (1 - stddev), weighted by hands_seen
    weighted_sum = 0.0
    total_w = 0.0
    for seat, s in seats.items():
        vp = s["vpip_per_hand"]
        if len(vp) < 3:
            continue  # need enough samples
        sd = _safe_stddev(vp)
        # Max stddev for Bernoulli is 0.5 — normalize
        sd_norm = min(sd / 0.5, 1.0)
        score = 1.0 - sd_norm  # high = consistent = bot
        w = s["hands_seen"]
        weighted_sum += w * score
        total_w += w
    return _safe_div(weighted_sum, total_w, 0.0)


def _per_seat_open_size_consistency_score(seats: Dict[int, dict]) -> float:
    """Per-seat stddev of first preflop aggressive size. Low = bot uses fixed open sizes."""
    if not seats:
        return 0.0
    weighted_sum = 0.0
    total_w = 0.0
    for seat, s in seats.items():
        opens = s["open_size_per_hand"]
        if len(opens) < 3:
            continue
        m = sum(opens) / len(opens)
        if m < 1e-6:
            continue
        sd = _safe_stddev(opens)
        cv = sd / m  # coefficient of variation
        cv_norm = min(cv, 1.0)
        score = 1.0 - cv_norm
        w = s["hands_seen"]
        weighted_sum += w * score
        total_w += w
    return _safe_div(weighted_sum, total_w, 0.0)


def _seat_action_entropy_norm(seat_stats: dict) -> float:
    """Returns 1 - normalized action-type entropy for a single seat (high = bot)."""
    types = seat_stats["action_types"]
    if not types:
        return 0.0
    cnt = Counter(types)
    H = _entropy(list(cnt.values()))
    H_max = math.log2(max(len(cnt), 1)) if cnt else 1.0
    return 1.0 - (H / max(H_max, 1e-6))


def _seat_sizing_entropy_norm(seat_stats: dict) -> float:
    """Returns 1 - normalized bet-size bucket entropy for a single seat."""
    sizes = seat_stats["bet_sizes"]
    if len(sizes) < 2:
        return 0.0
    buckets = [0.5, 1.5, 3.0, 5.0, 10.0]
    binned = []
    for amt in sizes:
        b = len(buckets)
        for i, top in enumerate(buckets):
            if amt < top:
                b = i
                break
        binned.append(b)
    cnt = Counter(binned)
    H = _entropy(list(cnt.values()))
    H_max = math.log2(max(len(cnt), 1)) if cnt else 1.0
    return 1.0 - (H / max(H_max, 1e-6))


def _seat_vpip_consistency_norm(seat_stats: dict) -> float:
    """1 if VPIP is constant across hands, 0 if max stddev. Bernoulli max stddev = 0.5."""
    vp = seat_stats["vpip_per_hand"]
    if len(vp) < 3:
        return 0.0
    sd = _safe_stddev(vp)
    sd_norm = min(sd / 0.5, 1.0)
    return 1.0 - sd_norm


def score_chunk_v6(hands: List[dict]) -> float:
    """V6 = v5 score + GATED per-seat bonus (codex verify iter 2/5 design).

    Architecture: v5 stays at FULL weight, per-seat features add a bonus ONLY when:
      - >= 2 reliable seats (>=6 hands each, >=10 actions each)
      - max seat bot-score >= 0.55 AND gap from median seat >= 0.18
      - v5 >= 0.30 (don't boost already-flat chunks blindly)
    This avoids the dilution of v5's proven signal. Bonus capped at 0.08 (max
    exceptional) or 0.04 (standard escalation).

    Codex iter 5 pseudocode shape, applied with seat reliability gates.
    """
    if not hands:
        return 0.5

    # Step 1: v5 baseline (FULL weight)
    v5 = score_chunk_v5(hands)

    # Step 2: collect reliable seats
    all_seats = _per_seat_stats(hands)
    reliable = []
    for seat_id, s in all_seats.items():
        if s["hands_seen"] >= 6 and len(s["action_types"]) >= 10:
            reliable.append(s)
    if len(reliable) < 2:
        return v5  # insufficient seat data, fall back to chunk-global only

    # Step 3: per-seat bot scores (3 signals weighted 0.5/0.3/0.2 per codex iter 5)
    seat_scores = []
    for s in reliable:
        a_ent = _seat_action_entropy_norm(s)
        s_ent = _seat_sizing_entropy_norm(s)
        vp_c = _seat_vpip_consistency_norm(s)
        seat_scores.append(0.5 * a_ent + 0.3 * s_ent + 0.2 * vp_c)

    best = max(seat_scores)
    import numpy as np
    median = float(np.median(seat_scores))
    gap = best - median

    # Step 4: gated bonus — only when ONE seat is clearly more bot-like than others
    # AND v5 already has some signal (don't boost flat chunks)
    if v5 >= 0.30 and best >= 0.55 and gap >= 0.18:
        # Exceptional bonus 0.08 if best very high and gap big; otherwise 0.04
        bonus_cap = 0.08 if (best >= 0.70 and gap >= 0.25) else 0.04
        bonus = min(bonus_cap, gap * 0.35)
        return min(1.0, v5 + bonus)
    return v5


def score_chunks_v6(chunks: List[List[dict]]) -> List[float]:
    return [score_chunk_v6(c) for c in chunks]
