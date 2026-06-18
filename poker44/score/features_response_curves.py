"""Response-curve features.

Per-pair response features can be sparse, so this module uses seat-level
response-to-any-opponent features:

  - seat_fold_when_facing_aggression (per-seat fold% after opponent bet/raise)
  - seat_call_when_facing_aggression (per-seat call% after opponent bet/raise)
  - seat_3bet_opportunity_response (raise% when facing single raise)
  - seat_response_entropy_after_bet (entropy of seat's response to bet)
  - seat_response_entropy_after_raise (entropy of seat's response to raise)

Chunk-level aggregates:
  - max_botlike_response_seat: most deterministic responder
  - gap_best_vs_median_response_seat: separation
  - low_response_entropy_share: % of seats with H_norm < 0.5
  - std_response_policy_across_seats

Bot signal: ONE seat with extremely deterministic response pattern = fixed-strategy bot.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List


def _entropy_norm(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    H = -sum(p * math.log2(p) for p in probs)
    H_max = math.log2(max(len(probs), 1))
    return H / max(H_max, 1e-6)


def _gather_per_seat_responses(hands: List[dict]) -> Dict[int, dict]:
    """For each seat, collect responses to (prior_action_type) events.

    Returns dict[seat -> {
      'after_bet': [seat_response_types],
      'after_raise': [seat_response_types],
      'after_check': [seat_response_types],
      'after_call': [seat_response_types],
      'hands_seen': N,
    }]
    """
    seats: Dict[int, dict] = defaultdict(lambda: {
        "after_bet": [],
        "after_raise": [],
        "after_check": [],
        "after_call": [],
        "hands_seen": 0,
        "total_actions": 0,
    })

    for hand in hands:
        actions = hand.get("actions") or []
        # Track seats in this hand
        seats_in_hand = set()
        for i, a in enumerate(actions):
            seat = a.get("actor_seat")
            if seat is None:
                continue
            seat = int(seat)
            seats_in_hand.add(seat)
            seats[seat]["total_actions"] += 1
            # Look at PRIOR action — what did this seat respond to?
            if i == 0:
                continue
            prior = actions[i - 1]
            prior_seat = prior.get("actor_seat")
            if prior_seat is None or int(prior_seat) == seat:
                # Skip if prior was same seat (e.g. blind posting then action)
                continue
            prior_type = prior.get("action_type", "")
            this_type = a.get("action_type", "")
            if prior_type == "bet":
                seats[seat]["after_bet"].append(this_type)
            elif prior_type == "raise":
                seats[seat]["after_raise"].append(this_type)
            elif prior_type == "check":
                seats[seat]["after_check"].append(this_type)
            elif prior_type == "call":
                seats[seat]["after_call"].append(this_type)

        for s in seats_in_hand:
            seats[s]["hands_seen"] += 1

    return dict(seats)


def _seat_response_bot_score(seat_stats: dict) -> float:
    """Per-seat score [0,1]: how deterministic/bot-like are responses.

    Combines:
      - fold-to-aggression rate (high = call station bot OR predictable fold bot)
      - response entropy after each prior-action context (low = deterministic)
      - dominant response concentration (high = one response wins always)
    """
    if seat_stats["total_actions"] < 10 or seat_stats["hands_seen"] < 6:
        return 0.0  # insufficient samples

    # Aggregate response entropy across all (prior, response) contexts
    contexts = [
        seat_stats["after_bet"],
        seat_stats["after_raise"],
        seat_stats["after_check"],
        seat_stats["after_call"],
    ]
    entropies = []
    max_concs = []  # max concentration per context
    for ctx in contexts:
        if len(ctx) >= 3:
            cnt = Counter(ctx)
            H = _entropy_norm(list(cnt.values()))
            entropies.append(H)
            max_concs.append(max(cnt.values()) / len(ctx))

    if not entropies:
        return 0.0

    avg_entropy = sum(entropies) / len(entropies)
    avg_max_conc = sum(max_concs) / len(max_concs)

    # Low entropy + high concentration = deterministic = bot
    entropy_signal = 1.0 - avg_entropy
    conc_signal = max(0.0, (avg_max_conc - 0.5) * 2.0)  # > 0.5 starts to count

    # Combined per-seat bot score
    return 0.6 * entropy_signal + 0.4 * conc_signal


def extract_response_curve_features(hands: List[dict]) -> dict:
    """Extract response curve sub-scores for a chunk. Returns dict [0,1] scores."""
    if not hands:
        return {
            "max_botlike_seat": 0.0,
            "gap_best_vs_median": 0.0,
            "low_entropy_share": 0.0,
            "response_curves_combined": 0.0,
        }

    seats = _gather_per_seat_responses(hands)
    seat_scores = [_seat_response_bot_score(s) for s in seats.values() if s["total_actions"] >= 10]

    if len(seat_scores) < 2:
        return {
            "max_botlike_seat": 0.0,
            "gap_best_vs_median": 0.0,
            "low_entropy_share": 0.0,
            "response_curves_combined": 0.0,
        }

    import numpy as np
    max_seat = float(max(seat_scores))
    median_seat = float(np.median(seat_scores))
    gap = max_seat - median_seat
    low_entropy_share = sum(1 for s in seat_scores if s >= 0.5) / len(seat_scores)

    # Combined: emphasize one-seat-stands-out pattern
    combined = (
        0.40 * max_seat
        + 0.30 * gap          # bigger gap = clearer bot among humans
        + 0.30 * low_entropy_share
    )

    return {
        "max_botlike_seat": max_seat,
        "gap_best_vs_median": gap,
        "low_entropy_share": low_entropy_share,
        "response_curves_combined": max(0.0, min(1.0, combined)),
    }


def score_chunk_response_curves(hands: List[dict]) -> float:
    return extract_response_curve_features(hands)["response_curves_combined"]


def score_chunks_response_curves(chunks: List[List[dict]]) -> List[float]:
    return [score_chunk_response_curves(c) for c in chunks]
