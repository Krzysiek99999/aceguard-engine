"""Pot geometry features.

Bots often expose rigid pot-growth and bet-fraction patterns. Live payload retains:
  - pot_before, pot_after per action
  - normalized_amount_bb per action

Sub-signals:
  - bet_to_pot_fraction_quantization: bets clustered at 0.5/0.75/1.0 pot (bot-templated)
  - pot_growth_regularity: pot_after/pot_before ratio std (low = bot)
  - spr_distribution: stack-to-pot ratio entropy (low = bot uses fixed SPR thresholds)
  - showdown_pot_size_cv: pot at end of hand (low CV = bot rigid commitments)

These features INDEPENDENT of v5/v6/v8 (different signal axis: pot dynamics).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List

import numpy as np


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-12 else default


def _entropy_norm(counts: List[int]) -> float:
    """Normalized entropy [0,1]."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    H = -sum(p * math.log2(p) for p in probs)
    H_max = math.log2(max(len(probs), 1))
    return H / max(H_max, 1e-6)


def extract_pot_geometry_features(hands: List[dict]) -> dict:
    """Extract 5 pot geometry features for a chunk. Returns dict with sub-scores [0,1]."""
    if not hands:
        return {
            "bet_to_pot_quant": 0.0,
            "pot_growth_regularity": 0.0,
            "spr_low_entropy": 0.0,
            "showdown_pot_low_cv": 0.0,
            "pot_geometry_combined": 0.0,
        }

    bet_pot_fractions: List[float] = []
    pot_growth_ratios: List[float] = []
    spr_values: List[float] = []
    showdown_pots: List[float] = []

    for hand in hands:
        actions = hand.get("actions") or []
        last_pot = 0.0
        for a in actions:
            pot_before = float(a.get("pot_before") or 0.0)
            pot_after = float(a.get("pot_after") or 0.0)
            amt = float(a.get("normalized_amount_bb") or 0.0)
            atype = a.get("action_type", "")

            # 1. Bet-to-pot fraction (only for bet/raise)
            if atype in ("bet", "raise") and amt > 0 and pot_before > 0:
                # Convert: amt in BB, pot_before in chips. Use proportion.
                # If pot_before in BB-equivalent (sb=0.01, bb=0.02 fixed), normalize:
                pot_bb = pot_before / 0.02  # convert to BB
                if pot_bb > 0:
                    frac = amt / pot_bb
                    bet_pot_fractions.append(frac)

            # 2. Pot growth ratio per action (skip blinds)
            if atype not in ("small_blind", "big_blind", "ante") and pot_before > 0 and pot_after > pot_before:
                ratio = pot_after / pot_before
                pot_growth_ratios.append(ratio)

            # 3. SPR — stack to pot ratio (rough approx — we don't have current stack mid-hand,
            #    use starting_stack from players list as proxy)
            last_pot = pot_after

        # Track showdown pot (last pot_after in hand)
        if last_pot > 0:
            showdown_pots.append(last_pot)

        # SPR approximation: starting_stack / first significant pot_before
        players = hand.get("players") or []
        if players and actions:
            avg_start = sum(float(p.get("starting_stack") or 0.0) for p in players) / max(len(players), 1)
            # First non-blind action pot
            first_real_pot = next(
                (float(a.get("pot_before") or 0.0) for a in actions
                 if a.get("action_type") not in ("small_blind", "big_blind", "ante") and (a.get("pot_before") or 0.0) > 0),
                0.0
            )
            if first_real_pot > 0:
                spr_values.append(avg_start / first_real_pot)

    # 1. Bet-to-pot quantization: % of bets near {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}
    targets = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    if bet_pot_fractions:
        n_near = sum(
            1 for f in bet_pot_fractions
            if any(abs(f - t) / max(t, 1e-6) < 0.05 for t in targets)
        )
        bet_to_pot_quant = n_near / len(bet_pot_fractions)
    else:
        bet_to_pot_quant = 0.0

    # 2. Pot growth regularity: low std of growth ratios → bot
    if len(pot_growth_ratios) >= 5:
        mean = sum(pot_growth_ratios) / len(pot_growth_ratios)
        std = math.sqrt(sum((x - mean) ** 2 for x in pot_growth_ratios) / len(pot_growth_ratios))
        cv = std / max(mean, 1e-6)
        pot_growth_regularity = max(0.0, 1.0 - min(cv, 1.0))
    else:
        pot_growth_regularity = 0.0

    # 3. SPR entropy: bucket SPR and compute entropy
    if len(spr_values) >= 5:
        # Bucket: <2, 2-5, 5-10, 10-25, 25-50, 50+
        buckets = [2, 5, 10, 25, 50]
        binned = []
        for spr in spr_values:
            b = len(buckets)
            for i, t in enumerate(buckets):
                if spr < t:
                    b = i
                    break
            binned.append(b)
        cnt = Counter(binned)
        H_norm = _entropy_norm(list(cnt.values()))
        spr_low_entropy = 1.0 - H_norm  # low entropy → bot signal
    else:
        spr_low_entropy = 0.0

    # 4. Showdown pot CV: low CV = rigid commitments
    if len(showdown_pots) >= 5:
        mean = sum(showdown_pots) / len(showdown_pots)
        std = math.sqrt(sum((x - mean) ** 2 for x in showdown_pots) / len(showdown_pots))
        cv = std / max(mean, 1e-6)
        showdown_pot_low_cv = max(0.0, 1.0 - min(cv, 1.0))
    else:
        showdown_pot_low_cv = 0.0

    # 5. Combined pot geometry score
    combined = (
        0.30 * bet_to_pot_quant
        + 0.25 * pot_growth_regularity
        + 0.25 * spr_low_entropy
        + 0.20 * showdown_pot_low_cv
    )

    return {
        "bet_to_pot_quant": bet_to_pot_quant,
        "pot_growth_regularity": pot_growth_regularity,
        "spr_low_entropy": spr_low_entropy,
        "showdown_pot_low_cv": showdown_pot_low_cv,
        "pot_geometry_combined": max(0.0, min(1.0, combined)),
    }


def score_chunk_pot_geometry(hands: List[dict]) -> float:
    """Single combined score [0,1] for pot geometry signal."""
    return extract_pot_geometry_features(hands)["pot_geometry_combined"]


def score_chunks_pot_geometry(chunks: List[List[dict]]) -> List[float]:
    return [score_chunk_pot_geometry(c) for c in chunks]
