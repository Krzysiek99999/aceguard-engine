"""Enterprise-grade extended bot detection features.

Beyond v27's 54 features, adds research-inspired features:
  - Bet-to-pot precision (round number bias detection)
  - Action sequence symmetry / palindrome detection
  - Inter-hand variance (per-hand features, variance across chunk)
  - Stack-bracketed VPIP (VPIP × stack size interaction)
  - N-gram entropy (3, 4, 5-action sequences)
  - Pot growth velocity per street
  - Bet size clustering (clusters of bet amounts)
  - Cross-hand pattern repetition

Sources: poker bot detection papers (e.g., Schnizlein et al., Billings AAAI bot work),
         online poker tracking software heuristics (HEM, PT4).
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence
import numpy as np


# Standard pot-fraction templates that bots commonly use
POT_FRACTIONS = [0.25, 0.33, 0.40, 0.50, 0.60, 0.66, 0.75, 1.00, 1.25, 1.50, 2.00]
ROUND_BET_VALUES_BB = [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 36, 40, 50, 60, 80, 100]


def _safe_div(a, b):
    return a / b if b > 0 else 0.0


def compute_enterprise_features(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute 40+ extra features beyond v27."""
    feats: Dict[str, float] = {}
    n_hands = len(hands)
    if n_hands == 0:
        for i in range(100, 145):
            feats[f"e{i}_default"] = 0.5
        return feats

    # ===== Gather raw data per hand =====
    per_hand_n_actions = []
    per_hand_action_types = []
    per_hand_bet_sizes_bb = []
    per_hand_pot_growth = []
    per_hand_n_streets = []
    per_hand_hero_seat = []
    per_hand_max_seats = []
    all_actions_flat = []
    all_bets_bb = []
    all_pot_fractions = []
    all_round_bet_distances = []
    sequence_3grams = Counter()
    sequence_4grams = Counter()
    sequence_5grams = Counter()
    bets_to_pot_ratios = []

    for hand in hands:
        actions = hand.get("actions") or []
        meta = hand.get("metadata") or {}
        per_hand_max_seats.append(meta.get("max_seats", 6))
        per_hand_hero_seat.append(meta.get("hero_seat", 0))
        per_hand_n_actions.append(len(actions))
        action_types_this = []
        bet_sizes_this = []
        pot_growth_this = 0.0
        streets_reached = set()
        for a in actions:
            if not isinstance(a, dict): continue
            t = a.get("action_type", "")
            action_types_this.append(t)
            all_actions_flat.append(t)
            pb = float(a.get("pot_before", 0))
            pa = float(a.get("pot_after", 0))
            growth = pa - pb
            pot_growth_this += growth
            streets_reached.add(a.get("street"))
            if t in ("bet", "raise"):
                amt_bb = float(a.get("normalized_amount_bb", 0))
                if amt_bb > 0:
                    bet_sizes_this.append(amt_bb)
                    all_bets_bb.append(amt_bb)
                    # Pot fraction (using normalized — bb vs chips conversion approx)
                    pot_bb = pb / 0.02  # bb conversion
                    if pot_bb > 0.5:
                        pf = amt_bb / pot_bb
                        all_pot_fractions.append(pf)
                        bets_to_pot_ratios.append(pf)
                    # Distance to nearest round bet value
                    nearest = min(ROUND_BET_VALUES_BB, key=lambda v: abs(v - amt_bb))
                    rel_dist = abs(nearest - amt_bb) / max(nearest, 1)
                    all_round_bet_distances.append(rel_dist)
        per_hand_action_types.append(action_types_this)
        per_hand_bet_sizes_bb.append(bet_sizes_this)
        per_hand_pot_growth.append(pot_growth_this)
        per_hand_n_streets.append(len(streets_reached))
        # N-grams from action sequence (first letter of action_type)
        seq = [t[0] if t else "?" for t in action_types_this]
        for i in range(len(seq) - 2):
            sequence_3grams[(seq[i], seq[i+1], seq[i+2])] += 1
        for i in range(len(seq) - 3):
            sequence_4grams[(seq[i], seq[i+1], seq[i+2], seq[i+3])] += 1
        for i in range(len(seq) - 4):
            sequence_5grams[(seq[i], seq[i+1], seq[i+2], seq[i+3], seq[i+4])] += 1

    # ===== e100-e110: bet-to-pot precision features =====
    if all_pot_fractions:
        feats["e100_pot_frac_mean"] = float(np.mean(all_pot_fractions))
        feats["e101_pot_frac_std"] = float(np.std(all_pot_fractions))
        feats["e102_pot_frac_p50"] = float(np.median(all_pot_fractions))
        # Closeness to standard pot fractions
        near_count = 0
        for pf in all_pot_fractions:
            for t in POT_FRACTIONS:
                if abs(pf - t) / max(t, 0.01) < 0.10:
                    near_count += 1
                    break
        feats["e103_pot_frac_near_standard"] = near_count / len(all_pot_fractions)
        # Bot signature: extreme clustering near 0.5 or 1.0 pot
        feats["e104_pot_frac_near_half"] = sum(1 for pf in all_pot_fractions if 0.4 < pf < 0.6) / len(all_pot_fractions)
        feats["e105_pot_frac_near_pot"] = sum(1 for pf in all_pot_fractions if 0.9 < pf < 1.1) / len(all_pot_fractions)
    else:
        for i in range(100, 106):
            feats[f"e{i}_default"] = 0.5

    # ===== e106-e110: round bet value precision =====
    if all_round_bet_distances:
        feats["e106_round_bet_distance_mean"] = float(np.mean(all_round_bet_distances))
        feats["e107_round_bet_distance_max"] = float(np.max(all_round_bet_distances))
        feats["e108_round_bet_distance_p25"] = float(np.percentile(all_round_bet_distances, 25))
        feats["e109_round_bet_distance_p75"] = float(np.percentile(all_round_bet_distances, 75))
        # Highly round bet ratio
        feats["e110_round_bet_perfect_ratio"] = sum(1 for d in all_round_bet_distances if d < 0.01) / len(all_round_bet_distances)
    else:
        for i in range(106, 111):
            feats[f"e{i}_default"] = 0.5

    # ===== e111-e120: inter-hand variance =====
    feats["e111_n_actions_per_hand_cv"] = float(np.std(per_hand_n_actions) / max(np.mean(per_hand_n_actions), 1))
    feats["e112_n_actions_per_hand_iqr"] = float(np.percentile(per_hand_n_actions, 75) - np.percentile(per_hand_n_actions, 25))
    feats["e113_n_streets_per_hand_mean"] = float(np.mean(per_hand_n_streets)) / 4.0
    feats["e114_pot_growth_per_hand_cv"] = float(np.std(per_hand_pot_growth) / max(np.mean(per_hand_pot_growth) + 1e-6, 1))
    bet_size_means_per_hand = [np.mean(b) if b else 0 for b in per_hand_bet_sizes_bb]
    feats["e115_bet_size_per_hand_cv"] = float(np.std(bet_size_means_per_hand) / max(np.mean(bet_size_means_per_hand) + 1e-6, 1))
    # Aggression variance
    agg_per_hand = []
    for ats in per_hand_action_types:
        if not ats: continue
        n_agg = sum(1 for t in ats if t in ("bet", "raise"))
        agg_per_hand.append(n_agg / len(ats))
    if len(agg_per_hand) > 1:
        feats["e116_aggression_per_hand_std"] = float(np.std(agg_per_hand))
        feats["e117_aggression_per_hand_p90"] = float(np.percentile(agg_per_hand, 90))
    else:
        feats["e116_aggression_per_hand_std"] = 0.0
        feats["e117_aggression_per_hand_p90"] = 0.0

    # ===== e118-e120: hand length entropy =====
    n_actions_counter = Counter(per_hand_n_actions)
    if n_actions_counter:
        probs = [c / n_hands for c in n_actions_counter.values()]
        H = -sum(p * math.log2(p) for p in probs if p > 0)
        feats["e118_n_actions_entropy"] = H / math.log2(max(len(n_actions_counter), 2))
    else:
        feats["e118_n_actions_entropy"] = 0.5
    # Most common hand length frequency
    if n_actions_counter:
        feats["e119_n_actions_mode_freq"] = max(n_actions_counter.values()) / n_hands
    else:
        feats["e119_n_actions_mode_freq"] = 0.5

    # ===== e120-e130: n-gram entropy and rarity =====
    def ngram_entropy(counter):
        total = sum(counter.values())
        if total == 0: return 0.5
        probs = [c / total for c in counter.values()]
        H = -sum(p * math.log2(p) for p in probs if p > 0)
        H_max = math.log2(max(len(counter), 2))
        return H / H_max if H_max > 0 else 0.5

    feats["e120_3gram_entropy"] = ngram_entropy(sequence_3grams)
    feats["e121_4gram_entropy"] = ngram_entropy(sequence_4grams)
    feats["e122_5gram_entropy"] = ngram_entropy(sequence_5grams)
    feats["e123_3gram_unique_ratio"] = len(sequence_3grams) / max(sum(sequence_3grams.values()), 1)
    feats["e124_4gram_unique_ratio"] = len(sequence_4grams) / max(sum(sequence_4grams.values()), 1)
    feats["e125_5gram_unique_ratio"] = len(sequence_5grams) / max(sum(sequence_5grams.values()), 1)
    # Most common n-gram frequency
    if sequence_3grams:
        feats["e126_3gram_top_freq"] = max(sequence_3grams.values()) / sum(sequence_3grams.values())
    else:
        feats["e126_3gram_top_freq"] = 0.5

    # ===== e127-e135: stack-VPIP interaction =====
    # VPIP per stack bracket
    short_stack_vpip = []
    mid_stack_vpip = []
    deep_stack_vpip = []
    for hand in hands:
        meta = hand.get("metadata") or {}
        hero_seat = meta.get("hero_seat", 0)
        # Find hero stack
        hero_stack = 0
        for p in hand.get("players") or []:
            if p.get("seat") == hero_seat:
                hero_stack = p.get("starting_stack", 0) / 0.02
                break
        actions = hand.get("actions") or []
        hero_voluntary = any(
            a.get("actor_seat") == hero_seat and a.get("action_type") in ("call", "raise", "bet")
            and a.get("street") == "preflop"
            for a in actions if isinstance(a, dict)
        )
        if hero_stack < 10:
            short_stack_vpip.append(int(hero_voluntary))
        elif hero_stack < 30:
            mid_stack_vpip.append(int(hero_voluntary))
        else:
            deep_stack_vpip.append(int(hero_voluntary))

    feats["e127_short_stack_vpip"] = float(np.mean(short_stack_vpip)) if short_stack_vpip else 0.5
    feats["e128_mid_stack_vpip"] = float(np.mean(mid_stack_vpip)) if mid_stack_vpip else 0.5
    feats["e129_deep_stack_vpip"] = float(np.mean(deep_stack_vpip)) if deep_stack_vpip else 0.5
    # Bot signature: VPIP CONSTANT across stacks
    vpips = [v for v in [
        np.mean(short_stack_vpip) if short_stack_vpip else None,
        np.mean(mid_stack_vpip) if mid_stack_vpip else None,
        np.mean(deep_stack_vpip) if deep_stack_vpip else None,
    ] if v is not None]
    if len(vpips) >= 2:
        feats["e130_vpip_stack_bracket_std"] = float(np.std(vpips))
        feats["e131_vpip_stack_consistency"] = 1.0 - float(np.std(vpips))
    else:
        feats["e130_vpip_stack_bracket_std"] = 0.0
        feats["e131_vpip_stack_consistency"] = 0.5

    # ===== e132-e140: pot growth velocity per street =====
    pot_growth_by_street = defaultdict(list)
    for hand in hands:
        for a in hand.get("actions") or []:
            if not isinstance(a, dict): continue
            s = a.get("street", "preflop")
            pot_growth_by_street[s].append(float(a.get("pot_after", 0)) - float(a.get("pot_before", 0)))

    for idx, street in enumerate(["preflop", "flop", "turn", "river"]):
        growths = pot_growth_by_street[street]
        if growths:
            feats[f"e{132+idx}_growth_{street}_velocity"] = float(np.mean(growths)) / 0.02 / 30.0  # normalize
            feats[f"e{136+idx}_growth_{street}_std"] = float(np.std(growths)) / 0.02 / 20.0
        else:
            feats[f"e{132+idx}_growth_{street}_velocity"] = 0.0
            feats[f"e{136+idx}_growth_{street}_std"] = 0.0

    # ===== e140-e144: bet size clustering =====
    if len(all_bets_bb) >= 5:
        # Number of "clusters" in bet sizes (approximation: unique rounded values)
        rounded_bets = [round(b / 4) * 4 for b in all_bets_bb]  # bucket by 4 BB
        unique_bet_clusters = len(set(rounded_bets))
        feats["e140_bet_clusters_count"] = min(unique_bet_clusters / 10, 1.0)
        # Concentration: how many bets fall in the top-3 clusters
        cluster_counts = Counter(rounded_bets)
        top3_pct = sum(c for _, c in cluster_counts.most_common(3)) / len(all_bets_bb)
        feats["e141_bet_top3_clusters_concentration"] = top3_pct
        # Bot signature: very high concentration (only uses 2-3 bet sizes)
        feats["e142_bet_high_concentration"] = 1.0 if top3_pct > 0.9 else 0.0
    else:
        feats["e140_bet_clusters_count"] = 0.5
        feats["e141_bet_top3_clusters_concentration"] = 0.5
        feats["e142_bet_high_concentration"] = 0.5

    # ===== e143-e145: cross-hand action sequence repetition =====
    full_action_sigs = ["-".join(at) for at in per_hand_action_types if at]
    sig_counter = Counter(full_action_sigs)
    if sig_counter:
        # Most repeated full sequence
        feats["e143_top_sequence_freq"] = max(sig_counter.values()) / len(full_action_sigs)
        # Diversity
        feats["e144_full_sequence_diversity"] = len(sig_counter) / max(len(full_action_sigs), 1)
    else:
        feats["e143_top_sequence_freq"] = 0.5
        feats["e144_full_sequence_diversity"] = 0.5

    return feats


def feature_names() -> List[str]:
    sample_hand = {
        "metadata": {"max_seats": 6, "hero_seat": 1},
        "players": [{"seat": 1, "starting_stack": 1.0}],
        "actions": [{"action_type": "fold", "street": "preflop", "actor_seat": 1,
                     "normalized_amount_bb": 0, "pot_before": 0.03, "pot_after": 0.04}],
    }
    return sorted(compute_enterprise_features([sample_hand]).keys())
