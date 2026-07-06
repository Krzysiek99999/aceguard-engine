"""V25 — 24-feature behavioral detector (full plan 2026-05-31).

Research base:
  - PartyPoker game integrity blog (HUD-style stats)
  - GTO Wizard Fair Play Check (solver correlation hint)
  - PokerCopilot HUD reference (VPIP/PFR/cbet/AF)
  - Internal v5 sequence signals

24 features split:
  - 14 bot signals (positive contribution)
  - 6 human signals (NEGATIVE contribution — subtract)
  - 4 cross-features (interactions: position×VPIP, stack×AF, street×cbet, handstrength×action)

Output: raw_score in [0, 1]; higher = more bot-like.

Trained via scripts/miner_training/train_v25_dual.py with multi-task loss:
  - 50% benchmark_real (ground truth)
  - 50% live distillation (UID 160/252 picks as pseudo-labels)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple


# === Helpers ===
def _entropy(counts: Sequence[float]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _cv(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    if m <= 1e-9:
        return 0.0
    return _std(values) / m


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# === Per-hand parsing helpers ===
def _extract_hand_info(hand: dict, hero_seat: int) -> Dict:
    """Return per-hand digest from hero perspective."""
    actions = hand.get("actions") or []
    info = {
        "n_actions": len(actions),
        "hero_actions": [],
        "preflop_aggressive": False,
        "preflop_voluntary": False,
        "preflop_raise_size_bb": None,
        "cbet_opportunity": False,
        "cbet_made": False,
        "fold_to_cbet_opp": False,
        "fold_to_cbet": False,
        "bet_size_pot_fractions": [],
        "open_size_bb": None,
        "min_raise": False,
        "donk_bet": False,
        "check_raise": False,
        "limp_then_raise": False,
        "bet_types_per_street": defaultdict(list),
        "agg_count_per_street": defaultdict(int),
        "call_count_per_street": defaultdict(int),
        "all_action_types": [],
    }
    last_aggressive_actor = None
    hero_preflop_actions = []
    hero_postflop_aggressors = set()
    raised_preflop = False
    limped_preflop = False

    for a in actions:
        if not isinstance(a, dict):
            continue
        t = a.get("action_type", "?")
        street = a.get("street", "?")
        actor = a.get("actor_seat")
        amt_bb = a.get("normalized_amount_bb") or 0.0
        pot_before = a.get("pot_before", 0.0) or 0.0
        info["all_action_types"].append(t)
        info["bet_types_per_street"][street].append(t)

        # Pot fraction for bets/raises
        if t in ("bet", "raise") and pot_before > 0:
            try:
                pf = float(amt_bb) / max(float(pot_before * 25.0), 1.0)  # rough pot-fraction estimate
                info["bet_size_pot_fractions"].append(_clip(pf, 0, 5))
            except Exception:
                pass

        if actor == hero_seat:
            info["hero_actions"].append((street, t, float(amt_bb)))

            if street == "preflop":
                hero_preflop_actions.append((t, float(amt_bb)))
                if t in ("call", "raise", "bet"):
                    info["preflop_voluntary"] = True
                if t in ("raise", "bet"):
                    info["preflop_aggressive"] = True
                    raised_preflop = True
                    if info["preflop_raise_size_bb"] is None:
                        info["preflop_raise_size_bb"] = float(amt_bb)
                if t == "call" and last_aggressive_actor is None:
                    limped_preflop = True

            if street in ("flop", "turn", "river"):
                if t in ("bet", "raise"):
                    info["agg_count_per_street"][street] += 1
                    hero_postflop_aggressors.add(street)
                if t == "call":
                    info["call_count_per_street"][street] += 1

        if t in ("bet", "raise"):
            last_aggressive_actor = actor

    # C-bet opportunity: hero raised preflop AND saw flop
    if raised_preflop:
        flop_actions = info["bet_types_per_street"].get("flop", [])
        if flop_actions:
            info["cbet_opportunity"] = True
            # Did hero bet on flop?
            hero_flop_actions = [
                (t, amt) for (s, t, amt) in info["hero_actions"] if s == "flop"
            ]
            if hero_flop_actions and hero_flop_actions[0][0] in ("bet", "raise"):
                info["cbet_made"] = True

    # Open size in BB
    if info["preflop_raise_size_bb"] is not None:
        info["open_size_bb"] = info["preflop_raise_size_bb"]
        # Min-raise = exactly 2x BB
        if 1.9 <= info["preflop_raise_size_bb"] <= 2.1:
            info["min_raise"] = True

    # Donk bet: hero bet flop OOP without preflop initiative
    if not raised_preflop:
        hero_flop_actions = [(t, amt) for (s, t, amt) in info["hero_actions"] if s == "flop"]
        if hero_flop_actions and hero_flop_actions[0][0] == "bet":
            info["donk_bet"] = True

    # Check-raise: hero checked then raised on same street
    by_street = defaultdict(list)
    for (s, t, amt) in info["hero_actions"]:
        by_street[s].append(t)
    for s, acts in by_street.items():
        if "check" in acts and "raise" in acts:
            ci = acts.index("check")
            ri = acts.index("raise")
            if ci < ri:
                info["check_raise"] = True

    # Limp-then-raise (very human)
    if limped_preflop and any(t == "raise" for (t, _) in hero_preflop_actions):
        info["limp_then_raise"] = True

    return info


def _hand_signature(hand: dict) -> str:
    """Compact action+sizing signature for repetition detection."""
    actions = hand.get("actions") or []
    parts = []
    for a in actions:
        t = a.get("action_type", "?")
        amt = a.get("normalized_amount_bb") or 0.0
        if t in ("bet", "raise"):
            if amt < 1.0:
                b = "s"
            elif amt < 3.0:
                b = "m"
            elif amt < 8.0:
                b = "l"
            else:
                b = "x"
            parts.append(f"{t[0]}{b}")
        else:
            parts.append(t[0])
    return "-".join(parts)


# === The 24 features ===
def compute_features(hands: List[dict]) -> Dict[str, float]:
    """Compute all 24 v25 features for a chunk of hands (single player)."""
    n_hands = len(hands)
    if n_hands == 0:
        return {f"f{i}": 0.0 for i in range(24)}

    # Identify hero seat from first hand
    hero_seat = (hands[0].get("metadata") or {}).get("hero_seat", 1)

    # Extract per-hand info
    per_hand = [_extract_hand_info(h, hero_seat) for h in hands]
    signatures = [_hand_signature(h) for h in hands]

    # === BOT SIGNALS (1-14) ===
    # Pre-flop stats
    vpip_flags = [int(h["preflop_voluntary"]) for h in per_hand]
    pfr_flags = [int(h["preflop_aggressive"]) for h in per_hand]
    vpip_pct = sum(vpip_flags) / n_hands
    pfr_pct = sum(pfr_flags) / n_hands

    # f1: VPIP-PFR gap small (bot has VPIP≈PFR)
    vpip_pfr_gap = abs(vpip_pct - pfr_pct)
    f1_vpip_pfr_gap_tight = 1.0 - _clip(vpip_pfr_gap / 0.2, 0, 1)  # gap<0.2 → tight=high

    # f2: VPIP stability across blocks of 20 hands (or halves if <20)
    block_size = max(10, n_hands // 4)
    if n_hands >= block_size * 2:
        block_vpip = []
        for i in range(0, n_hands, block_size):
            block = vpip_flags[i : i + block_size]
            if block:
                block_vpip.append(sum(block) / len(block))
        f2_vpip_stability = 1.0 - _clip(_std(block_vpip) / 0.3, 0, 1)
    else:
        f2_vpip_stability = 0.5

    # f3: PFR stability
    if n_hands >= block_size * 2:
        block_pfr = []
        for i in range(0, n_hands, block_size):
            block = pfr_flags[i : i + block_size]
            if block:
                block_pfr.append(sum(block) / len(block))
        f3_pfr_stability = 1.0 - _clip(_std(block_pfr) / 0.3, 0, 1)
    else:
        f3_pfr_stability = 0.5

    # f4: C-bet frequency (when opp exists)
    cbet_opps = [h for h in per_hand if h["cbet_opportunity"]]
    cbets_made = [h for h in cbet_opps if h["cbet_made"]]
    if len(cbet_opps) >= 3:
        f4_cbet_freq = len(cbets_made) / len(cbet_opps)
        # bot extreme (very high or very low) is suspicious
        f4_cbet_extreme = max(f4_cbet_freq, 1.0 - f4_cbet_freq)
    else:
        f4_cbet_extreme = 0.5

    # f5: Aggression factor stability across streets
    af_per_street = []
    for street in ("flop", "turn", "river"):
        total_agg = sum(h["agg_count_per_street"].get(street, 0) for h in per_hand)
        total_call = sum(h["call_count_per_street"].get(street, 0) for h in per_hand)
        if total_call > 0:
            af_per_street.append(total_agg / total_call)
        elif total_agg > 0:
            af_per_street.append(5.0)  # very aggressive
    if len(af_per_street) >= 2:
        f5_af_stability = 1.0 - _clip(_cv(af_per_street) / 1.0, 0, 1)
    else:
        f5_af_stability = 0.5

    # f6: Bet sizing template match
    all_pot_fractions = []
    for h in per_hand:
        all_pot_fractions.extend(h["bet_size_pot_fractions"])
    if all_pot_fractions:
        templates = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0]
        n_near = 0
        for pf in all_pot_fractions:
            for t in templates:
                if abs(pf - t) / max(t, 0.01) < 0.10:
                    n_near += 1
                    break
        f6_template_match = n_near / len(all_pot_fractions)
    else:
        f6_template_match = 0.5

    # f7: Min-raise frequency (preflop)
    min_raises = sum(1 for h in per_hand if h["min_raise"])
    pfr_count = sum(pfr_flags)
    f7_min_raise_freq = min_raises / max(pfr_count, 1) if pfr_count > 0 else 0.0

    # f8: Open size consistency
    open_sizes = [h["open_size_bb"] for h in per_hand if h["open_size_bb"] is not None]
    if len(open_sizes) >= 3:
        f8_open_consistency = 1.0 - _clip(_cv(open_sizes) / 0.5, 0, 1)
    else:
        f8_open_consistency = 0.5

    # f9: Sequence repetition
    sig_counts = Counter(signatures)
    n_repeated = sum(c for c in sig_counts.values() if c > 1)
    f9_repetition = n_repeated / n_hands

    # f10: Unique signatures low
    f10_unique_lo = 1.0 - (len(sig_counts) / n_hands)

    # f11: Action entropy low
    all_action_types: List[str] = []
    for h in per_hand:
        all_action_types.extend(h["all_action_types"])
    action_counts = Counter(all_action_types)
    if action_counts:
        H = _entropy(list(action_counts.values()))
        H_max = math.log2(max(len(action_counts), 1))
        f11_entropy_lo = 1.0 - (H / max(H_max, 1e-6))
    else:
        f11_entropy_lo = 0.5

    # f12: Action count regularity (per hand)
    n_actions_per_hand = [h["n_actions"] for h in per_hand]
    if len(n_actions_per_hand) > 1:
        f12_regularity = 1.0 - _clip(_cv(n_actions_per_hand), 0, 1)
    else:
        f12_regularity = 0.5

    # f13: Fold-to-cbet stability (proxy: how varied fold response across hands)
    # Simple proxy: variance of agg_count in flop
    flop_aggs = [h["agg_count_per_street"].get("flop", 0) for h in per_hand]
    if flop_aggs:
        f13_fold_to_cbet_stability = 1.0 - _clip(_std(flop_aggs) / 1.0, 0, 1)
    else:
        f13_fold_to_cbet_stability = 0.5

    # f14: 3bet sizing precision (proxy: open_size variance for raises after raise)
    # Limited data, use open_size precision as proxy
    if len(open_sizes) >= 3:
        # check how many are exactly 2.5, 3.0, 3.5 BB (common bot opens)
        exact_count = sum(1 for s in open_sizes if any(abs(s - t) / t < 0.05 for t in [2.0, 2.5, 3.0, 3.5]))
        f14_3bet_precision = exact_count / len(open_sizes)
    else:
        f14_3bet_precision = 0.5

    # === HUMAN SIGNALS (15-20) — NEGATIVE contribution ===
    # f15: Bet size variance (high variance → human)
    if all_pot_fractions:
        f15_bet_size_variance = _clip(_std(all_pot_fractions) / 0.5, 0, 1)
    else:
        f15_bet_size_variance = 0.0

    # f16: VPIP drift in chunk (first half vs second half)
    half = n_hands // 2
    if half >= 5:
        vpip_h1 = sum(vpip_flags[:half]) / half
        vpip_h2 = sum(vpip_flags[half:]) / (n_hands - half)
        f16_vpip_drift = _clip(abs(vpip_h1 - vpip_h2) / 0.3, 0, 1)
    else:
        f16_vpip_drift = 0.0

    # f17: Unusual lines (limp-reraise, donk-bet, check-raise count)
    unusual_count = sum(
        int(h["limp_then_raise"]) + int(h["donk_bet"]) + int(h["check_raise"])
        for h in per_hand
    )
    f17_unusual_lines = _clip(unusual_count / max(n_hands * 0.2, 1), 0, 1)

    # f18: Non-templated bet fractions (high non-match → human)
    f18_non_templated = 1.0 - f6_template_match

    # f19: Stack-aware deviation (varies in SPR responses)
    # Proxy: variance of aggression across hands
    f19_stack_deviation = _cv([h["n_actions"] for h in per_hand]) if n_hands > 1 else 0.0
    f19_stack_deviation = _clip(f19_stack_deviation, 0, 1)

    # f20: Hand strength compliance (if showdown data available)
    # Proxy: if showdown data exists, did hero with strong hand also bet aggressively?
    showdown_hands = [h for h in hands if (h.get("outcome") or {}).get("showdown")]
    f20_hand_strength_correlation = 0.5  # placeholder if no showdown data
    if showdown_hands:
        # Simple heuristic — bot plays "correctly" by hand strength; human deviates
        f20_hand_strength_correlation = 0.7  # without solver, just baseline

    # === CROSS FEATURES (21-24) ===
    # f21: Position-aware VPIP variance
    # Group hands by hero position (button vs blinds via hero_seat vs button_seat)
    pos_vpip = defaultdict(list)
    for i, h in enumerate(hands):
        meta = h.get("metadata") or {}
        button_seat = meta.get("button_seat", 0)
        hero_pos = "BTN" if hero_seat == button_seat else "BB" if hero_seat == button_seat - 1 else "OTHER"
        pos_vpip[hero_pos].append(vpip_flags[i])
    if len(pos_vpip) >= 2:
        pos_means = [sum(v) / len(v) for v in pos_vpip.values() if len(v) >= 3]
        if len(pos_means) >= 2:
            # bot: position VPIP differs predictably (high BTN VPIP, low UTG)
            # human: variance is messier
            f21_position_vpip_consistency = 1.0 - _clip(_std(pos_means) / 0.4, 0, 1)
        else:
            f21_position_vpip_consistency = 0.5
    else:
        f21_position_vpip_consistency = 0.5

    # f22: Stack-depth × aggression (stack varies, action should too if human)
    starting_stacks = []
    for h in hands:
        players = h.get("players") or []
        for p in players:
            if p.get("seat") == hero_seat:
                ss = p.get("starting_stack")
                if ss is not None:
                    starting_stacks.append(float(ss))
                break
    if len(starting_stacks) >= 5 and len(n_actions_per_hand) >= 5:
        # correlation between stack and actions: human bot drink more
        # simplified: variance ratio
        stack_cv = _cv(starting_stacks)
        action_cv = _cv(n_actions_per_hand)
        if stack_cv > 0.1:
            f22_stack_aggression_corr = _clip(1.0 - abs(stack_cv - action_cv), 0, 1)
        else:
            f22_stack_aggression_corr = 0.5
    else:
        f22_stack_aggression_corr = 0.5

    # f23: Street × cbet pattern (cbet consistency)
    flop_cbet_count = sum(1 for h in per_hand if h["cbet_made"])
    turn_agg = sum(h["agg_count_per_street"].get("turn", 0) for h in per_hand)
    if len(cbet_opps) >= 3 and turn_agg > 0:
        # bot: predictable continuation rate
        ratio = turn_agg / max(flop_cbet_count, 1)
        f23_street_cbet_pattern = 1.0 - _clip(abs(ratio - 0.5) / 0.5, 0, 1)
    else:
        f23_street_cbet_pattern = 0.5

    # f24: Hand strength × action correlation (showdown derived)
    if showdown_hands:
        # Without solver, can only do baseline
        f24_hand_action_corr = 0.5
    else:
        f24_hand_action_corr = 0.5

    return {
        "f1_vpip_pfr_gap_tight": f1_vpip_pfr_gap_tight,
        "f2_vpip_stability": f2_vpip_stability,
        "f3_pfr_stability": f3_pfr_stability,
        "f4_cbet_extreme": f4_cbet_extreme,
        "f5_af_stability": f5_af_stability,
        "f6_template_match": f6_template_match,
        "f7_min_raise_freq": f7_min_raise_freq,
        "f8_open_consistency": f8_open_consistency,
        "f9_repetition": f9_repetition,
        "f10_unique_lo": f10_unique_lo,
        "f11_entropy_lo": f11_entropy_lo,
        "f12_regularity": f12_regularity,
        "f13_fold_to_cbet_stability": f13_fold_to_cbet_stability,
        "f14_3bet_precision": f14_3bet_precision,
        "f15_bet_size_variance": f15_bet_size_variance,
        "f16_vpip_drift": f16_vpip_drift,
        "f17_unusual_lines": f17_unusual_lines,
        "f18_non_templated": f18_non_templated,
        "f19_stack_deviation": f19_stack_deviation,
        "f20_hand_strength_correlation": f20_hand_strength_correlation,
        "f21_position_vpip_consistency": f21_position_vpip_consistency,
        "f22_stack_aggression_corr": f22_stack_aggression_corr,
        "f23_street_cbet_pattern": f23_street_cbet_pattern,
        "f24_hand_action_corr": f24_hand_action_corr,
    }


# === Default heuristic weights (BEFORE training) ===
# Tuned via intuition; replaced by trained weights after F1.2
DEFAULT_WEIGHTS_BOT = {
    "f1_vpip_pfr_gap_tight": 0.06,
    "f2_vpip_stability": 0.06,
    "f3_pfr_stability": 0.05,
    "f4_cbet_extreme": 0.06,
    "f5_af_stability": 0.05,
    "f6_template_match": 0.08,
    "f7_min_raise_freq": 0.04,
    "f8_open_consistency": 0.05,
    "f9_repetition": 0.08,
    "f10_unique_lo": 0.07,
    "f11_entropy_lo": 0.05,
    "f12_regularity": 0.04,
    "f13_fold_to_cbet_stability": 0.04,
    "f14_3bet_precision": 0.03,
}
DEFAULT_WEIGHTS_HUMAN = {
    "f15_bet_size_variance": -0.06,
    "f16_vpip_drift": -0.05,
    "f17_unusual_lines": -0.05,
    "f18_non_templated": -0.04,
    "f19_stack_deviation": -0.04,
    "f20_hand_strength_correlation": -0.02,
}
DEFAULT_WEIGHTS_CROSS = {
    "f21_position_vpip_consistency": 0.04,
    "f22_stack_aggression_corr": 0.02,
    "f23_street_cbet_pattern": 0.03,
    "f24_hand_action_corr": 0.02,
}


def score_chunk_v25(hands: List[dict], weights: Dict[str, float] = None) -> Tuple[float, Dict[str, float]]:
    """Compute v25 raw score for a chunk. Returns (score, features_dict)."""
    if not hands:
        return 0.5, {}

    features = compute_features(hands)
    if weights is None:
        weights = {**DEFAULT_WEIGHTS_BOT, **DEFAULT_WEIGHTS_HUMAN, **DEFAULT_WEIGHTS_CROSS}

    score = 0.5  # neutral baseline
    for fname, fval in features.items():
        w = weights.get(fname, 0.0)
        score += w * (fval - 0.5)  # center features around 0.5
    return _clip(score, 0.0, 1.0), features


def score_chunks_v25(chunks: List[List[dict]], weights: Dict[str, float] = None) -> List[float]:
    """Score a list of chunks. Returns list of floats in [0,1]."""
    return [score_chunk_v25(c, weights=weights)[0] for c in chunks]
