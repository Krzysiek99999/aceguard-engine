"""V13 robust feature extractor — independent implementation.

NIE importuje poker44_ml/* z Travis861/Poker44_v1. Re-implementacja od zera na bazie:
- jawnej specyfikacji DetectionSynapse z poker44/validator/synapse.py
- spec payload_view.py (post-0.1.20 tighten: 8 action types, 16 bb buckets, seat aliasing)
- własnego inwentarza co JEST DOSTĘPNE post-tighten (LIVE_CHUNK_ANALYSIS_POST_TIGHTEN.md)

Design:
- ~25 per-hand features × 8 chunk-level statistics → ~200 base features
- ~15 chunk-level signature/consistency features
- Total ~215 features per chunk

Polarność: high score = bot (sklearn standard label=1=bot).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


# === Helpers ===

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None: return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _entropy_normalized(seq: list) -> float:
    """Shannon entropy normalized to [0, 1] (divided by log of unique count)."""
    if not seq: return 0.0
    counts = Counter(seq)
    total = float(sum(counts.values()))
    if total <= 0 or len(counts) <= 1: return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent / math.log(len(counts))


def _quantile(values: list, q: float) -> float:
    if not values: return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1: return xs[0]
    pos = max(0.0, min(1.0, float(q))) * (len(xs) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi: return xs[lo]
    return xs[lo] * (1.0 - (pos - lo)) + xs[hi] * (pos - lo)


def _mean(v: list) -> float:
    return _safe_div(sum(v), len(v))


def _std(v: list) -> float:
    if not v: return 0.0
    m = _mean(v)
    return math.sqrt(max(0.0, _mean([(x - m) ** 2 for x in v])))


def _longest_run(seq: list) -> int:
    if not seq: return 0
    longest = cur = 1
    for a, b in zip(seq, seq[1:]):
        if a == b:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return longest


def _bb_bucket(value: float) -> str:
    """Discrete bucket (zgodne z payload_view _VISIBLE_BB_BUCKETS quantization)."""
    if value <= 0: return 'z'
    if value <= 1.0: return 'xs'
    if value <= 2.0: return 's'
    if value <= 4.0: return 'm'
    if value <= 8.0: return 'l'
    if value <= 24.0: return 'xl'
    return 'xxl'


# === Per-hand features (25) ===

def _hand_features(hand: dict) -> dict:
    metadata = hand.get('metadata') or {}
    players = hand.get('players') or []
    streets = hand.get('streets') or []
    actions = hand.get('actions') or []
    outcome = hand.get('outcome') or {}

    hero_seat = _safe_int(metadata.get('hero_seat'), 0)
    button_seat = _safe_int(metadata.get('button_seat'), 0)
    n_players = float(len(players))
    n_streets = float(len(streets))
    n_actions = float(len(actions))

    action_types, actor_seats, street_names = [], [], []
    amounts_bb, pot_before_bb, pot_after_bb, stack_bb = [], [], [], []
    raise_to_count = call_to_count = 0

    for p in players:
        if isinstance(p, dict):
            stack_bb.append(_safe_div(_safe_float(p.get('starting_stack'), 0.0), 0.02))

    for a in actions:
        if not isinstance(a, dict): continue
        at = str(a.get('action_type') or '').lower().strip()
        actor = _safe_int(a.get('actor_seat'), 0)
        street = str(a.get('street') or '').lower().strip()
        amt = _safe_float(a.get('normalized_amount_bb'), 0.0)
        pb = _safe_div(_safe_float(a.get('pot_before'), 0.0), 0.02)
        pa = _safe_div(_safe_float(a.get('pot_after'), 0.0), 0.02)
        action_types.append(at)
        if actor > 0: actor_seats.append(actor)
        street_names.append(street)
        amounts_bb.append(max(0.0, amt))
        pot_before_bb.append(max(0.0, pb))
        pot_after_bb.append(max(0.0, pa))
        raise_to_count += int(a.get('raise_to') is not None)
        call_to_count += int(a.get('call_to') is not None)

    counts = Counter(action_types)
    # Meaningful = decisional actions (po tighten: bez all_in, bez blinds/ante system actions)
    meaningful = max(1, counts.get('call', 0) + counts.get('check', 0)
                     + counts.get('bet', 0) + counts.get('raise', 0) + counts.get('fold', 0))
    aggressive = counts.get('bet', 0) + counts.get('raise', 0)
    passive = counts.get('call', 0) + counts.get('check', 0)
    preflop_n = sum(1 for s in street_names if s == 'preflop')
    postflop_n = sum(1 for s in street_names if s not in {'', 'preflop'})
    hero_acts = sum(1 for s in actor_seats if hero_seat > 0 and s == hero_seat)
    pot_delta = [max(0.0, pa - pb) for pa, pb in zip(pot_after_bb, pot_before_bb)]
    monotonic_steps = sum(1 for a, b in zip(pot_after_bb, pot_after_bb[1:]) if b + 1e-9 >= a)

    return {
        'h_n_players': n_players,
        'h_n_streets': n_streets,
        'h_n_actions': n_actions,
        'h_call_share': _safe_div(counts.get('call', 0), meaningful),
        'h_check_share': _safe_div(counts.get('check', 0), meaningful),
        'h_fold_share': _safe_div(counts.get('fold', 0), meaningful),
        'h_bet_share': _safe_div(counts.get('bet', 0), meaningful),
        'h_raise_share': _safe_div(counts.get('raise', 0), meaningful),
        'h_aggression_share': _safe_div(aggressive, max(1, meaningful)),
        'h_passive_share': _safe_div(passive, max(1, meaningful)),
        'h_preflop_share': _safe_div(preflop_n, max(1, n_actions)),
        'h_postflop_share': _safe_div(postflop_n, max(1, n_actions)),
        'h_action_entropy': _entropy_normalized(action_types),
        'h_actor_entropy': _entropy_normalized(actor_seats),
        'h_unique_actor_share': _safe_div(len(set(actor_seats)), max(1, n_players)),
        'h_action_run_share': _safe_div(_longest_run(action_types), max(1, len(action_types))),
        'h_actor_run_share': _safe_div(_longest_run(actor_seats), max(1, len(actor_seats))),
        'h_amount_mean_bb': _mean(amounts_bb),
        'h_amount_std_bb': _std(amounts_bb),
        'h_amount_max_bb': max(amounts_bb) if amounts_bb else 0.0,
        'h_nonzero_amount_share': _safe_div(sum(1 for v in amounts_bb if v > 0), max(1, len(amounts_bb))),
        'h_pot_growth_bb': (max(pot_after_bb) - min(pot_before_bb)) if pot_after_bb and pot_before_bb else 0.0,
        'h_pot_monotonic_share': _safe_div(monotonic_steps, max(1, len(pot_after_bb) - 1)),
        'h_showdown_flag': 1.0 if outcome.get('showdown') else 0.0,
        'h_hero_action_share': _safe_div(hero_acts, max(1, n_actions)),
    }


# === Chunk aggregations ===

_AGG_STATS = ['mean', 'std', 'min', 'max', 'q10', 'q50', 'q90', 'iqr']


def _aggregate(values: list[float], prefix: str, out: dict) -> None:
    out[f'{prefix}_mean'] = _mean(values)
    out[f'{prefix}_std'] = _std(values)
    out[f'{prefix}_min'] = min(values) if values else 0.0
    out[f'{prefix}_max'] = max(values) if values else 0.0
    out[f'{prefix}_q10'] = _quantile(values, 0.1)
    out[f'{prefix}_q50'] = _quantile(values, 0.5)
    out[f'{prefix}_q90'] = _quantile(values, 0.9)
    out[f'{prefix}_iqr'] = _quantile(values, 0.75) - _quantile(values, 0.25)


# === Chunk-level features (signature + consistency) ===

def chunk_features_v13(chunk: list[dict]) -> dict:
    """Main entry: return ~215 features for a chunk (list of hands)."""
    if not chunk:
        return {'hand_count': 0.0}

    out = {'hand_count': float(len(chunk))}
    per_hand = [_hand_features(h) for h in chunk]
    feature_names = sorted(per_hand[0].keys())  # 25 features

    # 25 per-hand × 8 stats = 200 features
    for name in feature_names:
        series = [float(f[name]) for f in per_hand]
        _aggregate(series, name, out)

    # Signature features (15)
    action_sigs, actor_sigs, street_sigs, bucket_sigs = [], [], [], []
    high_aggr, low_action_ent, high_actor_ent, long_hand = 0, 0, 0, 0

    for hand, feats in zip(chunk, per_hand):
        actions = hand.get('actions') or []
        ats = tuple(str((a or {}).get('action_type') or '').lower().strip() for a in actions)
        acts = tuple(_safe_int((a or {}).get('actor_seat'), 0) for a in actions if _safe_int((a or {}).get('actor_seat'), 0) > 0)
        strs = tuple(str((a or {}).get('street') or '').lower().strip() for a in actions)
        bcks = tuple(_bb_bucket(_safe_float((a or {}).get('normalized_amount_bb'), 0.0)) for a in actions)
        action_sigs.append(ats)
        actor_sigs.append(acts)
        street_sigs.append(strs)
        bucket_sigs.append(bcks)
        high_aggr += int(feats['h_aggression_share'] >= 0.35)
        low_action_ent += int(feats['h_action_entropy'] <= 0.35)
        high_actor_ent += int(feats['h_actor_entropy'] >= 0.75)
        long_hand += int(feats['h_n_actions'] >= 10.0)

    n = float(len(chunk))
    out['sig_action_top_share'] = _safe_div(max(Counter(action_sigs).values()), n)
    out['sig_action_unique_share'] = _safe_div(len(set(action_sigs)), n)
    out['sig_actor_top_share'] = _safe_div(max(Counter(actor_sigs).values()), n)
    out['sig_actor_unique_share'] = _safe_div(len(set(actor_sigs)), n)
    out['sig_street_top_share'] = _safe_div(max(Counter(street_sigs).values()), n)
    out['sig_street_unique_share'] = _safe_div(len(set(street_sigs)), n)
    out['sig_bucket_top_share'] = _safe_div(max(Counter(bucket_sigs).values()), n)
    out['sig_bucket_unique_share'] = _safe_div(len(set(bucket_sigs)), n)
    out['rate_high_aggression'] = _safe_div(high_aggr, n)
    out['rate_low_action_entropy'] = _safe_div(low_action_ent, n)
    out['rate_high_actor_entropy'] = _safe_div(high_actor_ent, n)
    out['rate_long_hand'] = _safe_div(long_hand, n)

    # Chunk-level meta (3 more)
    showdown_rate = _safe_div(sum(1 for h in chunk if (h.get('outcome') or {}).get('showdown')), n)
    out['chunk_showdown_rate'] = showdown_rate

    # Hand-length spread (chunk-level diversity)
    action_counts = [float(len((h.get('actions') or []))) for h in chunk]
    out['chunk_action_count_iqr'] = _quantile(action_counts, 0.75) - _quantile(action_counts, 0.25)
    out['chunk_action_count_std'] = _std(action_counts)

    return out


FEATURE_NAMES_V13: list[str] | None = None


def get_feature_names() -> list[str]:
    """Deterministic ordering of features (run once on synthetic input)."""
    global FEATURE_NAMES_V13
    if FEATURE_NAMES_V13 is not None:
        return FEATURE_NAMES_V13
    # Synthesize minimal chunk to get all keys
    synth_hand = {
        'metadata': {'hero_seat': 1, 'button_seat': 2, 'max_seats': 6},
        'players': [{'starting_stack': 100.0}] * 6,
        'streets': ['preflop', 'flop'],
        'actions': [
            {'action_type': 'fold', 'actor_seat': 1, 'street': 'preflop',
             'normalized_amount_bb': 1.0, 'pot_before': 0.03, 'pot_after': 0.03},
        ],
        'outcome': {'showdown': True},
    }
    feats = chunk_features_v13([synth_hand, synth_hand])
    FEATURE_NAMES_V13 = sorted(feats.keys())
    return FEATURE_NAMES_V13
