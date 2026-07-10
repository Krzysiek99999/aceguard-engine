"""Temporal consistency features for Poker44 live-sized chunks.

The extractor is deterministic and uses only miner-visible hand/action fields.
It complements the aggregate schema and hashed sequence features with explicit
session-stability signals: lag autocorrelation, quartile drift, bet/pot
clustering, street mix, and action-pattern concentration.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence


ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
AGGRESSIVE = {"bet", "raise"}
PASSIVE = {"check", "call"}
STREETS = ("preflop", "flop", "turn", "river")
BB_BUCKETS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if abs(float(den)) > 1e-12 else 0.0


def _action(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in ACTION_TYPES:
        return raw
    for name in ACTION_TYPES:
        if name in raw:
            return name
    return "check"


def _street(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in STREETS else ""


def _stats(values: Sequence[float]) -> dict[str, float]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return {name: 0.0 for name in ("mean", "std", "min", "max", "q10", "q50", "q90")}
    xs_sorted = sorted(xs)
    mean = sum(xs) / len(xs)
    var = sum((v - mean) * (v - mean) for v in xs) / max(len(xs), 1)

    def q(p: float) -> float:
        if len(xs_sorted) == 1:
            return xs_sorted[0]
        pos = min(max(p, 0.0), 1.0) * (len(xs_sorted) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs_sorted[lo]
        w = pos - lo
        return xs_sorted[lo] * (1.0 - w) + xs_sorted[hi] * w

    return {
        "mean": float(mean),
        "std": float(math.sqrt(max(0.0, var))),
        "min": float(xs_sorted[0]),
        "max": float(xs_sorted[-1]),
        "q10": float(q(0.10)),
        "q50": float(q(0.50)),
        "q90": float(q(0.90)),
    }


def _entropy_norm(values: Iterable[Any]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    counts = Counter(vals)
    if len(counts) <= 1:
        return 0.0
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)
    return _safe_div(entropy, math.log(len(counts)))


def _top_share(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    return max(Counter(values).values()) / len(values)


def _nearest_bucket_gap(value: float) -> float:
    if value <= 0.0:
        return 0.0
    return min(abs(float(value) - bucket) for bucket in BB_BUCKETS)


def _bucket_index(value: float) -> int:
    if value <= 0.0:
        return -1
    return min(range(len(BB_BUCKETS)), key=lambda idx: abs(float(value) - BB_BUCKETS[idx]))


def _lag_autocorr(values: Sequence[float], lag: int) -> float:
    if len(values) < lag + 3:
        return 0.0
    x = [float(v) for v in values[:-lag]]
    y = [float(v) for v in values[lag:]]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return max(-1.0, min(1.0, _safe_div(num, dx * dy)))


def _trend_slope(values: Sequence[float]) -> float:
    if len(values) < 4:
        return 0.0
    n = len(values)
    mx = (n - 1) / 2.0
    my = sum(values) / n
    denom = sum((idx - mx) ** 2 for idx in range(n))
    slope = _safe_div(sum((idx - mx) * (values[idx] - my) for idx in range(n)), denom)
    return _safe_div(slope, abs(my) + 1e-9)


def _half_delta(values: Sequence[float]) -> float:
    if len(values) < 4:
        return 0.0
    mid = len(values) // 2
    left = sum(values[:mid]) / max(mid, 1)
    right = sum(values[mid:]) / max(len(values) - mid, 1)
    return abs(left - right)


def _quartile_features(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 8:
        return 0.0, 0.0
    n = len(values)
    means: list[float] = []
    for idx in range(4):
        start = int(round(idx * n / 4.0))
        end = int(round((idx + 1) * n / 4.0))
        part = values[start:end]
        means.append(sum(part) / max(len(part), 1))
    mean_stats = _stats(means)
    return float(mean_stats["std"]), float(means[-1] - means[0])


def _coeff_var(values: Sequence[float]) -> float:
    s = _stats(values)
    return _safe_div(s["std"], abs(s["mean"]) + 1e-9)


def _action_bigrams(actions: Sequence[str]) -> list[str]:
    return [f"{left}>{right}" for left, right in zip(actions, actions[1:])]


def _hand_row(hand: dict[str, Any]) -> dict[str, float]:
    metadata = hand.get("metadata") or {}
    hero_seat = _safe_int(metadata.get("hero_seat"), 0)
    actions_raw = [a for a in hand.get("actions") or [] if isinstance(a, dict)]
    actions = [_action(a.get("action_type")) for a in actions_raw]
    streets = [_street(a.get("street")) for a in actions_raw]
    actors = [_safe_int(a.get("actor_seat"), 0) for a in actions_raw]
    amounts = [max(0.0, _safe_float(a.get("normalized_amount_bb"), 0.0)) for a in actions_raw]
    bucket_idxs = [_bucket_index(v) for v in amounts if v > 0.0]

    pot_ratios: list[float] = []
    hero_pot_ratios: list[float] = []
    pot_growth: list[float] = []
    for action, amount, actor in zip(actions_raw, amounts, actors, strict=True):
        pot_before = max(0.0, _safe_float(action.get("pot_before"), 0.0))
        pot_after = max(0.0, _safe_float(action.get("pot_after"), 0.0))
        if pot_before > 0.0 and amount > 0.0:
            ratio = amount / pot_before
            pot_ratios.append(ratio)
            if hero_seat and actor == hero_seat:
                hero_pot_ratios.append(ratio)
        if pot_after > 0.0 or pot_before > 0.0:
            pot_growth.append(max(0.0, pot_after - pot_before))

    n = max(len(actions), 1)
    action_counts = Counter(actions)
    aggro_n = sum(action_counts[a] for a in AGGRESSIVE)
    passive_n = sum(action_counts[a] for a in PASSIVE)
    hero_actions = [a for a, actor in zip(actions, actors, strict=True) if hero_seat and actor == hero_seat]
    hero_aggro = sum(1 for a in hero_actions if a in AGGRESSIVE)
    hero_folds = sum(1 for a in hero_actions if a == "fold")
    actor_switches = sum(1 for left, right in zip(actors, actors[1:]) if left != right)

    out: dict[str, float] = {
        "n_actions": float(len(actions)),
        "frac_fold": action_counts["fold"] / n,
        "frac_check": action_counts["check"] / n,
        "frac_call": action_counts["call"] / n,
        "frac_bet": action_counts["bet"] / n,
        "frac_raise": action_counts["raise"] / n,
        "action_entropy": _entropy_norm(actions),
        "aggression_rate": aggro_n / n,
        "passive_rate": passive_n / n,
        "aggression_factor": _safe_div(aggro_n, passive_n + action_counts["fold"]),
        "hero_participation": len(hero_actions) / n,
        "hero_aggression_rate": _safe_div(hero_aggro, len(hero_actions)),
        "hero_fold_rate": _safe_div(hero_folds, len(hero_actions)),
        "actor_switch_rate": _safe_div(actor_switches, max(len(actors) - 1, 1)),
        "unique_actor_share": _safe_div(len(set(actor for actor in actors if actor > 0)), max(len(actors), 1)),
        "size_cv": _coeff_var([v for v in amounts if v > 0.0]),
        "size_bucket_entropy": _entropy_norm(bucket_idxs),
        "size_bucket_top_share": _top_share(bucket_idxs),
        "size_bucket_snap_gap": _stats([_nearest_bucket_gap(v) for v in amounts if v > 0.0])["mean"],
        "bet_pot_ratio_cv": _coeff_var(pot_ratios),
        "bet_pot_ratio_top_cluster": _top_share([round(v, 1) for v in pot_ratios]),
        "hero_bet_pot_ratio_cv": _coeff_var(hero_pot_ratios),
        "pot_growth_mean": _stats(pot_growth)["mean"],
        "reaches_flop": float("flop" in streets or "turn" in streets or "river" in streets),
        "reaches_turn": float("turn" in streets or "river" in streets),
        "reaches_river": float("river" in streets),
    }
    for street in STREETS:
        out[f"street_{street}_share"] = streets.count(street) / n
    return out


def _add_series_features(out: dict[str, float], name: str, values: Sequence[float]) -> None:
    for stat, value in _stats(values).items():
        out[f"{name}_{stat}"] = value
    out[f"{name}_lag1"] = _lag_autocorr(values, 1)
    out[f"{name}_lag2"] = _lag_autocorr(values, 2)
    out[f"{name}_lag3"] = _lag_autocorr(values, 3)
    out[f"{name}_trend"] = _trend_slope(values)
    out[f"{name}_half_delta"] = _half_delta(values)
    qstd, qdelta = _quartile_features(values)
    out[f"{name}_quartile_std"] = qstd
    out[f"{name}_q4_minus_q1"] = qdelta


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    hands = [hand for hand in chunk if isinstance(hand, dict)]
    out: dict[str, float] = {"tc_hand_count": float(len(hands))}
    if not hands:
        return out

    rows = [_hand_row(hand) for hand in hands]
    series_names = sorted(rows[0])
    for name in series_names:
        _add_series_features(out, f"tc_{name}", [row[name] for row in rows])

    hand_action_sigs: list[tuple[str, ...]] = []
    hand_street_sigs: list[tuple[str, ...]] = []
    hand_bucket_sigs: list[tuple[int, ...]] = []
    all_actions: list[str] = []
    all_streets: list[str] = []
    all_bucket_idxs: list[int] = []
    all_actor_roles: list[str] = []
    all_pot_ratios: list[float] = []

    for hand in hands:
        metadata = hand.get("metadata") or {}
        hero_seat = _safe_int(metadata.get("hero_seat"), 0)
        actions_raw = [a for a in hand.get("actions") or [] if isinstance(a, dict)]
        actions = tuple(_action(a.get("action_type")) for a in actions_raw)
        streets = tuple(_street(a.get("street")) for a in actions_raw)
        bucket_idxs = tuple(
            _bucket_index(max(0.0, _safe_float(a.get("normalized_amount_bb"), 0.0)))
            for a in actions_raw
        )
        roles = tuple(
            "H" if hero_seat and _safe_int(a.get("actor_seat"), 0) == hero_seat else "O"
            for a in actions_raw
        )
        hand_action_sigs.append(actions)
        hand_street_sigs.append(streets)
        hand_bucket_sigs.append(bucket_idxs)
        all_actions.extend(actions)
        all_streets.extend(streets)
        all_bucket_idxs.extend(idx for idx in bucket_idxs if idx >= 0)
        all_actor_roles.extend(roles)
        for action in actions_raw:
            amount = max(0.0, _safe_float(action.get("normalized_amount_bb"), 0.0))
            pot_before = max(0.0, _safe_float(action.get("pot_before"), 0.0))
            if amount > 0.0 and pot_before > 0.0:
                all_pot_ratios.append(amount / pot_before)

    n_hands = max(len(hands), 1)
    bigrams = _action_bigrams(all_actions)
    out.update(
        {
            "tc_action_signature_top_share": _top_share(hand_action_sigs),
            "tc_action_signature_unique_share": len(set(hand_action_sigs)) / n_hands,
            "tc_street_signature_top_share": _top_share(hand_street_sigs),
            "tc_street_signature_unique_share": len(set(hand_street_sigs)) / n_hands,
            "tc_bucket_signature_top_share": _top_share(hand_bucket_sigs),
            "tc_bucket_signature_unique_share": len(set(hand_bucket_sigs)) / n_hands,
            "tc_global_action_entropy": _entropy_norm(all_actions),
            "tc_global_street_entropy": _entropy_norm(all_streets),
            "tc_global_bucket_entropy": _entropy_norm(all_bucket_idxs),
            "tc_global_actor_role_entropy": _entropy_norm(all_actor_roles),
            "tc_action_bigram_entropy": _entropy_norm(bigrams),
            "tc_action_bigram_top_share": _top_share(bigrams),
            "tc_bet_pot_ratio_cv": _coeff_var(all_pot_ratios),
            "tc_bet_pot_ratio_cluster_share": _top_share([round(v, 1) for v in all_pot_ratios]),
        }
    )
    return out
