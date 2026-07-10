from __future__ import annotations

import gzip
import math
from collections import Counter
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p + 1e-12)
    return _safe_div(ent, math.log(len(counts)))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _mean(values: list[float]) -> float:
    return _safe_div(sum(values), len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(max(0.0, _mean([(v - m) * (v - m) for v in values])))


def _max_run_share(values: list[Any]) -> float:
    if not values:
        return 0.0
    longest = 1
    cur = 1
    for prev, cur_value in zip(values, values[1:]):
        if prev == cur_value:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return _safe_div(longest, len(values))


def _amount_bucket(value: float) -> str:
    if value <= 0.0:
        return "z"
    if value <= 0.5:
        return "xs"
    if value <= 1.0:
        return "s"
    if value <= 2.0:
        return "m"
    if value <= 5.0:
        return "l"
    return "xl"


def _ngram_set(values: tuple[Any, ...], size: int) -> frozenset[tuple[Any, ...]]:
    if len(values) < size:
        return frozenset()
    return frozenset(tuple(values[i : i + size]) for i in range(len(values) - size + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _lz_complexity_ratio(text: str) -> float:
    n = len(text)
    if n <= 1:
        return 0.0
    seen: set[str] = set()
    pos = 0
    pieces = 0
    while pos < n:
        end = pos + 1
        while end <= n and text[pos:end] in seen:
            end += 1
        seen.add(text[pos:end])
        pieces += 1
        pos = end
    normalizer = n / max(math.log2(n), 1.0)
    return _safe_div(float(pieces), normalizer)


def _conditional_entropy(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    pairs = Counter(zip(tokens[:-1], tokens[1:]))
    prev = Counter(tokens[:-1])
    total = float(len(tokens) - 1)
    value = 0.0
    for (head, _tail), count in pairs.items():
        p_pair = count / total
        p_next = count / max(prev[head], 1)
        value -= p_pair * math.log2(p_next)
    return value


def _lag_autocorr(values: list[float], lag: int) -> float:
    if len(values) < lag + 3:
        return 0.0
    left = [float(v) for v in values[:-lag]]
    right = [float(v) for v in values[lag:]]
    mean_left = _mean(left)
    mean_right = _mean(right)
    num = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    den_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    den_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    return max(-1.0, min(1.0, _safe_div(num, den_left * den_right)))


def _half_delta(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    mid = len(values) // 2
    return abs(_mean(values[:mid]) - _mean(values[mid:]))


def _trend_slope(values: list[float]) -> float:
    n = len(values)
    if n < 4:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = _mean(values)
    numerator = sum((idx - mean_x) * (float(value) - mean_y) for idx, value in enumerate(values))
    denominator = sum((idx - mean_x) ** 2 for idx in range(n))
    return _safe_div(numerator, denominator * (abs(mean_y) + 1e-9))


def _temporal_series_features(prefix: str, values: list[float], out: dict[str, float]) -> None:
    out[f"{prefix}_lag1_autocorr"] = _lag_autocorr(values, 1)
    out[f"{prefix}_lag2_autocorr"] = _lag_autocorr(values, 2)
    out[f"{prefix}_lag3_autocorr"] = _lag_autocorr(values, 3)
    out[f"{prefix}_half_delta"] = _half_delta(values)
    out[f"{prefix}_trend_slope"] = _trend_slope(values)


def _repeat_pattern_features(
    action_signatures: list[tuple[str, ...]],
    amount_bucket_signatures: list[tuple[str, ...]],
    street_signatures: list[tuple[str, ...]],
) -> dict[str, float]:
    total = len(action_signatures)
    if total < 2:
        return {
            "repeat_pair_jaccard_mean": 0.0,
            "repeat_effective_diversity": 1.0,
            "repeat_exact_action_share": 0.0,
            "repeat_exact_rich_share": 0.0,
            "repeat_gzip_ratio": 1.0,
            "repeat_lz_complexity": 0.0,
            "repeat_transition_entropy": 0.0,
        }

    rich_signatures = [
        tuple(f"{street}|{action}|{bucket}" for street, action, bucket in zip(streets, actions, buckets))
        for streets, actions, buckets in zip(
            street_signatures,
            action_signatures,
            amount_bucket_signatures,
        )
    ]

    stride = max(1, total // 60)
    sampled = list(range(0, total, stride))[:60]
    bigrams = [_ngram_set(rich_signatures[idx], 2) for idx in sampled]

    similarities: list[float] = []
    row_mass = [1.0 for _ in bigrams]
    for i in range(len(bigrams)):
        for j in range(i + 1, len(bigrams)):
            score = _jaccard(bigrams[i], bigrams[j])
            similarities.append(score)
            row_mass[i] += score
            row_mass[j] += score

    mass_sum = sum(row_mass)
    if mass_sum > 0:
        weights = [value / mass_sum for value in row_mass]
        entropy = -sum(weight * math.log(weight) for weight in weights if weight > 0)
        effective_diversity = _safe_div(math.exp(entropy), len(row_mass))
    else:
        effective_diversity = 1.0

    hand_strings = ["".join(token[:1] for token in signature) or "-" for signature in rich_signatures]
    joined = "#".join(hand_strings).encode()
    whole_len = len(gzip.compress(joined, compresslevel=5))
    part_len = sum(len(gzip.compress(item.encode(), compresslevel=5)) for item in hand_strings)

    flat_actions = [action for signature in action_signatures for action in signature]
    flat_text = "".join((action[:1] or "?") for action in flat_actions)

    return {
        "repeat_pair_jaccard_mean": _mean(similarities),
        "repeat_effective_diversity": _clamp01(effective_diversity),
        "repeat_exact_action_share": 1.0 - _safe_div(len(set(action_signatures)), total),
        "repeat_exact_rich_share": 1.0 - _safe_div(len(set(rich_signatures)), total),
        "repeat_gzip_ratio": _safe_div(float(whole_len), float(part_len)),
        "repeat_lz_complexity": _lz_complexity_ratio(flat_text[:300]),
        "repeat_transition_entropy": _conditional_entropy(flat_actions[:4000]),
    }


def _hand_features(hand: dict[str, Any]) -> dict[str, float]:
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    actions = hand.get("actions") or []

    max_seats = max(1, _safe_int(metadata.get("max_seats"), 6))
    hero_seat = _safe_int(metadata.get("hero_seat"), 0)
    button_seat = _safe_int(metadata.get("button_seat"), 0)
    player_count = float(len(players))
    street_count = float(len(streets))
    action_count = float(len(actions))

    action_types: list[str] = []
    actor_seats: list[int] = []
    street_names: list[str] = []
    amount_bb: list[float] = []
    pot_before: list[float] = []
    pot_after: list[float] = []
    stack_bb: list[float] = []
    raise_to_present = 0
    call_to_present = 0

    for player in players:
        if not isinstance(player, dict):
            continue
        stack_bb.append(_safe_div(_safe_float(player.get("starting_stack"), 0.0), 0.02))

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type") or "").lower().strip()
        actor = _safe_int(action.get("actor_seat"), 0)
        street = str(action.get("street") or "").lower().strip()
        amt = _safe_float(action.get("normalized_amount_bb"), 0.0)
        pb = _safe_div(_safe_float(action.get("pot_before"), 0.0), 0.02)
        pa = _safe_div(_safe_float(action.get("pot_after"), 0.0), 0.02)

        action_types.append(action_type)
        if actor > 0:
            actor_seats.append(actor)
        street_names.append(street)
        amount_bb.append(max(0.0, amt))
        pot_before.append(max(0.0, pb))
        pot_after.append(max(0.0, pa))
        raise_to_present += int(action.get("raise_to") is not None)
        call_to_present += int(action.get("call_to") is not None)

    counts = Counter(action_types)
    meaningful = max(
        counts.get("call", 0)
        + counts.get("check", 0)
        + counts.get("bet", 0)
        + counts.get("raise", 0)
        + counts.get("fold", 0),
        1,
    )
    aggressive = counts.get("bet", 0) + counts.get("raise", 0)
    passive = counts.get("call", 0) + counts.get("check", 0)

    preflop_n = sum(1 for s in street_names if s == "preflop")
    postflop_n = sum(1 for s in street_names if s not in {"", "preflop"})
    nonzero_amount = sum(1 for v in amount_bb if v > 0.0)
    hero_actions = sum(1 for s in actor_seats if s == hero_seat and hero_seat > 0)
    button_actions = sum(1 for s in actor_seats if s == button_seat and button_seat > 0)

    pot_delta = [max(0.0, a - b) for a, b in zip(pot_after, pot_before)]
    monotonic = sum(
        1 for prev, cur in zip(pot_after, pot_after[1:]) if cur + 1e-9 >= prev
    )

    return {
        "schema_player_count": player_count,
        "schema_seat_utilization": _safe_div(player_count, max_seats),
        "schema_action_count": action_count,
        "schema_street_count": street_count,
        "schema_call_share": _safe_div(counts.get("call", 0), meaningful),
        "schema_check_share": _safe_div(counts.get("check", 0), meaningful),
        "schema_fold_share": _safe_div(counts.get("fold", 0), meaningful),
        "schema_bet_share": _safe_div(counts.get("bet", 0), meaningful),
        "schema_raise_share": _safe_div(counts.get("raise", 0), meaningful),
        "schema_blind_share": _safe_div(
            counts.get("small_blind", 0) + counts.get("big_blind", 0) + counts.get("ante", 0),
            max(1.0, action_count),
        ),
        "schema_allin_share": _safe_div(counts.get("all_in", 0), max(1.0, action_count)),
        "schema_aggression_share": _safe_div(aggressive, max(1.0, action_count)),
        "schema_passive_share": _safe_div(passive, max(1.0, action_count)),
        "schema_preflop_share": _safe_div(preflop_n, max(1.0, action_count)),
        "schema_postflop_share": _safe_div(postflop_n, max(1.0, action_count)),
        "schema_action_entropy": _entropy(action_types),
        "schema_actor_entropy": _entropy(actor_seats),
        "schema_street_entropy": _entropy(street_names),
        "schema_unique_actor_share": _safe_div(len(set(actor_seats)), max(1.0, player_count)),
        "schema_actor_switch_rate": _safe_div(
            sum(1 for prev, cur in zip(actor_seats, actor_seats[1:]) if prev != cur),
            max(len(actor_seats) - 1, 1),
        ),
        "schema_actor_run_max_share": _max_run_share(actor_seats),
        "schema_action_run_max_share": _max_run_share(action_types),
        "schema_amount_mean_bb": _mean(amount_bb),
        "schema_amount_std_bb": _std(amount_bb),
        "schema_amount_q90_bb": _quantile(amount_bb, 0.9),
        "schema_amount_max_bb": max(amount_bb) if amount_bb else 0.0,
        "schema_nonzero_amount_share": _safe_div(nonzero_amount, max(1.0, action_count)),
        "schema_pot_before_mean_bb": _mean(pot_before),
        "schema_pot_after_mean_bb": _mean(pot_after),
        "schema_pot_delta_mean_bb": _mean(pot_delta),
        "schema_pot_growth_bb": (
            max(pot_after) - min(pot_before) if pot_after and pot_before else 0.0
        ),
        "schema_pot_monotonic_rate": _safe_div(monotonic, max(len(pot_after) - 1, 1)),
        "schema_raise_to_share": _safe_div(raise_to_present, max(1.0, action_count)),
        "schema_call_to_share": _safe_div(call_to_present, max(1.0, action_count)),
        "schema_starting_stack_mean_bb": _mean(stack_bb),
        "schema_starting_stack_std_bb": _std(stack_bb),
        "schema_starting_stack_iqr_bb": _quantile(stack_bb, 0.75) - _quantile(stack_bb, 0.25),
        "schema_hero_action_share": _safe_div(hero_actions, max(1.0, action_count)),
        "schema_button_action_share": _safe_div(button_actions, max(1.0, action_count)),
        "schema_hero_button_same": float(hero_seat > 0 and hero_seat == button_seat),
    }


def _aggregate_feature(prefix: str, values: list[float], out: dict[str, float]) -> None:
    out[f"{prefix}_mean"] = _mean(values)
    out[f"{prefix}_std"] = _std(values)
    out[f"{prefix}_min"] = min(values) if values else 0.0
    out[f"{prefix}_max"] = max(values) if values else 0.0
    out[f"{prefix}_q10"] = _quantile(values, 0.1)
    out[f"{prefix}_q25"] = _quantile(values, 0.25)
    out[f"{prefix}_q50"] = _quantile(values, 0.5)
    out[f"{prefix}_q75"] = _quantile(values, 0.75)
    out[f"{prefix}_q90"] = _quantile(values, 0.9)


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    if not chunk:
        return {"hand_count": 0.0}

    out: dict[str, float] = {"hand_count": float(len(chunk))}
    per_hand = [_hand_features(hand) for hand in chunk]
    feature_names = sorted(per_hand[0].keys())

    for name in feature_names:
        series = [float(features[name]) for features in per_hand]
        _aggregate_feature(name, series, out)

    action_signatures: list[tuple[str, ...]] = []
    actor_signatures: list[tuple[int, ...]] = []
    street_signatures: list[tuple[str, ...]] = []
    amount_bucket_signatures: list[tuple[str, ...]] = []
    street_action_counts: Counter[str] = Counter()
    street_aggressive_counts: Counter[str] = Counter()
    action_bigram_counts: Counter[tuple[str, str]] = Counter()

    high_aggressive = 0
    low_action_entropy = 0
    high_actor_entropy = 0
    long_action_hand = 0

    for hand, feats in zip(chunk, per_hand):
        actions = hand.get("actions") or []
        action_types = tuple(str((a or {}).get("action_type") or "").lower().strip() for a in actions)
        actor_seq = tuple(
            _safe_int((a or {}).get("actor_seat"), 0) for a in actions if _safe_int((a or {}).get("actor_seat"), 0) > 0
        )
        street_seq = tuple(str((a or {}).get("street") or "").lower().strip() for a in actions)
        amounts = [
            max(0.0, _safe_float((a or {}).get("normalized_amount_bb"), 0.0))
            for a in actions
        ]
        amount_buckets = tuple(_amount_bucket(value) for value in amounts)

        action_signatures.append(action_types)
        actor_signatures.append(actor_seq)
        street_signatures.append(street_seq)
        amount_bucket_signatures.append(amount_buckets)
        action_bigram_counts.update(zip(action_types[:-1], action_types[1:]))
        for action_type, street in zip(action_types, street_seq):
            if street:
                street_action_counts[street] += 1
                if action_type in {"bet", "raise"}:
                    street_aggressive_counts[street] += 1

        high_aggressive += int(feats["schema_aggression_share"] >= 0.35)
        low_action_entropy += int(feats["schema_action_entropy"] <= 0.35)
        high_actor_entropy += int(feats["schema_actor_entropy"] >= 0.75)
        long_action_hand += int(feats["schema_action_count"] >= 12.0)

    n = float(len(chunk))
    out["schema_action_signature_top_share"] = _safe_div(max(Counter(action_signatures).values()), n)
    out["schema_action_signature_unique_share"] = _safe_div(len(set(action_signatures)), n)
    out["schema_actor_signature_top_share"] = _safe_div(max(Counter(actor_signatures).values()), n)
    out["schema_actor_signature_unique_share"] = _safe_div(len(set(actor_signatures)), n)
    out["schema_street_signature_top_share"] = _safe_div(max(Counter(street_signatures).values()), n)
    out["schema_street_signature_unique_share"] = _safe_div(len(set(street_signatures)), n)
    out["schema_amount_bucket_signature_top_share"] = _safe_div(
        max(Counter(amount_bucket_signatures).values()), n
    )
    out["schema_amount_bucket_signature_unique_share"] = _safe_div(
        len(set(amount_bucket_signatures)), n
    )
    out["schema_high_aggression_hand_rate"] = _safe_div(high_aggressive, n)
    out["schema_low_action_entropy_hand_rate"] = _safe_div(low_action_entropy, n)
    out["schema_high_actor_entropy_hand_rate"] = _safe_div(high_actor_entropy, n)
    out["schema_long_action_hand_rate"] = _safe_div(long_action_hand, n)
    total_street_actions = max(1, sum(street_action_counts.values()))
    total_street_aggressive = max(1, sum(street_aggressive_counts.values()))
    for street in ("preflop", "flop", "turn", "river"):
        out[f"schema_{street}_action_share"] = _safe_div(
            street_action_counts.get(street, 0),
            total_street_actions,
        )
        out[f"schema_{street}_aggression_share"] = _safe_div(
            street_aggressive_counts.get(street, 0),
            total_street_aggressive,
        )
    _temporal_series_features(
        "schema_aggression_share_temporal",
        [float(features["schema_aggression_share"]) for features in per_hand],
        out,
    )
    _temporal_series_features(
        "schema_fold_share_temporal",
        [float(features["schema_fold_share"]) for features in per_hand],
        out,
    )
    _temporal_series_features(
        "schema_amount_std_temporal",
        [float(features["schema_amount_std_bb"]) for features in per_hand],
        out,
    )
    out["schema_action_bigram_entropy"] = _entropy(list(action_bigram_counts.elements()))
    out.update(
        _repeat_pattern_features(
            action_signatures,
            amount_bucket_signatures,
            street_signatures,
        )
    )
    return out
