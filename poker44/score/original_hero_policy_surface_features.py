"""Fixed-width, identity-free HERO policy features for Poker44 chunks.

Only the HERO role, action order, street, action type, normalized amount, pot
before the action, and big-blind size are observed.  Every feature that can
change with monetary fields lives below the ``amount__`` prefix.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable


STREETS = ("preflop", "flop", "turn", "river", "other")
ACTION_CLASSES = ("fold", "check", "call", "aggressive", "other")
FACING_CLASSES = ("open", "check", "call", "aggressive", "fold", "forced", "other")
NGRAM_BINS = 32

_AGGRESSIVE = {"bet", "raise", "all_in", "allin", "jam", "shove"}
_FORCED = {"ante", "small_blind", "big_blind", "straddle", "bring_in"}
_AMOUNT_EDGES = (0.01, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
_POT_FRACTION_EDGES = (0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _integer(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _normalized_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _street(value: Any) -> str:
    name = _normalized_name(value)
    return name if name in STREETS[:-1] else "other"


def _action_class(value: Any) -> str:
    name = _normalized_name(value)
    if name in _AGGRESSIVE:
        return "aggressive"
    if name in ACTION_CLASSES[:-1]:
        return name
    return "other"


def _facing_class(value: Any) -> str:
    name = _normalized_name(value)
    if name in _AGGRESSIVE:
        return "aggressive"
    if name in _FORCED:
        return "forced"
    if name in {"check", "call", "fold"}:
        return name
    return "other"


def _bucket(value: float, edges: Iterable[float]) -> int:
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(tuple(edges))


def _hash_bin(namespace: str, token: str) -> int:
    digest = hashlib.blake2b(
        f"{namespace}|{token}".encode("utf-8", errors="ignore"),
        digest_size=8,
        person=b"hero-policy",
    ).digest()
    return int.from_bytes(digest, "big") % NGRAM_BINS


def _normalized_entropy(counter: Counter[Any]) -> float:
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy / math.log2(len(counter))


def _concentration(counter: Counter[Any]) -> float:
    total = sum(counter.values())
    return max(counter.values()) / total if total else 0.0


def _ngrams(tokens: list[str], length: int) -> list[str]:
    return [">".join(tokens[index : index + length]) for index in range(len(tokens) - length + 1)]


def _empty_features() -> dict[str, float]:
    output: dict[str, float] = {}
    for street in STREETS:
        for facing in FACING_CLASSES:
            prefix = f"policy__{street}__facing_{facing}"
            output[f"{prefix}__coverage_share"] = 0.0
            output[f"{prefix}__response_entropy"] = 0.0
            output[f"{prefix}__response_concentration"] = 0.0
            for action in ACTION_CLASSES:
                output[f"{prefix}__response_{action}_share"] = 0.0

    for family in ("token", "amount"):
        for length in (1, 2, 3):
            prefix = f"{family}__hero_ngram{length}"
            output[f"{prefix}__entropy"] = 0.0
            output[f"{prefix}__concentration"] = 0.0
            for index in range(NGRAM_BINS):
                output[f"{prefix}__h{index:02d}"] = 0.0

    for family in ("signature", "amount__signature"):
        output[f"{family}__repeat_pair_share"] = 0.0
        output[f"{family}__top_share"] = 0.0
        output[f"{family}__entropy"] = 0.0
        output[f"{family}__adjacent_repeat_share"] = 0.0

    for previous in ACTION_CLASSES:
        for current in ACTION_CLASSES:
            output[f"sequence__transition_{previous}_to_{current}_share"] = 0.0
    for action in ACTION_CLASSES:
        output[f"sequence__{action}__first_half_share"] = 0.0
        output[f"sequence__{action}__second_half_share"] = 0.0
        output[f"sequence__{action}__drift"] = 0.0
    output["sequence__transition_same_share"] = 0.0
    output["sequence__lag2_same_share"] = 0.0
    output["sequence__signature_streak_share"] = 0.0
    output["sequence__aggressive_lag1_correlation"] = 0.0
    output["sequence__aggressive_linear_drift"] = 0.0

    output["amount__sequence__adjacent_bucket_same_share"] = 0.0
    output["amount__sequence__lag2_bucket_same_share"] = 0.0
    output["amount__sequence__pot_fraction_linear_drift"] = 0.0
    return output


def _pair_repeat_share(counter: Counter[Any]) -> float:
    total = sum(counter.values())
    possible = total * (total - 1) // 2
    repeated = sum(count * (count - 1) // 2 for count in counter.values())
    return repeated / possible if possible else 0.0


def _adjacent_same(values: list[Any], lag: int = 1) -> float:
    if len(values) <= lag:
        return 0.0
    return sum(left == right for left, right in zip(values, values[lag:])) / (len(values) - lag)


def _max_run_share(values: list[Any]) -> float:
    if not values:
        return 0.0
    longest = current = 1
    for previous, value in zip(values, values[1:]):
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
    return longest / len(values)


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator > 0.0 else 0.0


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = (len(values) - 1) / 2.0
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    if denominator <= 0.0:
        return 0.0
    mean = sum(values) / len(values)
    slope = sum((index - center) * (value - mean) for index, value in enumerate(values)) / denominator
    return slope * max(len(values) - 1, 1)


def _fill_distribution(output: dict[str, float], prefix: str, values: list[str]) -> None:
    counter = Counter(values)
    total = max(len(values), 1)
    output[f"{prefix}__entropy"] = _normalized_entropy(counter)
    output[f"{prefix}__concentration"] = _concentration(counter)
    bins: Counter[int] = Counter(_hash_bin(prefix, value) for value in values)
    for index in range(NGRAM_BINS):
        output[f"{prefix}__h{index:02d}"] = bins[index] / total


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Extract a deterministic, fixed-width HERO policy surface."""
    output = _empty_features()
    context_responses: dict[tuple[str, str], list[str]] = {
        (street, facing): [] for street in STREETS for facing in FACING_CLASSES
    }
    action_grams: dict[int, list[str]] = {length: [] for length in (1, 2, 3)}
    amount_grams: dict[int, list[str]] = {length: [] for length in (1, 2, 3)}
    action_signatures: list[str] = []
    amount_signatures: list[str] = []
    hand_action_shares: list[dict[str, float]] = []
    first_actions: list[str] = []
    last_actions: list[str] = []
    hand_amount_buckets: list[str] = []
    hand_pot_fraction_means: list[float] = []

    for hand in chunk or []:
        if not isinstance(hand, dict):
            continue
        metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
        hero_seat = _integer(metadata.get("hero_seat"), -1)
        bb = max(_finite_float(metadata.get("bb"), 1.0), 1e-9)
        prior_opponent: dict[str, str] = {}
        action_tokens: list[str] = []
        amount_tokens: list[str] = []
        hero_classes: list[str] = []
        amount_buckets: list[int] = []
        pot_fractions: list[float] = []

        actions = hand.get("actions") if isinstance(hand.get("actions"), list) else []
        for raw in actions:
            if not isinstance(raw, dict):
                continue
            street = _street(raw.get("street"))
            actor = _integer(raw.get("actor_seat"), -2)
            is_hero = hero_seat >= 0 and actor == hero_seat
            if is_hero:
                action = _action_class(raw.get("action_type"))
                facing = prior_opponent.get(street, "open")
                context_responses[(street, facing)].append(action)
                action_token = f"{street}:{facing}:{action}"
                amount_bb = max(0.0, _finite_float(raw.get("normalized_amount_bb")))
                pot_bb = max(0.0, _finite_float(raw.get("pot_before")) / bb)
                fraction = amount_bb / max(pot_bb, 0.25)
                amount_bucket = _bucket(amount_bb, _AMOUNT_EDGES)
                fraction_bucket = _bucket(fraction, _POT_FRACTION_EDGES)
                amount_token = f"{action_token}:a{amount_bucket}:p{fraction_bucket}"
                action_tokens.append(action_token)
                amount_tokens.append(amount_token)
                hero_classes.append(action)
                amount_buckets.append(amount_bucket)
                pot_fractions.append(min(fraction, 8.0))
            else:
                prior_opponent[street] = _facing_class(raw.get("action_type"))

        if not action_tokens:
            continue
        for length in (1, 2, 3):
            action_grams[length].extend(_ngrams(action_tokens, length))
            amount_grams[length].extend(_ngrams(amount_tokens, length))
        action_signatures.append(">".join(action_tokens))
        amount_signatures.append(">".join(amount_tokens))
        counts = Counter(hero_classes)
        hand_action_shares.append(
            {action: counts[action] / len(hero_classes) for action in ACTION_CLASSES}
        )
        first_actions.append(hero_classes[0])
        last_actions.append(hero_classes[-1])
        hand_amount_buckets.append("-".join(str(value) for value in amount_buckets))
        hand_pot_fraction_means.append(sum(pot_fractions) / len(pot_fractions))

    decision_total = sum(len(values) for values in context_responses.values())
    for (street, facing), responses in context_responses.items():
        prefix = f"policy__{street}__facing_{facing}"
        counts = Counter(responses)
        output[f"{prefix}__coverage_share"] = len(responses) / max(decision_total, 1)
        output[f"{prefix}__response_entropy"] = _normalized_entropy(counts)
        output[f"{prefix}__response_concentration"] = _concentration(counts)
        for action in ACTION_CLASSES:
            output[f"{prefix}__response_{action}_share"] = counts[action] / max(len(responses), 1)

    for length in (1, 2, 3):
        _fill_distribution(output, f"token__hero_ngram{length}", action_grams[length])
        _fill_distribution(output, f"amount__hero_ngram{length}", amount_grams[length])

    for prefix, signatures in (
        ("signature", action_signatures),
        ("amount__signature", amount_signatures),
    ):
        counts = Counter(signatures)
        output[f"{prefix}__repeat_pair_share"] = _pair_repeat_share(counts)
        output[f"{prefix}__top_share"] = _concentration(counts)
        output[f"{prefix}__entropy"] = _normalized_entropy(counts)
        output[f"{prefix}__adjacent_repeat_share"] = _adjacent_same(signatures)

    transitions = list(zip(last_actions, first_actions[1:]))
    transition_total = max(len(transitions), 1)
    transition_counts = Counter(transitions)
    for previous in ACTION_CLASSES:
        for current in ACTION_CLASSES:
            output[f"sequence__transition_{previous}_to_{current}_share"] = (
                transition_counts[(previous, current)] / transition_total
            )
    output["sequence__transition_same_share"] = (
        sum(previous == current for previous, current in transitions) / transition_total
    )
    output["sequence__lag2_same_share"] = _adjacent_same(action_signatures, lag=2)
    output["sequence__signature_streak_share"] = _max_run_share(action_signatures)

    midpoint = (len(hand_action_shares) + 1) // 2
    for action in ACTION_CLASSES:
        values = [row[action] for row in hand_action_shares]
        first = sum(values[:midpoint]) / max(len(values[:midpoint]), 1)
        second = sum(values[midpoint:]) / max(len(values[midpoint:]), 1)
        output[f"sequence__{action}__first_half_share"] = first
        output[f"sequence__{action}__second_half_share"] = second
        output[f"sequence__{action}__drift"] = second - first if len(values) > 1 else 0.0
    aggressive_values = [row["aggressive"] for row in hand_action_shares]
    output["sequence__aggressive_lag1_correlation"] = _correlation(
        aggressive_values[:-1], aggressive_values[1:]
    )
    output["sequence__aggressive_linear_drift"] = _linear_slope(aggressive_values)

    output["amount__sequence__adjacent_bucket_same_share"] = _adjacent_same(hand_amount_buckets)
    output["amount__sequence__lag2_bucket_same_share"] = _adjacent_same(hand_amount_buckets, lag=2)
    output["amount__sequence__pot_fraction_linear_drift"] = _linear_slope(hand_pot_fraction_means)
    return {
        key: float(value) if math.isfinite(float(value)) else 0.0
        for key, value in output.items()
    }


__all__ = ["chunk_features"]
