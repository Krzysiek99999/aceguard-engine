"""Independent, identity-free Poker44 schema with role-aware extensions.

The base extractor expresses forty per-hand behavioural measurements through
seven distribution summaries, then adds thirteen chunk-level consistency
measurements.  It is deliberately implemented without importing any public
miner package.  The optional role-aware extractor adds seat-normalized sequence
summaries that remain stable when player seat numbers rotate between hands.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np


SUMMARY_NAMES = ("mean", "std", "min", "max", "q10", "q50", "q90")
HAND_MEASUREMENTS = (
    "players",
    "seat_fill",
    "actions",
    "streets",
    "call_fraction",
    "check_fraction",
    "fold_fraction",
    "bet_fraction",
    "raise_fraction",
    "blind_fraction",
    "allin_fraction",
    "aggressive_fraction",
    "passive_fraction",
    "preflop_fraction",
    "postflop_fraction",
    "action_diversity",
    "actor_diversity",
    "street_diversity",
    "actor_coverage",
    "actor_switch_fraction",
    "actor_longest_run",
    "action_longest_run",
    "amount_mean",
    "amount_std",
    "amount_q90",
    "amount_max",
    "amount_nonzero_fraction",
    "pot_before_mean",
    "pot_after_mean",
    "pot_delta_mean",
    "pot_span",
    "pot_nondecreasing_fraction",
    "raise_target_fraction",
    "call_target_fraction",
    "stack_mean",
    "stack_std",
    "stack_iqr",
    "hero_action_fraction",
    "button_action_fraction",
    "hero_is_button",
)


def _real(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return result if math.isfinite(result) else float(fallback)


def _whole(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _fraction(part: float, total: float) -> float:
    return float(part / total) if total else 0.0


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalized_information(values: Sequence[Any]) -> float:
    frequencies = np.asarray(list(Counter(values).values()), dtype=float)
    if frequencies.size < 2:
        return 0.0
    probabilities = frequencies / frequencies.sum()
    return float(-(probabilities * np.log2(probabilities)).sum() / math.log2(frequencies.size))


def _quantile(values: Sequence[float], level: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), level, method="linear"))


def _population_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float))) if values else 0.0


def _longest_constant_fraction(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    longest = run = 1
    for left, right in zip(values, values[1:]):
        run = run + 1 if left == right else 1
        longest = max(longest, run)
    return float(longest / len(values))


def _size_band(value: float) -> int:
    if value <= 0.0:
        return 0
    if value <= 0.5:
        return 1
    if value <= 1.0:
        return 2
    if value <= 2.0:
        return 3
    if value <= 5.0:
        return 4
    return 5


def _mode_and_variety(signatures: Sequence[tuple[Any, ...]]) -> tuple[float, float]:
    if not signatures:
        return 0.0, 0.0
    counts = Counter(signatures)
    size = float(len(signatures))
    return float(max(counts.values()) / size), float(len(counts) / size)


def _hand_view(hand: dict[str, Any]) -> tuple[dict[str, float], dict[str, tuple[Any, ...]]]:
    metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
    raw_players = hand.get("players") if isinstance(hand.get("players"), list) else []
    raw_streets = hand.get("streets") if isinstance(hand.get("streets"), list) else []
    actions = [row for row in (hand.get("actions") or []) if isinstance(row, dict)]
    players = [row for row in raw_players if isinstance(row, dict)]

    hero = _whole(metadata.get("hero_seat"), 0)
    button = _whole(metadata.get("button_seat"), 0)
    table_size = max(1, _whole(metadata.get("max_seats"), 6))
    action_names = [_text(row.get("action_type")) for row in actions]
    action_streets = [_text(row.get("street")) for row in actions]
    actors = [_whole(row.get("actor_seat"), 0) for row in actions]
    known_actors = [seat for seat in actors if seat > 0]
    amounts = [max(0.0, _real(row.get("normalized_amount_bb"))) for row in actions]
    pot_before = [max(0.0, _real(row.get("pot_before")) / 0.02) for row in actions]
    pot_after = [max(0.0, _real(row.get("pot_after")) / 0.02) for row in actions]
    stacks = [max(0.0, _real(row.get("starting_stack")) / 0.02) for row in players]

    counts = Counter(action_names)
    decision_count = max(
        1,
        sum(counts.get(name, 0) for name in ("call", "check", "bet", "raise", "fold")),
    )
    action_count = len(actions)
    action_denominator = max(1, action_count)
    player_count = len(raw_players)
    player_denominator = max(1, player_count)
    aggressive = counts.get("bet", 0) + counts.get("raise", 0)
    passive = counts.get("call", 0) + counts.get("check", 0)
    preflop = sum(name == "preflop" for name in action_streets)
    postflop = sum(name not in {"", "preflop"} for name in action_streets)
    pot_increase = [max(0.0, after - before) for before, after in zip(pot_before, pot_after)]
    nondecreasing = sum(
        current + 1e-9 >= previous for previous, current in zip(pot_after, pot_after[1:])
    )

    metrics = {
        "players": float(player_count),
        "seat_fill": _fraction(player_count, table_size),
        "actions": float(action_count),
        "streets": float(len(raw_streets)),
        "call_fraction": _fraction(counts.get("call", 0), decision_count),
        "check_fraction": _fraction(counts.get("check", 0), decision_count),
        "fold_fraction": _fraction(counts.get("fold", 0), decision_count),
        "bet_fraction": _fraction(counts.get("bet", 0), decision_count),
        "raise_fraction": _fraction(counts.get("raise", 0), decision_count),
        "blind_fraction": _fraction(
            counts.get("small_blind", 0) + counts.get("big_blind", 0) + counts.get("ante", 0),
            action_denominator,
        ),
        "allin_fraction": _fraction(counts.get("all_in", 0), action_denominator),
        "aggressive_fraction": _fraction(aggressive, action_denominator),
        "passive_fraction": _fraction(passive, action_denominator),
        "preflop_fraction": _fraction(preflop, action_denominator),
        "postflop_fraction": _fraction(postflop, action_denominator),
        "action_diversity": _normalized_information(action_names),
        "actor_diversity": _normalized_information(known_actors),
        "street_diversity": _normalized_information(action_streets),
        "actor_coverage": _fraction(len(set(known_actors)), player_denominator),
        "actor_switch_fraction": _fraction(
            sum(left != right for left, right in zip(known_actors, known_actors[1:])),
            max(1, len(known_actors) - 1),
        ),
        "actor_longest_run": _longest_constant_fraction(known_actors),
        "action_longest_run": _longest_constant_fraction(action_names),
        "amount_mean": float(np.mean(amounts)) if amounts else 0.0,
        "amount_std": _population_std(amounts),
        "amount_q90": _quantile(amounts, 0.90),
        "amount_max": max(amounts, default=0.0),
        "amount_nonzero_fraction": _fraction(sum(value > 0.0 for value in amounts), action_denominator),
        "pot_before_mean": float(np.mean(pot_before)) if pot_before else 0.0,
        "pot_after_mean": float(np.mean(pot_after)) if pot_after else 0.0,
        "pot_delta_mean": float(np.mean(pot_increase)) if pot_increase else 0.0,
        "pot_span": max(pot_after) - min(pot_before) if pot_after and pot_before else 0.0,
        "pot_nondecreasing_fraction": _fraction(nondecreasing, max(1, len(pot_after) - 1)),
        "raise_target_fraction": _fraction(sum(row.get("raise_to") is not None for row in actions), action_denominator),
        "call_target_fraction": _fraction(sum(row.get("call_to") is not None for row in actions), action_denominator),
        "stack_mean": float(np.mean(stacks)) if stacks else 0.0,
        "stack_std": _population_std(stacks),
        "stack_iqr": _quantile(stacks, 0.75) - _quantile(stacks, 0.25),
        "hero_action_fraction": _fraction(sum(seat == hero and hero > 0 for seat in known_actors), action_denominator),
        "button_action_fraction": _fraction(sum(seat == button and button > 0 for seat in known_actors), action_denominator),
        "hero_is_button": float(hero > 0 and hero == button),
    }

    role_sequence: list[str] = []
    for seat in actors:
        if seat <= 0:
            continue
        if hero > 0 and seat == hero:
            role_sequence.append("hero")
        elif button > 0 and seat == button:
            role_sequence.append("button")
        elif button > 0:
            role_sequence.append(f"offset_{(seat - button) % table_size}")
        else:
            role_sequence.append("other")

    signatures = {
        "action": tuple(action_names),
        "actor": tuple(known_actors),
        "street": tuple(action_streets),
        "size": tuple(_size_band(value) for value in amounts),
        "role": tuple(role_sequence),
        "role_action": tuple(zip(role_sequence, [name for name, actor in zip(action_names, actors) if actor > 0])),
        "street_action": tuple(zip(action_streets, action_names)),
    }
    return metrics, signatures


def _summary(values: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return (0.0,) * len(SUMMARY_NAMES)
    return (
        float(np.mean(array)),
        float(np.std(array)),
        float(np.min(array)),
        float(np.max(array)),
        float(np.quantile(array, 0.10, method="linear")),
        float(np.quantile(array, 0.50, method="linear")),
        float(np.quantile(array, 0.90, method="linear")),
    )


def _base_chunk_features(chunk: Iterable[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, float]], dict[str, list[tuple[Any, ...]]]]:
    hands = [row for row in (chunk or []) if isinstance(row, dict)]
    hand_rows: list[dict[str, float]] = []
    signature_rows = {name: [] for name in ("action", "actor", "street", "size", "role", "role_action", "street_action")}
    for hand in hands:
        measurements, signatures = _hand_view(hand)
        hand_rows.append(measurements)
        for name, value in signatures.items():
            signature_rows[name].append(value)

    out: dict[str, float] = {}
    for measurement in HAND_MEASUREMENTS:
        values = [row[measurement] for row in hand_rows]
        for summary_name, value in zip(SUMMARY_NAMES, _summary(values)):
            out[f"hand__{measurement}__{summary_name}"] = value

    count = float(len(hands))
    out["chunk__hands"] = count
    for signature_name in ("action", "actor", "street", "size"):
        mode, variety = _mode_and_variety(signature_rows[signature_name])
        out[f"chunk__{signature_name}_pattern__mode"] = mode
        out[f"chunk__{signature_name}_pattern__variety"] = variety
    out["chunk__high_aggression_rate"] = _fraction(
        sum(row["aggressive_fraction"] >= 0.35 for row in hand_rows), count
    )
    out["chunk__low_action_diversity_rate"] = _fraction(
        sum(row["action_diversity"] <= 0.35 for row in hand_rows), count
    )
    out["chunk__high_actor_diversity_rate"] = _fraction(
        sum(row["actor_diversity"] >= 0.75 for row in hand_rows), count
    )
    out["chunk__long_hand_rate"] = _fraction(sum(row["actions"] >= 12.0 for row in hand_rows), count)
    return out, hand_rows, signature_rows


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Return the fixed 293-value independently implemented base schema."""
    out, _rows, _signatures = _base_chunk_features(chunk)
    return out


def role_aware_chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Extend the base schema with seat-rotation-resistant sequence summaries."""
    out, _rows, signatures = _base_chunk_features(chunk)
    for name in ("role", "role_action", "street_action"):
        mode, variety = _mode_and_variety(signatures[name])
        out[f"role_context__{name}__mode"] = mode
        out[f"role_context__{name}__variety"] = variety
        out[f"role_context__{name}__entropy"] = _normalized_information(signatures[name])
    return out


assert len(chunk_features([])) == 293
