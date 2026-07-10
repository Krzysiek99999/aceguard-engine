"""Action-order anomaly features for miner-visible Poker44 chunks.

These features intentionally use only fields exposed to miners. They are
order-aware inside each hand, but aggregate over hands at chunk level.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}
ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "all_in")
AMOUNT_EDGES = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = float(sum(counts.values()))
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p + 1e-12)
    return _div(ent, math.log(len(counts)))


def _amount_bucket(value: float) -> int:
    if value <= 0.0:
        return 0
    for idx, edge in enumerate(AMOUNT_EDGES[1:], start=1):
        if value <= edge:
            return idx
    return len(AMOUNT_EDGES)


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        arr = np.asarray([0.0], dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_q75": float(np.quantile(arr, 0.75)),
    }


def hand_action_anomaly_features(hand: dict[str, Any]) -> dict[str, float]:
    metadata = hand.get("metadata") or {}
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    hero_seat = _i(metadata.get("hero_seat"), 0)
    max_seats = max(1, _i(metadata.get("max_seats"), 6))

    action_types: list[str] = []
    streets: list[str] = []
    street_ranks: list[int] = []
    actors: list[int] = []
    amounts: list[float] = []
    amount_buckets: list[int] = []
    pot_after_values: list[float] = []

    folded: set[int] = set()
    acted_after_fold = 0
    same_actor_run = 0
    street_regression = 0
    street_jump = 0
    max_street_jump = 0
    pot_mismatch = 0
    pot_decrease = 0
    amount_negative = 0
    zero_aggressive_amount = 0
    raise_to_missing = 0
    call_to_missing = 0
    hero_actions = 0

    prev_actor: int | None = None
    prev_street_rank: int | None = None
    prev_pot_after: float | None = None

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type") or "").lower().strip()
        street = str(action.get("street") or "").lower().strip()
        actor = _i(action.get("actor_seat"), -1)
        amount = _f(action.get("normalized_amount_bb"), 0.0)
        pot_before = _f(action.get("pot_before"), 0.0)
        pot_after = _f(action.get("pot_after"), 0.0)

        action_types.append(action_type)
        streets.append(street)
        amounts.append(amount)
        amount_buckets.append(_amount_bucket(amount))
        if actor >= 0:
            actors.append(actor)
        if pot_after:
            pot_after_values.append(pot_after)

        if hero_seat and actor == hero_seat:
            hero_actions += 1
        if actor in folded:
            acted_after_fold += 1
        if action_type == "fold" and actor >= 0:
            folded.add(actor)
        if prev_actor is not None and actor == prev_actor:
            same_actor_run += 1
        if amount < 0:
            amount_negative += 1
        if action_type in {"bet", "raise", "all_in"} and amount <= 0.0:
            zero_aggressive_amount += 1
        if action_type == "raise" and action.get("raise_to") is None:
            raise_to_missing += 1
        if action_type == "call" and action.get("call_to") is None:
            call_to_missing += 1
        if prev_pot_after is not None:
            if abs(pot_before - prev_pot_after) > 0.01:
                pot_mismatch += 1
            if pot_before < prev_pot_after - 0.01:
                pot_decrease += 1

        rank = STREET_ORDER.get(street)
        if rank is not None:
            street_ranks.append(rank)
        if prev_street_rank is not None and rank is not None:
            diff = rank - prev_street_rank
            if diff < 0:
                street_regression += 1
            if diff > 1:
                street_jump += 1
                max_street_jump = max(max_street_jump, diff)

        prev_actor = actor
        prev_street_rank = rank if rank is not None else prev_street_rank
        prev_pot_after = pot_after

    n_actions = max(1, len(action_types))
    action_counts = Counter(action_types)
    aggressive = action_counts.get("bet", 0) + action_counts.get("raise", 0) + action_counts.get("all_in", 0)
    passive = action_counts.get("call", 0) + action_counts.get("check", 0)
    reached = {name: any(rank == value for rank in street_ranks) for name, value in STREET_ORDER.items()}
    last_street = max(street_ranks) if street_ranks else 0
    positive_amounts = [value for value in amounts if value > 0.0]

    out = {
        "aa_hand_count": 1.0,
        "aa_player_count": float(len(players)),
        "aa_seat_utilization": _div(len(players), max_seats),
        "aa_action_count": float(len(action_types)),
        "aa_unique_actor_share": _div(len(set(actors)), max(1, len(players))),
        "aa_action_entropy": _entropy(action_types),
        "aa_actor_entropy": _entropy(actors),
        "aa_street_entropy": _entropy(streets),
        "aa_amount_bucket_entropy": _entropy(amount_buckets),
        "aa_actor_switch_rate": _div(sum(1 for left, right in zip(actors, actors[1:]) if left != right), max(1, len(actors) - 1)),
        "aa_same_actor_run_rate": _div(same_actor_run, n_actions),
        "aa_acted_after_fold_rate": _div(acted_after_fold, n_actions),
        "aa_street_regression_rate": _div(street_regression, n_actions),
        "aa_street_jump_rate": _div(street_jump, n_actions),
        "aa_max_street_jump": float(max_street_jump),
        "aa_pot_mismatch_rate": _div(pot_mismatch, n_actions),
        "aa_pot_decrease_rate": _div(pot_decrease, n_actions),
        "aa_amount_negative_rate": _div(amount_negative, n_actions),
        "aa_zero_aggressive_amount_rate": _div(zero_aggressive_amount, n_actions),
        "aa_raise_to_missing_rate": _div(raise_to_missing, n_actions),
        "aa_call_to_missing_rate": _div(call_to_missing, n_actions),
        "aa_aggression_rate": _div(aggressive, n_actions),
        "aa_aggression_to_passive": _div(aggressive, max(1, passive)),
        "aa_hero_action_rate": _div(hero_actions, n_actions),
        "aa_last_street_rank": float(last_street),
        "aa_reached_flop": float(reached["flop"]),
        "aa_reached_turn": float(reached["turn"]),
        "aa_reached_river": float(reached["river"]),
        "aa_reached_showdown": float(reached["showdown"]),
        "aa_nonzero_amount_rate": _div(len(positive_amounts), n_actions),
        "aa_amount_log_mean": float(np.mean(np.log1p(np.asarray(positive_amounts or [0.0], dtype=float)))),
        "aa_pot_after_log_mean": float(np.mean(np.log1p(np.asarray(pot_after_values or [0.0], dtype=float)))),
    }
    for action_type in ACTION_TYPES:
        out[f"aa_action_{action_type}_rate"] = _div(action_counts.get(action_type, 0), n_actions)
    return out


def chunk_action_anomaly_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    rows = [hand_action_anomaly_features(hand) for hand in chunk if isinstance(hand, dict)]
    if not rows:
        rows = [hand_action_anomaly_features({})]
    keys = sorted({key for row in rows for key in row})
    out: dict[str, float] = {
        "aa_chunk_hands": float(len(rows)),
        "aa_chunk_total_actions": float(sum(row.get("aa_action_count", 0.0) for row in rows)),
    }
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        out.update(_stats(values, key))
    return out
