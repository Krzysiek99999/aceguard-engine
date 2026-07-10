"""Original identity-free behavioral features for Poker44 chunks.

The extractor uses only miner-visible gameplay fields. It intentionally ignores
hand IDs, player UIDs, cards, outcomes, labels, dates, and deployment metadata.
Features describe within-hand decisions and how those decisions vary over the
ordered sequence of hands in a chunk.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import numpy as np


ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "all_in", "other")
STREETS = ("preflop", "flop", "turn", "river", "showdown", "other")
STREET_ORDER = {name: index for index, name in enumerate(STREETS)}
AGGRESSIVE_ACTIONS = {"bet", "raise", "all_in"}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _action_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"allin", "all_in", "jam"}:
        return "all_in"
    return text if text in ACTION_TYPES else "other"


def _street_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in STREETS else "other"


def _ratio(numerator: float, denominator: float, *, cap: float = 50.0) -> float:
    if abs(denominator) <= 1e-9:
        return 0.0
    return float(np.clip(numerator / denominator, -cap, cap))


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else 0.0


def _distribution_entropy(items: Iterable[Any]) -> float:
    counts = np.asarray(list(Counter(items).values()), dtype=float)
    if counts.size <= 1 or float(np.sum(counts)) <= 0.0:
        return 0.0
    probs = counts / float(np.sum(counts))
    entropy = -float(np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0))))
    return entropy / max(math.log2(len(counts)), 1.0)


def _numeric_summary(prefix: str, values: Iterable[float], out: dict[str, float]) -> None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.asarray([0.0], dtype=float)
    out[f"{prefix}__mean"] = float(np.mean(arr))
    out[f"{prefix}__std"] = float(np.std(arr))
    out[f"{prefix}__q10"] = _quantile(arr, 0.10)
    out[f"{prefix}__q50"] = _quantile(arr, 0.50)
    out[f"{prefix}__q90"] = _quantile(arr, 0.90)
    out[f"{prefix}__iqr"] = _quantile(arr, 0.75) - _quantile(arr, 0.25)
    out[f"{prefix}__range"] = float(np.max(arr) - np.min(arr))
    rounded = np.round(arr, 3)
    out[f"{prefix}__unique_share"] = float(len(np.unique(rounded)) / max(arr.size, 1))

    if arr.size > 1:
        x = np.linspace(-0.5, 0.5, arr.size)
        out[f"{prefix}__slope"] = _ratio(float(np.dot(x, arr - np.mean(arr))), float(np.dot(x, x)))
        split = max(1, arr.size // 3)
        out[f"{prefix}__edge_delta"] = float(np.mean(arr[-split:]) - np.mean(arr[:split]))
        out[f"{prefix}__diff_abs_mean"] = float(np.mean(np.abs(np.diff(arr))))
        left = arr[:-1]
        right = arr[1:]
        if float(np.std(left)) > 1e-9 and float(np.std(right)) > 1e-9:
            out[f"{prefix}__lag1"] = float(np.corrcoef(left, right)[0, 1])
        else:
            out[f"{prefix}__lag1"] = 0.0
    else:
        out[f"{prefix}__slope"] = 0.0
        out[f"{prefix}__edge_delta"] = 0.0
        out[f"{prefix}__diff_abs_mean"] = 0.0
        out[f"{prefix}__lag1"] = 0.0


def _signature_summary(prefix: str, signatures: list[tuple[Any, ...]], out: dict[str, float]) -> None:
    n = len(signatures)
    counts = Counter(signatures)
    if n == 0:
        for suffix in ("unique_share", "repeat_share", "top_share", "entropy", "adjacent_same", "max_run_share"):
            out[f"{prefix}__{suffix}"] = 0.0
        return
    out[f"{prefix}__unique_share"] = len(counts) / n
    out[f"{prefix}__repeat_share"] = sum(count for count in counts.values() if count > 1) / n
    out[f"{prefix}__top_share"] = max(counts.values()) / n
    out[f"{prefix}__entropy"] = _distribution_entropy(signatures)
    out[f"{prefix}__adjacent_same"] = (
        sum(1 for left, right in zip(signatures, signatures[1:]) if left == right) / max(n - 1, 1)
    )
    longest = 1
    run = 1
    for left, right in zip(signatures, signatures[1:]):
        if left == right:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    out[f"{prefix}__max_run_share"] = longest / n


def _amount_bucket(value: float) -> int:
    if value <= 0.0:
        return 0
    return int(np.clip(math.floor(math.log2(1.0 + value)) + 1, 1, 12))


def _hand_metrics(hand: dict[str, Any]) -> tuple[dict[str, float], dict[str, tuple[Any, ...]]]:
    metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
    actions = [action for action in (hand.get("actions") or []) if isinstance(action, dict)]
    players = [player for player in (hand.get("players") or []) if isinstance(player, dict)]
    hero_seat = int(_number(metadata.get("hero_seat"), -1))
    button_seat = int(_number(metadata.get("button_seat"), -1))
    max_seats = max(2, int(_number(metadata.get("max_seats"), len(players) or 2)))
    bb = max(_number(metadata.get("bb"), 1.0), 1e-6)

    names = [_action_name(action.get("action_type")) for action in actions]
    streets = [_street_name(action.get("street")) for action in actions]
    actors = [int(_number(action.get("actor_seat"), -1)) for action in actions]
    hero_mask = np.asarray([actor == hero_seat and hero_seat >= 0 for actor in actors], dtype=bool)
    amount_bb = np.asarray([max(0.0, _number(action.get("normalized_amount_bb"))) for action in actions], dtype=float)
    pot_before_bb = np.asarray([max(0.0, _number(action.get("pot_before")) / bb) for action in actions], dtype=float)
    pot_after_bb = np.asarray([max(0.0, _number(action.get("pot_after")) / bb) for action in actions], dtype=float)
    pot_delta_bb = pot_after_bb - pot_before_bb
    amount_to_pot = np.asarray(
        [_ratio(amount, max(pot, 0.25), cap=20.0) for amount, pot in zip(amount_bb, pot_before_bb)],
        dtype=float,
    )
    nonzero = amount_bb > 1e-9
    aggressive = np.asarray([name in AGGRESSIVE_ACTIONS for name in names], dtype=bool)

    row: dict[str, float] = {
        "n_actions": float(len(actions)),
        "n_players": float(len(players)),
        "max_seats": float(max_seats),
        "hero_action_share": float(np.mean(hero_mask)) if hero_mask.size else 0.0,
        "nonzero_amount_share": float(np.mean(nonzero)) if nonzero.size else 0.0,
        "aggressive_share": float(np.mean(aggressive)) if aggressive.size else 0.0,
        "street_count": float(len(set(streets))),
        "action_entropy": _distribution_entropy(names),
        "street_entropy": _distribution_entropy(streets),
        "actor_entropy": _distribution_entropy(actors),
    }

    if hero_seat >= 0 and button_seat >= 0:
        row["hero_button_distance"] = float((hero_seat - button_seat) % max_seats) / max(max_seats - 1, 1)
    else:
        row["hero_button_distance"] = 0.0
    hero_positions = np.flatnonzero(hero_mask)
    row["hero_first_action_pos"] = float(hero_positions[0] / max(len(actions) - 1, 1)) if hero_positions.size else 0.0
    row["hero_last_action_pos"] = float(hero_positions[-1] / max(len(actions) - 1, 1)) if hero_positions.size else 0.0
    row["actor_switch_share"] = (
        sum(left != right for left, right in zip(actors, actors[1:])) / max(len(actors) - 1, 1)
    )
    row["street_progress_mean"] = float(np.mean([STREET_ORDER[street] for street in streets])) if streets else 0.0
    row["street_progress_max"] = float(max((STREET_ORDER[street] for street in streets), default=0))
    row["street_monotonic_share"] = (
        sum(STREET_ORDER[right] >= STREET_ORDER[left] for left, right in zip(streets, streets[1:]))
        / max(len(streets) - 1, 1)
    )

    for action_name in ACTION_TYPES:
        mask = np.asarray([name == action_name for name in names], dtype=bool)
        row[f"action_{action_name}_share"] = float(np.mean(mask)) if mask.size else 0.0
        row[f"hero_{action_name}_share"] = float(np.mean(mask[hero_mask])) if np.any(hero_mask) else 0.0
    for street in STREETS:
        mask = np.asarray([name == street for name in streets], dtype=bool)
        row[f"street_{street}_share"] = float(np.mean(mask)) if mask.size else 0.0
        row[f"hero_street_{street}_share"] = float(np.mean(mask[hero_mask])) if np.any(hero_mask) else 0.0

    for prefix, values in (
        ("amount", amount_bb),
        ("amount_nonzero", amount_bb[nonzero]),
        ("amount_hero", amount_bb[hero_mask] if hero_mask.size else np.asarray([])),
        ("amount_aggressive", amount_bb[aggressive] if aggressive.size else np.asarray([])),
        ("pot_before", pot_before_bb),
        ("pot_delta", pot_delta_bb),
        ("amount_to_pot", amount_to_pot),
    ):
        arr = np.asarray(values, dtype=float)
        row[f"{prefix}_mean"] = float(np.mean(arr)) if arr.size else 0.0
        row[f"{prefix}_std"] = float(np.std(arr)) if arr.size else 0.0
        row[f"{prefix}_q90"] = _quantile(arr, 0.90)
        row[f"{prefix}_max"] = float(np.max(arr)) if arr.size else 0.0

    stacks = []
    hero_stack = 0.0
    for player in players:
        stack_bb = max(0.0, _number(player.get("starting_stack")) / bb)
        stacks.append(stack_bb)
        if int(_number(player.get("seat"), -2)) == hero_seat:
            hero_stack = stack_bb
    stack_arr = np.asarray(stacks, dtype=float)
    row["stack_mean"] = float(np.mean(stack_arr)) if stack_arr.size else 0.0
    row["stack_std"] = float(np.std(stack_arr)) if stack_arr.size else 0.0
    row["stack_range"] = float(np.ptp(stack_arr)) if stack_arr.size else 0.0
    row["hero_stack"] = hero_stack
    row["hero_stack_ratio"] = _ratio(hero_stack, float(np.mean(stack_arr)) if stack_arr.size else 0.0, cap=10.0)

    response_types: list[tuple[str, str]] = []
    response_ratios: list[float] = []
    for index in hero_positions:
        if index <= 0:
            continue
        response_types.append((names[index - 1], names[index]))
        response_ratios.append(_ratio(amount_bb[index], max(amount_bb[index - 1], 0.25), cap=20.0))
    row["hero_response_count"] = float(len(response_types))
    row["hero_response_ratio_mean"] = float(np.mean(response_ratios)) if response_ratios else 0.0
    row["hero_response_ratio_std"] = float(np.std(response_ratios)) if response_ratios else 0.0
    row["hero_response_entropy"] = _distribution_entropy(response_types)

    signatures = {
        "action": tuple(names),
        "street_action": tuple(zip(streets, names)),
        "hero_action": tuple((actor == hero_seat, name) for actor, name in zip(actors, names)),
        "amount_action": tuple((name, _amount_bucket(amount)) for name, amount in zip(names, amount_bb)),
        "response": tuple(response_types),
    }
    return row, signatures


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    hands = [hand for hand in (chunk or []) if isinstance(hand, dict)]
    hand_rows: list[dict[str, float]] = []
    signature_rows: dict[str, list[tuple[Any, ...]]] = {
        "action": [],
        "street_action": [],
        "hero_action": [],
        "amount_action": [],
        "response": [],
    }
    global_actions: list[str] = []
    global_streets: list[str] = []
    global_hero_flags: list[bool] = []
    global_amounts: list[float] = []
    transition_counts: Counter[tuple[str, str]] = Counter()

    for hand in hands:
        metrics, signatures = _hand_metrics(hand)
        hand_rows.append(metrics)
        for name, signature in signatures.items():
            signature_rows[name].append(signature)

        metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
        hero_seat = int(_number(metadata.get("hero_seat"), -1))
        actions = [action for action in (hand.get("actions") or []) if isinstance(action, dict)]
        names = [_action_name(action.get("action_type")) for action in actions]
        global_actions.extend(names)
        global_streets.extend(_street_name(action.get("street")) for action in actions)
        global_hero_flags.extend(int(_number(action.get("actor_seat"), -2)) == hero_seat for action in actions)
        global_amounts.extend(max(0.0, _number(action.get("normalized_amount_bb"))) for action in actions)
        transition_counts.update(zip(names, names[1:]))

    out: dict[str, float] = {
        "chunk__hands": float(len(hands)),
        "chunk__actions": float(len(global_actions)),
        "chunk__actions_per_hand": _ratio(float(len(global_actions)), float(len(hands)), cap=100.0),
        "chunk__action_entropy": _distribution_entropy(global_actions),
        "chunk__street_entropy": _distribution_entropy(global_streets),
        "chunk__hero_action_share": float(np.mean(global_hero_flags)) if global_hero_flags else 0.0,
    }

    keys = sorted({key for row in hand_rows for key in row})
    for key in keys:
        _numeric_summary(f"hand__{key}", (row.get(key, 0.0) for row in hand_rows), out)
    _numeric_summary("global__amount", global_amounts, out)

    for name, signatures in signature_rows.items():
        _signature_summary(f"motif__{name}", signatures, out)

    action_total = max(len(global_actions), 1)
    for action in ACTION_TYPES:
        out[f"global__action_{action}_share"] = global_actions.count(action) / action_total
        hero_values = [name for name, is_hero in zip(global_actions, global_hero_flags) if is_hero]
        out[f"global__hero_{action}_share"] = hero_values.count(action) / max(len(hero_values), 1)
    for street in STREETS:
        out[f"global__street_{street}_share"] = global_streets.count(street) / max(len(global_streets), 1)
    transition_total = max(sum(transition_counts.values()), 1)
    for left in ACTION_TYPES:
        for right in ACTION_TYPES:
            out[f"transition__{left}__{right}"] = transition_counts[(left, right)] / transition_total

    return {
        key: float(value) if math.isfinite(float(value)) else 0.0
        for key, value in out.items()
    }
