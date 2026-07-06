"""Deterministic sequence features for Poker44 chunk scoring.

These features use only miner-visible hand/action fields. They are designed to
capture temporal/action-order signal that aggregate schema features lose:
action n-grams, role/street patterns, amount-bucket rhythms, and simple motif
concentration statistics.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable


ACTION_NAMES = ("check", "call", "bet", "raise", "fold")
STREET_NAMES = ("preflop", "flop", "turn", "river")
AMOUNT_BUCKETS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
N_HASH_BINS = 64


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _action(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in ACTION_NAMES:
        return raw
    for name in ACTION_NAMES:
        if name in raw:
            return name
    return "check"


def _street(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in STREET_NAMES else ""


def _amount_bucket(value: Any) -> str:
    amount = max(0.0, _safe_float(value, 0.0))
    if amount <= 0.0:
        return "0"
    nearest = min(AMOUNT_BUCKETS, key=lambda b: abs(b - amount))
    if amount >= 48.0:
        return "48p"
    return str(nearest).replace(".", "_")


def _entropy(values: Iterable[Any]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts.values():
        p = count / total
        out -= p * math.log2(max(p, 1e-12))
    return out


def _norm_entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    unique = len(set(values))
    if unique <= 1:
        return 0.0
    return _entropy(values) / max(math.log2(unique), 1e-9)


def _hash_bin(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % N_HASH_BINS


def _ngram_tokens(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return ["|".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _add_hash_features(out: dict[str, float], prefix: str, tokens: list[str]) -> None:
    bins = [0.0] * N_HASH_BINS
    total = max(len(tokens), 1)
    for token in tokens:
        bins[_hash_bin(token)] += 1.0
    for idx, value in enumerate(bins):
        out[f"{prefix}_h{idx:02d}"] = value / total


def _hand_tokens(hand: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    metadata = hand.get("metadata") or {}
    hero_seat = _safe_int(metadata.get("hero_seat"), 0)
    full: list[str] = []
    hero: list[str] = []
    coarse: list[str] = []
    for action in hand.get("actions") or []:
        if not isinstance(action, dict):
            continue
        role = "H" if hero_seat and _safe_int(action.get("actor_seat"), -1) == hero_seat else "O"
        act = _action(action.get("action_type"))
        street = _street(action.get("street"))
        bucket = _amount_bucket(action.get("normalized_amount_bb"))
        full_token = f"{street}:{role}:{act}:{bucket}"
        coarse_token = f"{street}:{role}:{act}"
        full.append(full_token)
        coarse.append(coarse_token)
        if role == "H":
            hero.append(full_token)
    return full, hero, coarse


def _max_share(values: list[Any]) -> float:
    if not values:
        return 0.0
    return max(Counter(values).values()) / len(values)


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    hands = [hand for hand in chunk if isinstance(hand, dict)]
    n_hands = len(hands)
    out["seq_hand_count"] = float(n_hands)
    if not hands:
        for n in (1, 2, 3, 4):
            _add_hash_features(out, f"seq_full_ng{n}", [])
        for n in (1, 2, 3):
            _add_hash_features(out, f"seq_hero_ng{n}", [])
        return out

    full_stream: list[str] = []
    hero_stream: list[str] = []
    coarse_motifs: list[tuple[str, ...]] = []
    action_names: list[str] = []
    street_names: list[str] = []
    amount_names: list[str] = []
    hero_action_count = 0

    per_hand_action_counts: list[int] = []
    per_hand_hero_counts: list[int] = []
    per_hand_aggressive_counts: list[int] = []

    for hand in hands:
        full, hero, coarse = _hand_tokens(hand)
        full_stream.extend(full)
        full_stream.append("<EOH>")
        hero_stream.extend(hero)
        hero_stream.append("<EOH>")
        coarse_motifs.append(tuple(coarse))
        per_hand_action_counts.append(len(full))
        per_hand_hero_counts.append(len(hero))
        aggressive = sum(1 for token in coarse if token.endswith(":bet") or token.endswith(":raise"))
        per_hand_aggressive_counts.append(aggressive)
        hero_action_count += len(hero)
        for token in full:
            street, _role, action, amount = token.split(":", 3)
            action_names.append(action)
            street_names.append(street)
            amount_names.append(amount)

    full_no_eoh = [token for token in full_stream if token != "<EOH>"]
    hero_no_eoh = [token for token in hero_stream if token != "<EOH>"]
    n_actions = len(full_no_eoh)

    out["seq_action_count"] = float(n_actions)
    out["seq_actions_per_hand_mean"] = sum(per_hand_action_counts) / max(n_hands, 1)
    out["seq_actions_per_hand_max"] = float(max(per_hand_action_counts or [0]))
    out["seq_hero_action_share"] = hero_action_count / max(n_actions, 1)
    out["seq_aggressive_per_hand_mean"] = sum(per_hand_aggressive_counts) / max(n_hands, 1)
    out["seq_hero_per_hand_mean"] = sum(per_hand_hero_counts) / max(n_hands, 1)
    out["seq_action_entropy"] = _norm_entropy(action_names)
    out["seq_street_entropy"] = _norm_entropy(street_names)
    out["seq_amount_entropy"] = _norm_entropy(amount_names)
    out["seq_action_max_share"] = _max_share(action_names)
    out["seq_amount_max_share"] = _max_share(amount_names)
    out["seq_motif_max_share"] = _max_share(coarse_motifs)
    out["seq_motif_entropy_lo"] = 1.0 - _norm_entropy(list(coarse_motifs))

    for name in ACTION_NAMES:
        out[f"seq_action_share_{name}"] = action_names.count(name) / max(n_actions, 1)
    for name in STREET_NAMES:
        out[f"seq_street_share_{name}"] = street_names.count(name) / max(n_actions, 1)

    for n in (1, 2, 3, 4):
        _add_hash_features(out, f"seq_full_ng{n}", _ngram_tokens(full_stream, n))
    for n in (1, 2, 3):
        _add_hash_features(out, f"seq_hero_ng{n}", _ngram_tokens(hero_stream, n))

    return out
