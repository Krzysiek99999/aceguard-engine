"""Order-invariant repetition and conditional-policy features for hand bags."""
from __future__ import annotations

import math
import zlib
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from poker44.score.model_view_hand_hash import canonical_hand_bag


CHANNELS = ("action", "street_action", "hero_action", "amount_action")
NGRAM_SIZES = (1, 2, 3)
PAIRWISE_HAND_CAP = 32


def _number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _amount_bucket(value: Any) -> str:
    amount = max(0.0, _number(value))
    if amount <= 0.0:
        return "z"
    return str(int(np.clip(math.floor(math.log2(1.0 + amount)), 0, 10)))


def _entropy(items: Iterable[Any]) -> float:
    counts = np.asarray(list(Counter(items).values()), dtype=float)
    if counts.size <= 1:
        return 0.0
    probabilities = counts / float(np.sum(counts))
    raw = -float(np.sum(probabilities * np.log2(np.clip(probabilities, 1e-12, 1.0))))
    return float(raw / max(math.log2(len(counts)), 1.0))


def _summary(prefix: str, values: Iterable[float], output: dict[str, float]) -> None:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        array = np.asarray([0.0], dtype=float)
    output[f"{prefix}__mean"] = float(np.mean(array))
    output[f"{prefix}__std"] = float(np.std(array))
    output[f"{prefix}__q10"] = float(np.quantile(array, 0.10))
    output[f"{prefix}__q50"] = float(np.quantile(array, 0.50))
    output[f"{prefix}__q90"] = float(np.quantile(array, 0.90))
    output[f"{prefix}__max"] = float(np.max(array))


def _ngrams(sequence: tuple[str, ...], size: int) -> frozenset[tuple[str, ...]]:
    if len(sequence) < size:
        return frozenset()
    return frozenset(
        tuple(sequence[index : index + size])
        for index in range(len(sequence) - size + 1)
    )


def _jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _hand_sequences(hand: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
    hero_seat = int(_number(metadata.get("hero_seat")))
    actions = [row for row in (hand.get("actions") or []) if isinstance(row, dict)]
    result: dict[str, list[str]] = {name: [] for name in CHANNELS}
    for row in actions:
        action = str(row.get("action_type") or "other").strip().lower()
        street = str(row.get("street") or "other").strip().lower()
        actor = int(_number(row.get("actor_seat")))
        role = "h" if hero_seat > 0 and actor == hero_seat else "o"
        amount = _amount_bucket(row.get("normalized_amount_bb"))
        result["action"].append(action)
        result["street_action"].append(f"{street}:{action}")
        result["hero_action"].append(f"{role}:{action}")
        result["amount_action"].append(f"{action}:{amount}")
    return {name: tuple(values) for name, values in result.items()}


def _conditional_determinism(sequences: list[tuple[str, ...]]) -> tuple[float, float]:
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            transitions[left][right] += 1
    total = sum(sum(counts.values()) for counts in transitions.values())
    if total <= 0:
        return 0.0, 0.0
    weighted_entropy = 0.0
    weighted_top = 0.0
    for counts in transitions.values():
        count = sum(counts.values())
        weight = count / total
        weighted_entropy += weight * _entropy(
            token for token, frequency in counts.items() for _ in range(frequency)
        )
        weighted_top += weight * (max(counts.values()) / count)
    return float(weighted_entropy), float(weighted_top)


def _compression_ratio(sequences: list[tuple[str, ...]]) -> float:
    raw = "|".join(",".join(sequence) for sequence in sorted(sequences)).encode()
    if not raw:
        return 0.0
    compressed = zlib.compress(raw, level=9)
    return float(len(compressed) / len(raw))


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Describe exact and near repetition without observing source hand order."""
    hands = canonical_hand_bag(list(chunk or []), max_hands=PAIRWISE_HAND_CAP)
    sequences = [_hand_sequences(hand) for hand in hands]
    output: dict[str, float] = {}
    for channel in CHANNELS:
        channel_sequences = [row[channel] for row in sequences]
        counts = Counter(channel_sequences)
        denominator = max(len(channel_sequences), 1)
        output[f"{channel}__exact_unique_share"] = len(counts) / denominator
        output[f"{channel}__exact_top_share"] = (
            max(counts.values()) / denominator if counts else 0.0
        )
        output[f"{channel}__exact_repeat_share"] = (
            sum(value for value in counts.values() if value > 1) / denominator
        )
        output[f"{channel}__exact_entropy"] = _entropy(channel_sequences)
        conditional_entropy, conditional_top = _conditional_determinism(channel_sequences)
        output[f"{channel}__conditional_entropy"] = conditional_entropy
        output[f"{channel}__conditional_top_share"] = conditional_top
        output[f"{channel}__compression_ratio"] = _compression_ratio(channel_sequences)

        for size in NGRAM_SIZES:
            sets = [_ngrams(sequence, size) for sequence in channel_sequences]
            pair_values = [_jaccard(left, right) for left, right in combinations(sets, 2)]
            _summary(f"{channel}__ngram{size}__pair_jaccard", pair_values, output)
            nearest: list[float] = []
            for index, current in enumerate(sets):
                others = [
                    _jaccard(current, other)
                    for other_index, other in enumerate(sets)
                    if other_index != index
                ]
                nearest.append(max(others) if others else 0.0)
            _summary(f"{channel}__ngram{size}__nearest", nearest, output)
    return output
