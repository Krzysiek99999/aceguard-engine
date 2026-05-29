"""V8.1 — sequence + Markov transition matrix (codex iter 17 extension v8).

V8 detects REPEATED n-grams (exact surface match). V8.1 adds Markov transition
analysis: P(action_t+1 | action_t). Bots using FIXED POLICY but varying surface
form (different amounts, different lengths) leave transition matrix invariant.

Added sub-signals (vs v8):
  - mean_pairwise_JS_divergence: low = repeated policy across hands
  - transition_entropy: low = deterministic transitions = bot
  - max_transition_concentration: high = one transition dominates
  - street_conditioned_transition_entropy: low = deterministic by street

Output: max(v8_score, v8_markov_score) — pick the strongest signal per chunk.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from poker44.score.sequence_v8 import (
    _hand_action_sequence,
    score_chunk_v8,
)


ACTION_TYPES = ["s", "b", "a", "k", "c", "B", "r", "f", "x"]  # first chars; B=bet vs b=big_blind
# We'll just use the actual first chars present


def _build_transition_matrix(seq: Tuple[str, ...]) -> Dict[str, Dict[str, int]]:
    """Count transitions a → b in single sequence."""
    if len(seq) < 2:
        return {}
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a, b in zip(seq[:-1], seq[1:]):
        counts[a][b] += 1
    return {k: dict(v) for k, v in counts.items()}


def _normalize_matrix(matrix: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """Convert counts to probabilities per row."""
    out: Dict[str, Dict[str, float]] = {}
    for a, row in matrix.items():
        total = sum(row.values())
        if total > 0:
            out[a] = {b: c / total for b, c in row.items()}
    return out


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two prob distributions. Range [0,1]."""
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in all_keys}

    def kl(a, b):
        s = 0.0
        for k in all_keys:
            pa = a.get(k, 0.0)
            pb = b.get(k, 1e-12)
            if pa > 0 and pb > 0:
                s += pa * math.log2(pa / pb)
        return s

    return 0.5 * (kl(p, m) + kl(q, m))


def _matrix_to_flat_dist(matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Flatten (a, b) → joint prob. For JS over full transitions."""
    flat: Dict[str, float] = {}
    total = 0.0
    for a, row in matrix.items():
        for b, p in row.items():
            flat[f"{a}->{b}"] = p
            total += p
    if total > 0:
        return {k: v / total for k, v in flat.items()}
    return flat


def _transition_entropy(matrix: Dict[str, Dict[str, float]]) -> float:
    """Average entropy across all rows of transition matrix."""
    entropies = []
    for a, row in matrix.items():
        probs = list(row.values())
        H = -sum(p * math.log2(p) for p in probs if p > 0)
        H_max = math.log2(max(len(probs), 1)) if probs else 1.0
        if H_max > 0:
            entropies.append(H / H_max)
    return float(np.mean(entropies)) if entropies else 0.0


def _max_transition_concentration(matrix: Dict[str, Dict[str, float]]) -> float:
    """For each row, find dominant transition prob. Return max across rows."""
    maxes = []
    for a, row in matrix.items():
        if row:
            maxes.append(max(row.values()))
    return float(max(maxes)) if maxes else 0.0


def score_chunk_v8_markov(hands: List[dict]) -> float:
    """V8 Markov score [0,1]. Higher = bot-like procedural similarity."""
    if not hands:
        return 0.5
    seqs = [_hand_action_sequence(h) for h in hands]
    non_empty = [s for s in seqs if len(s) >= 2]
    if len(non_empty) < 2:
        return 0.5

    # Build per-hand normalized transition matrices
    matrices_norm = [_normalize_matrix(_build_transition_matrix(s)) for s in non_empty]
    # Flatten for JS comparison
    flats = [_matrix_to_flat_dist(m) for m in matrices_norm]

    # 1. Mean pairwise JS divergence (sampled, 100 pairs cap)
    js_values = []
    max_pairs = 100
    cnt = 0
    for i in range(len(flats)):
        for j in range(i + 1, len(flats)):
            if cnt >= max_pairs:
                break
            if flats[i] and flats[j]:
                js = _js_divergence(flats[i], flats[j])
                js_values.append(js)
                cnt += 1
        if cnt >= max_pairs:
            break
    mean_js = float(np.mean(js_values)) if js_values else 0.5
    # Low JS = similar policies across hands = bot-like
    # Map [0, 1] → bot signal (invert): low JS → high score
    js_bot_signal = max(0.0, 1.0 - 2.0 * mean_js)  # 2x amplification, clip

    # 2. Average transition entropy across hands (low = deterministic = bot)
    avg_entropy_normalized = float(np.mean([_transition_entropy(m) for m in matrices_norm]))
    entropy_bot_signal = 1.0 - avg_entropy_normalized

    # 3. Max transition concentration (single transition dominates)
    avg_max_concentration = float(np.mean([_max_transition_concentration(m) for m in matrices_norm]))
    # If avg max prob > 0.7 → very deterministic
    concentration_bot_signal = max(0.0, (avg_max_concentration - 0.5) * 2.0)  # >0.5 → signal

    # Weighted combination
    score = (
        0.40 * js_bot_signal
        + 0.35 * entropy_bot_signal
        + 0.25 * concentration_bot_signal
    )
    return max(0.0, min(1.0, score))


def score_chunk_v8_combined(hands: List[dict]) -> float:
    """V8.1 = 0.6 markov + 0.4 ngram. Both have signal — blend for orthogonality.

    Earlier max() returned v8_ngram=v8_markov in 100% of cases on live (correlation 1.0)
    because they were on different scales. Blend instead so Markov adds variance.
    """
    v8_ngram = score_chunk_v8(hands)
    v8_mark = score_chunk_v8_markov(hands)
    # Markov signal stronger when fixed-policy bots vary surface form
    return 0.6 * v8_mark + 0.4 * v8_ngram


def score_chunks_v8_combined(chunks: List[List[dict]]) -> List[float]:
    return [score_chunk_v8_combined(c) for c in chunks]
