"""Sequence-only heuristic V8.

V5/V6 use:
  - hand-signature HASH (exact match)
  - bet sizing quantization
  - per-seat behavioral stats

V8 attacks a DIFFERENT signal family:
  - action n-gram (3-gram and 4-gram) repetition across hands (NOT hash match)
  - longest common SUBSEQUENCE (LCS) — captures procedural similarity even with
    different surface form
  - motif entropy — diversity of action sequence motifs in chunk
  - top motif concentration

This is orthogonal because:
  - v5_repetition uses hand-signature hash (action+sizing bucket); v8 uses pure
    action n-grams ignoring sizing
  - LCS catches "same procedural path, different surface" cases v5 misses

If bot generator produces hands following same PROCEDURE (same n-gram skeleton)
but with varying sizings/seat assignments, v5 misses but v8 catches.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

import numpy as np


def _hand_action_sequence(hand: dict) -> Tuple[str, ...]:
    """Return action types as tuple (no sizing). e.g. ('f',), ('c','c','k','b','f')."""
    actions = hand.get("actions") or []
    return tuple((a.get("action_type") or "?")[0] for a in actions)  # first char


def _ngrams(seq: Tuple[str, ...], n: int) -> List[Tuple[str, ...]]:
    if len(seq) < n:
        return []
    return [seq[i:i + n] for i in range(len(seq) - n + 1)]


def _lcs_length(a: Tuple[str, ...], b: Tuple[str, ...]) -> int:
    """Standard LCS DP. O(len(a) * len(b))."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    prev = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur = [0] * (lb + 1)
        ai = a[i - 1]
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[lb]


def _entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    p = [c / total for c in counts if c > 0]
    return -sum(pi * math.log2(pi) for pi in p)


def score_chunk_v8(hands: List[dict]) -> float:
    """V8 score [0,1]. Higher = more procedural similarity → bot-like.

    4 sub-signals:
      1. ngram3_repetition: fraction of 3-grams that appear >= 3 times globally
      2. ngram4_repetition: fraction of 4-grams that appear >= 2 times globally
      3. lcs_avg_ratio: average pairwise LCS / max_len over a sample of hand pairs
      4. motif_entropy_lo: 1 - normalized entropy of hand-sequence motifs (low entropy = bot)
    """
    if not hands:
        return 0.5

    n_hands = len(hands)
    seqs = [_hand_action_sequence(h) for h in hands]
    # Skip empty sequences
    non_empty = [s for s in seqs if s]
    if not non_empty:
        return 0.5

    # 1. 3-gram repetition
    all_3grams: List[Tuple[str, ...]] = []
    for s in non_empty:
        all_3grams.extend(_ngrams(s, 3))
    ng3_counts = Counter(all_3grams)
    if all_3grams:
        n_repeated_3 = sum(c for c in ng3_counts.values() if c >= 3)
        ngram3_repetition = n_repeated_3 / len(all_3grams)
    else:
        ngram3_repetition = 0.0

    # 2. 4-gram repetition
    all_4grams: List[Tuple[str, ...]] = []
    for s in non_empty:
        all_4grams.extend(_ngrams(s, 4))
    ng4_counts = Counter(all_4grams)
    if all_4grams:
        n_repeated_4 = sum(c for c in ng4_counts.values() if c >= 2)
        ngram4_repetition = n_repeated_4 / len(all_4grams)
    else:
        ngram4_repetition = 0.0

    # 3. LCS — sample of pairs (capped to keep O(N^2) bounded)
    # For 40 hands → 780 pairs, each ~12x12 LCS = manageable. Cap at 100 pairs.
    pair_ratios = []
    max_pairs = 100
    cnt = 0
    for i in range(len(non_empty)):
        for j in range(i + 1, len(non_empty)):
            if cnt >= max_pairs:
                break
            a, b = non_empty[i], non_empty[j]
            ml = max(len(a), len(b))
            if ml == 0:
                continue
            lcs = _lcs_length(a, b)
            pair_ratios.append(lcs / ml)
            cnt += 1
        if cnt >= max_pairs:
            break
    lcs_avg_ratio = float(np.mean(pair_ratios)) if pair_ratios else 0.0

    # 4. Motif entropy — diversity of full hand sequences
    motif_counts = Counter(non_empty)
    H = _entropy(list(motif_counts.values()))
    H_max = math.log2(max(len(motif_counts), 1)) if motif_counts else 1.0
    motif_entropy_norm = H / max(H_max, 1e-6)
    motif_entropy_lo = 1.0 - motif_entropy_norm

    # Combined weighted (sums to 1.0): emphasize n-gram + LCS (procedural similarity)
    score = (
        0.30 * ngram3_repetition
        + 0.30 * ngram4_repetition
        + 0.25 * lcs_avg_ratio
        + 0.15 * motif_entropy_lo
    )
    return max(0.0, min(1.0, score))


def score_chunks_v8(chunks: List[List[dict]]) -> List[float]:
    return [score_chunk_v8(c) for c in chunks]
