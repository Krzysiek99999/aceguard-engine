"""Generic rank-cap remap — gwarantuje exactly top_n pozytywów przy threshold @0.5.

Input: list raw scores (any range, any polarity assumes high=positive).
Output: final scores [0, 1] gdzie:
  - top_n najwyższych raw → final ∈ [0.55, 0.95] (ranking preserved, descending)
  - reszta → final ∈ [0.05, 0.49] (ranking preserved, descending)
  - score >= 0.5 dokładnie dla top_n chunków

Stateless, deterministic, idempotent on rank.
"""
from __future__ import annotations
from typing import Sequence


def rank_cap_remap(raw_scores: Sequence[float], top_n: int) -> list[float]:
    n = len(raw_scores)
    if n == 0:
        return []
    if top_n <= 0:
        # All negative; spread in [0.05, 0.49]
        return _spread_range(raw_scores, lo=0.05, hi=0.49)
    if top_n >= n:
        # All positive; spread in [0.55, 0.95]
        return _spread_range(raw_scores, lo=0.55, hi=0.95)

    # Compute descending rank per index
    sorted_idx = sorted(range(n), key=lambda i: -float(raw_scores[i]))
    rank = [0] * n
    for new_rank, i in enumerate(sorted_idx):
        rank[i] = new_rank

    out = [0.0] * n
    others = n - top_n
    for i in range(n):
        r = rank[i]
        if r < top_n:
            # Top: r=0 → 0.95, r=top_n-1 → 0.55
            if top_n == 1:
                out[i] = 0.75
            else:
                out[i] = 0.95 - (r / (top_n - 1)) * 0.40
        else:
            # Bottom: r=top_n → 0.49, r=n-1 → 0.05
            rel = r - top_n
            if others == 1:
                out[i] = 0.25
            else:
                out[i] = 0.49 - (rel / (others - 1)) * 0.44
        # Clamp dla bezpieczeństwa
        out[i] = max(0.0, min(1.0, out[i]))
    return out


def _spread_range(raw_scores: Sequence[float], lo: float, hi: float) -> list[float]:
    """Spread scores in [lo, hi] zachowując ranking."""
    n = len(raw_scores)
    if n == 0: return []
    if n == 1: return [(lo + hi) / 2]
    sorted_idx = sorted(range(n), key=lambda i: -float(raw_scores[i]))
    out = [0.0] * n
    for new_rank, i in enumerate(sorted_idx):
        # rank 0 → hi (highest), rank n-1 → lo
        out[i] = hi - (new_rank / (n - 1)) * (hi - lo)
    return out


# Sanity self-test (na imporcie nic nie odpala — tylko jeśli run jako main)
if __name__ == '__main__':
    import random
    random.seed(0)
    raw = [random.random() for _ in range(40)]
    for n in [0, 1, 2, 3, 5, 10, 40]:
        out = rank_cap_remap(raw, n)
        pos = sum(1 for x in out if x >= 0.5)
        print(f"top_n={n}: positives_after_remap={pos} (expected={n}) range=[{min(out):.3f}, {max(out):.3f}]")
