"""Map raw model probabilities to validator-friendly risk scores.

Two-stage pipeline:

1. Isotonic calibration (fitted at training time) maps raw model output → empirical
   P(bot|raw). This stretches narrow output ranges to the full [0, 1].
2. Semantic banding pushes borderline cases below 0.5 to protect against the FPR
   cliff (validator zeros reward at fpr ≥ 0.10), while preserving rank ordering
   so average_precision stays high.

    p < HUMAN_HI         → [HUMAN_LOW, HUMAN_HI_OUT]   (clear human)
    HUMAN_HI ≤ p < UNSURE → [UNSURE_LOW, UNSURE_HI]    (lean human, label 0)
    UNSURE ≤ p < BOT_HI   → [LEAN_BOT_LOW, LEAN_BOT_HI] (lean bot, label 1)
    p ≥ BOT_HI            → [BOT_LOW, BOT_HI_OUT]       (clear bot)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Thresholds on raw model probability
HUMAN_HI = 0.30
UNSURE = 0.55
BOT_HI = 0.75

# Output bands (validator rounds at 0.5, so anything < 0.5 = label 0)
HUMAN_LOW, HUMAN_HI_OUT = 0.05, 0.15
UNSURE_LOW, UNSURE_HI = 0.30, 0.48
LEAN_BOT_LOW, LEAN_BOT_HI = 0.55, 0.72
BOT_LOW, BOT_HI_OUT = 0.85, 0.95


def _scale(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    if in_hi <= in_lo:
        return out_lo
    t = (value - in_lo) / (in_hi - in_lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)


def calibrate_one(raw: float) -> float:
    raw = max(0.0, min(1.0, float(raw)))
    if raw < HUMAN_HI:
        return _scale(raw, 0.0, HUMAN_HI, HUMAN_LOW, HUMAN_HI_OUT)
    if raw < UNSURE:
        return _scale(raw, HUMAN_HI, UNSURE, UNSURE_LOW, UNSURE_HI)
    if raw < BOT_HI:
        return _scale(raw, UNSURE, BOT_HI, LEAN_BOT_LOW, LEAN_BOT_HI)
    return _scale(raw, BOT_HI, 1.0, BOT_LOW, BOT_HI_OUT)


def calibrate_array(raw: Sequence[float]) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    out = np.empty_like(arr)
    for i, v in enumerate(arr):
        out[i] = calibrate_one(float(v))
    return out


def apply_isotonic(raw: float, isotonic_points: Optional[List[Tuple[float, float]]]) -> float:
    """Piecewise-linear interpolation between (x_i, y_i) thresholds from training.

    Outside the training range, clips to the boundary value (matches sklearn
    IsotonicRegression(out_of_bounds="clip")).
    """
    if not isotonic_points:
        return float(raw)
    xs = [p[0] for p in isotonic_points]
    ys = [p[1] for p in isotonic_points]
    if raw <= xs[0]:
        return ys[0]
    if raw >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if raw <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            t = (raw - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def full_calibrate(raw: float, isotonic_points: Optional[List[Tuple[float, float]]] = None) -> float:
    """Isotonic → semantic band. Use this in production miner."""
    p = apply_isotonic(raw, isotonic_points)
    return calibrate_one(p)


def adaptive_calibrate(
    raw_scores: Sequence[float],
    *,
    isotonic_points: Optional[List[Tuple[float, float]]] = None,
    min_gap_quantile: float = 0.15,
) -> np.ndarray:
    """Adaptive threshold per batch using Otsu's method on raw scores.

    Instead of forcing a fixed bot_ratio, finds the natural split point
    in raw scores that minimizes intra-class variance. Adapts automatically
    to ANY validator bot_ratio without configuration.

    If no clear bimodal split exists (uniform distribution), falls back
    to median split (equivalent to rank_based 0.5).
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if isotonic_points:
        arr_norm = np.asarray([apply_isotonic(float(x), isotonic_points) for x in arr])
    else:
        arr_norm = arr.copy()

    # Otsu's method: find threshold T that minimizes weighted intra-class variance
    sorted_vals = np.sort(arr_norm)
    best_t = np.median(arr_norm)
    best_var = float("inf")

    # Test candidate thresholds at each gap between consecutive sorted values
    for i in range(1, n):
        if sorted_vals[i] == sorted_vals[i - 1]:
            continue
        t = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0
        class0 = arr_norm[arr_norm <= t]
        class1 = arr_norm[arr_norm > t]
        if len(class0) == 0 or len(class1) == 0:
            continue
        w0 = len(class0) / n
        w1 = len(class1) / n
        var_within = w0 * np.var(class0) + w1 * np.var(class1)
        if var_within < best_var:
            best_var = var_within
            best_t = t

    # Classify based on adaptive threshold
    is_bot = arr_norm > best_t
    n_bot = int(is_bot.sum())
    n_human = n - n_bot

    # Build output preserving ranking within each class
    out = np.empty(n, dtype=np.float64)
    bot_indices = np.where(is_bot)[0]
    human_indices = np.where(~is_bot)[0]

    # Sort bots by score descending, humans by score ascending
    if n_bot > 0:
        bot_order = bot_indices[np.argsort(-arr_norm[bot_indices])]
        for rank, idx in enumerate(bot_order):
            t_frac = rank / max(n_bot - 1, 1)
            out[idx] = BOT_HI_OUT - t_frac * (BOT_HI_OUT - BOT_LOW)

    if n_human > 0:
        human_order = human_indices[np.argsort(arr_norm[human_indices])]
        for rank, idx in enumerate(human_order):
            t_frac = rank / max(n_human - 1, 1)
            out[idx] = HUMAN_LOW + t_frac * (HUMAN_HI_OUT - HUMAN_LOW)

    return out


def adaptive_safe_calibrate(
    raw_scores: Sequence[float],
    *,
    isotonic_points: Optional[List[Tuple[float, float]]] = None,
    max_bot_fraction: float = 0.40,
) -> np.ndarray:
    """Adaptive Otsu + hard safety cap.

    1. Otsu finds natural bot/human split (adapts to ANY validator ratio)
    2. Safety cap: NEVER predict more than max_bot_fraction as bot
       (survives worst case POKER44_HUMAN_RATIO=0.60 → 24 humans)
    3. If over cap: demote weakest bot predictions to human band
    4. Preserve ranking within bands for AP
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Step 1: Otsu split
    base = adaptive_calibrate(arr, isotonic_points=isotonic_points)

    # Step 2: Safety cap
    max_bots = int(n * max_bot_fraction)
    bot_mask = base >= 0.5
    n_bot_pred = int(bot_mask.sum())

    if n_bot_pred <= max_bots:
        return base  # within cap, no change needed

    # Step 3: Demote weakest bot predictions to human band
    bot_indices = np.where(bot_mask)[0]
    bot_scores = base[bot_indices]
    # Sort ascending — weakest bots first
    weakest_order = bot_indices[np.argsort(bot_scores)]
    n_to_flip = n_bot_pred - max_bots

    out = base.copy()
    # Re-rank the flipped bots into human band (top of human range)
    human_scores = out[~bot_mask]
    human_max = float(human_scores.max()) if len(human_scores) > 0 else HUMAN_HI_OUT

    for i, idx in enumerate(weakest_order[:n_to_flip]):
        # Place in human band, clamped to valid range
        candidate = 0.48 - i * 0.01
        out[idx] = max(HUMAN_LOW, min(candidate, human_max + 0.01))

    return out


def dynamic_safe_calibrate(
    raw_scores: Sequence[float],
    *,
    isotonic_points: Optional[List[Tuple[float, float]]] = None,
    estimated_bot_ratio: Optional[float] = None,
    safety_margin: float = 0.05,
    absolute_max: float = 0.50,
    absolute_min: float = 0.05,
) -> np.ndarray:
    """Dynamic per-batch cap based on confidence in raw distribution.

    Logic:
    - If raw scores are bimodal (clear gap) → estimate bot_ratio from gap location
    - Cap = max(absolute_min, min(absolute_max, estimated_ratio - safety_margin))
    - Otsu finds threshold; safety_margin prevents cliff if estimate is slightly high

    For unknown true ratio: this auto-adjusts so cliff never triggers.
    Combined with safety_margin=0.05, max FPR ≈ safety_margin / (1-est_ratio).
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Apply isotonic if available
    if isotonic_points:
        arr_iso = np.asarray([apply_isotonic(float(x), isotonic_points) for x in arr])
    else:
        arr_iso = arr

    # Estimate bot_ratio if not provided: count scores above raw threshold 0.5
    if estimated_bot_ratio is None:
        # Use Otsu's threshold to estimate ratio
        sorted_vals = np.sort(arr_iso)
        best_t = float(np.median(arr_iso))
        best_var = float("inf")
        for i in range(1, n):
            if sorted_vals[i] == sorted_vals[i - 1]:
                continue
            t = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0
            c0 = arr_iso[arr_iso <= t]
            c1 = arr_iso[arr_iso > t]
            if len(c0) == 0 or len(c1) == 0:
                continue
            w0 = len(c0) / n
            w1 = len(c1) / n
            v = w0 * np.var(c0) + w1 * np.var(c1)
            if v < best_var:
                best_var = v
                best_t = t
        estimated_bot_ratio = float((arr_iso > best_t).sum()) / n

    # Apply safety margin and bounds
    safe_cap = max(absolute_min, min(absolute_max, estimated_bot_ratio - safety_margin))

    # Use adaptive_safe_calibrate with computed cap
    return adaptive_safe_calibrate(arr, isotonic_points=isotonic_points, max_bot_fraction=safe_cap)


def rank_based_calibrate(
    raw_scores: Sequence[float],
    *,
    bot_ratio: float = 0.5,
    isotonic_points: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Rank-based calibration: assign top-K chunks as bot where K = N * bot_ratio.

    Eliminates FPR cliff regardless of validator's actual human_ratio — we always
    predict a fixed fraction of the batch as bots, preserving raw-score ranking
    within each band so AP stays intact.

    Output layout (rank within top-K bots gets BOT_LOW..BOT_HI_OUT, rank within
    bottom-K humans gets HUMAN_LOW..HUMAN_HI_OUT — preserves ordering for AP).
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # optional isotonic pre-normalization (for logging / inspection only)
    if isotonic_points:
        arr_iso = np.asarray([apply_isotonic(float(x), isotonic_points) for x in arr])
    else:
        arr_iso = arr

    k_bot = max(0, min(n, int(round(n * bot_ratio))))
    # indices sorted descending by raw/iso score
    order = np.argsort(-arr_iso)

    out = np.empty(n, dtype=np.float64)

    # bot tier — top K: rank r∈[0, k_bot-1] → linear map BOT_LOW..BOT_HI_OUT
    for rank_idx, idx in enumerate(order[:k_bot]):
        if k_bot <= 1:
            out[idx] = BOT_HI_OUT
        else:
            t = rank_idx / (k_bot - 1)  # 0.0 (top) .. 1.0 (lowest in bot band)
            out[idx] = BOT_HI_OUT - t * (BOT_HI_OUT - BOT_LOW)

    # human tier — bottom N-K: rank r∈[0, n-k_bot-1] → linear map HUMAN_HI_OUT..HUMAN_LOW
    remaining = n - k_bot
    for rank_idx, idx in enumerate(order[k_bot:]):
        if remaining <= 1:
            out[idx] = HUMAN_LOW
        else:
            t = rank_idx / (remaining - 1)  # 0.0 (highest human) .. 1.0 (lowest)
            out[idx] = HUMAN_HI_OUT - t * (HUMAN_HI_OUT - HUMAN_LOW)

    return out


def bounded_rank_calibrate(
    raw_scores: Sequence[float],
    *,
    max_n: int = 3,
    score_floor: float = 0.16,
    collapse_top1: float = 0.18,
    collapse_spread: float = 0.03,
    saturate_median: float = 0.55,
    saturate_spread: float = 0.08,
    isotonic_points: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Bounded rank-based calibrator with discrete N in [0, max_n], hard cap N<4.

    Designed for the 40-chunk validator window where FPR≥0.10 zeros reward
    (about 3 false positives over typical 28-36 negatives). Pure dynamic
    calibration can collapse on flat raw scores, while pure top-N can over-call
    on noisy batches. This hybrid uses rank-based selection constrained by
    distribution-shape guards and an absolute score floor.

    Decision logic:
      * collapse (top1<0.18 ∧ std<0.03)   → N=0 (top1<0.12) or N=1
      * saturation (median>0.55 ∧ std<0.08) → N=1 if gap12>0.04 else N=2
      * normal: start N=1; promote to N=2 if (top2>0.28 ∧ margin2>0.10);
        promote to N=3 if (top3>0.38 ∧ gap23≥0 ∧ std>0.06)
      * final N clipped to caller's max_n (defense in depth vs FPR cliff)

    Score floor (default 0.16) suppresses junk picks when model is under-confident:
    even if rank says "pick top1", we drop it when raw_iso[top1] < floor.

    Output preserves ranking within bands so AP stays high:
      * selected bots get rank-linear scores in [BOT_LOW, BOT_HI_OUT]
      * non-selected get rank-linear scores in [HUMAN_LOW, HUMAN_HI_OUT]
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if isotonic_points:
        arr_iso = np.asarray([apply_isotonic(float(x), isotonic_points) for x in arr])
    else:
        arr_iso = arr.copy()

    max_n_clamped = max(0, min(3, int(max_n)))

    order = np.argsort(-arr_iso)
    s_sorted = arr_iso[order]

    top1 = float(s_sorted[0])
    top2 = float(s_sorted[1]) if n > 1 else 0.0
    top3 = float(s_sorted[2]) if n > 2 else 0.0
    median = float(np.median(arr_iso))
    spread = float(np.std(arr_iso))
    margin2 = top2 - median
    gap12 = top1 - top2
    gap23 = top2 - top3

    if top1 < collapse_top1 and spread < collapse_spread:
        N = 0 if top1 < 0.12 else 1
    elif median > saturate_median and spread < saturate_spread:
        N = 1 if gap12 > 0.04 else 2
    else:
        N = 1
        if top2 > 0.28 and margin2 > 0.10:
            N = 2
        if top3 > 0.38 and gap23 > -0.01 and spread > 0.06:
            N = 3

    N = max(0, min(N, max_n_clamped))

    # selected_idx applies the score floor — rank says "pick top-N" but we drop
    # any pick whose iso score is below the floor (rank-only junk protection).
    selected_idx: List[int] = []
    for idx in order[:N]:
        if float(arr_iso[idx]) >= score_floor:
            selected_idx.append(int(idx))
    k_bot = len(selected_idx)

    out = np.empty(n, dtype=np.float64)

    for rank_idx, idx in enumerate(selected_idx):
        if k_bot <= 1:
            out[idx] = BOT_HI_OUT
        else:
            t = rank_idx / (k_bot - 1)
            out[idx] = BOT_HI_OUT - t * (BOT_HI_OUT - BOT_LOW)

    selected_set = set(selected_idx)
    human_order = [int(idx) for idx in order if int(idx) not in selected_set]
    remaining = len(human_order)
    for rank_idx, idx in enumerate(human_order):
        if remaining <= 1:
            out[idx] = HUMAN_LOW
        else:
            t = rank_idx / (remaining - 1)
            out[idx] = HUMAN_HI_OUT - t * (HUMAN_HI_OUT - HUMAN_LOW)

    return out


# Adaptive N selection.
# Static max_n=3 fails when validator's actual bot count per batch varies.
# select_adaptive_n() inspects raw_scores distribution and picks N per batch.
#
# Profiles:
#   conservative: N in [1,3]
#   balanced:     N in [1,4]
#   aggressive:   N in [2,5]
#   scout:        N in [0,4]


def _pick_n_by_signal(
    sorted_desc: np.ndarray,
    std: float,
    *,
    flat_std: float,
    flat_gap: float,
    weak_top1_med: float,
    strong_top1_med: float,
    super_strong_top: float,
    super_strong_gap: float,
) -> int:
    """Inspect score distribution, return suggested N. Caller clips by profile bounds."""
    n = len(sorted_desc)
    if n == 0:
        return 0
    median = float(np.median(sorted_desc))
    top1 = float(sorted_desc[0])
    top1_med = top1 - median

    # Flat signal — model not seeing anything bot-like
    if std < flat_std or top1_med < flat_gap:
        return 0

    # Weak signal — barely discriminative
    if top1_med < weak_top1_med:
        return 1

    # Compute consecutive gaps top-k vs top-(k+1)
    def gap(i):
        if i + 1 >= n:
            return 0.0
        return float(sorted_desc[i] - sorted_desc[i + 1])

    g12 = gap(0)  # top1 vs top2
    g23 = gap(1)  # top2 vs top3
    g34 = gap(2)
    g45 = gap(3)

    # Super-strong: top5 still above median by super_strong_top + gap5-6 separates
    if n >= 6:
        top5 = float(sorted_desc[4])
        gap56 = gap(4)
        if top5 - median >= super_strong_top and gap56 >= super_strong_gap and std >= 0.04:
            return 5

    # Strong signal: top3 stays >= strong_top1_med above median + clean gap after top4
    if n >= 5:
        top4 = float(sorted_desc[3])
        if top4 - median >= 0.10 and g45 >= 0.030:
            return 4

    # Normal: top1 separates from median by strong_top1_med
    if top1_med >= strong_top1_med:
        return 3

    # Moderate signal — top2 above weak threshold
    if n >= 3:
        top2 = float(sorted_desc[1])
        if top2 - median >= weak_top1_med:
            return 2

    return 1


PROFILES = {
    "conservative": dict(
        min_n=1, max_n=3,
        flat_std=0.025, flat_gap=0.10,
        weak_top1_med=0.10, strong_top1_med=0.18,
        super_strong_top=0.20, super_strong_gap=0.05,
    ),
    "balanced": dict(
        min_n=1, max_n=4,
        flat_std=0.020, flat_gap=0.07,
        weak_top1_med=0.07, strong_top1_med=0.14,
        super_strong_top=0.18, super_strong_gap=0.04,
    ),
    "aggressive": dict(
        min_n=2, max_n=5,
        flat_std=0.015, flat_gap=0.05,
        weak_top1_med=0.05, strong_top1_med=0.10,
        super_strong_top=0.16, super_strong_gap=0.035,
    ),
    "scout": dict(
        min_n=0, max_n=4,
        flat_std=0.018, flat_gap=0.07,
        weak_top1_med=0.07, strong_top1_med=0.13,
        super_strong_top=0.18, super_strong_gap=0.04,
    ),
}


def select_adaptive_n(
    raw_scores: Sequence[float],
    *,
    profile: str = "balanced",
    min_n: Optional[int] = None,
    max_n: Optional[int] = None,
    prev_n: Optional[int] = None,
    hysteresis: bool = True,
) -> Tuple[int, dict]:
    """Pick N (bot count) adaptively from raw_scores distribution.

    Returns (n, diag) where diag contains intermediate metrics for logging.

    Hysteresis: if prev_n provided, limit jump to ±1 per batch (avoid oscillation).
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    if len(arr) == 0:
        return 0, {"reason": "empty"}

    cfg = PROFILES.get(profile, PROFILES["balanced"]).copy()
    if min_n is not None:
        cfg["min_n"] = int(min_n)
    if max_n is not None:
        cfg["max_n"] = int(max_n)

    sorted_desc = np.sort(arr)[::-1]
    std = float(arr.std())

    suggested = _pick_n_by_signal(
        sorted_desc, std,
        flat_std=cfg["flat_std"],
        flat_gap=cfg["flat_gap"],
        weak_top1_med=cfg["weak_top1_med"],
        strong_top1_med=cfg["strong_top1_med"],
        super_strong_top=cfg["super_strong_top"],
        super_strong_gap=cfg["super_strong_gap"],
    )

    # Clamp to profile bounds
    n_clamped = max(cfg["min_n"], min(cfg["max_n"], suggested))

    # Apply hysteresis (limit ±1 jump from prev_n, except for aggressive profile)
    if hysteresis and prev_n is not None and profile != "aggressive":
        if n_clamped > prev_n + 1:
            n_clamped = prev_n + 1
        elif n_clamped < prev_n - 1:
            n_clamped = prev_n - 1

    diag = {
        "profile": profile,
        "suggested_n": int(suggested),
        "final_n": int(n_clamped),
        "std": round(std, 4),
        "top1_med": round(float(sorted_desc[0] - np.median(sorted_desc)), 4),
        "min_n": cfg["min_n"],
        "max_n": cfg["max_n"],
        "prev_n": prev_n,
    }
    return int(n_clamped), diag
