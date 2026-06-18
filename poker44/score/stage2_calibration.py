"""V10 deterministic stage-2 calibration.

The same base scorer can be exposed with different risk postures.

Three modes:
  - mild:     compressed scores (less aggressive top-N), Nmax=2, higher floor
  - balanced: identity (same as baseline v5), Nmax=3
  - sharp:    stretched scores (more aggressive top-N), Nmax=3 + rare 4, low floor

Used as second stage AFTER any base scorer (v5, v6, v9). Profile chosen by variant
config, NOT detected automatically.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def stage2_calibrate(scores: List[float], mode: str = "balanced") -> Tuple[List[float], int, float]:
    """Apply stage-2 transform + return (calibrated_scores, max_n, score_floor).

    mode:
      'mild':     compress around median, max_n=2, floor=0.20
      'balanced': identity, max_n=3, floor=0.0
      'sharp':   stretch around median, max_n=3 (rare 4), floor=0.0

    Recent-distribution median is computed from input scores.
    """
    if not scores:
        return [], 2, 0.0

    arr = np.array(scores, dtype=np.float64)
    median = float(np.median(arr))

    if mode == "mild":
        # Compress around median — less aggressive, only highest-confidence picks
        centered = arr - median
        calibrated = median + 0.75 * centered
        max_n = 2
        floor = 0.20  # higher floor — only picks well above median get through

    elif mode == "sharp":
        # Stretch around median — amplify top scores, more aggressive picks
        centered = arr - median
        calibrated = median + 1.35 * centered
        max_n = 3  # could escalate to 4 in exceptional cases
        floor = 0.0  # accept any rank-based pick

    else:  # balanced
        calibrated = arr.copy()
        max_n = 3
        floor = 0.0

    # Clip to [0, 1]
    calibrated = np.clip(calibrated, 0.0, 1.0)
    return calibrated.tolist(), max_n, floor


def stage2_max_n_adaptive(scores: List[float], mode: str = "balanced") -> int:
    """For 'sharp' mode, optionally escalate to N=4 on exceptional separation.

    Used by v9_adaptive_n-like variants combined with sharp profile.
    """
    if mode != "sharp" or len(scores) < 5:
        return 3 if mode != "mild" else 2

    sd = np.sort(np.array(scores))[::-1]
    top1, top4, top5 = float(sd[0]), float(sd[3]), float(sd[4])
    median = float(np.median(scores))
    std_top5 = float(np.std(sd[:5]))
    gap_45 = top4 - top5

    # Same gates as adaptive top-4 mode.
    if (
        top4 >= 0.52
        and gap_45 >= 0.06
        and (top4 - median) >= 0.20
        and std_top5 >= 0.05
    ):
        return 4
    return 3
