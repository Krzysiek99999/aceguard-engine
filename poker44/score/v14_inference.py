"""V14 inference wrapper — load LightGBM + isotonic + score_shift + rank_cap_remap.

Loaded lazily on first call. Model file: models/v14_live_stable.txt
Meta file: models/v14_live_stable_meta.json (contains feature_names, isotonic
thresholds, score_shift_logit, best_iteration).

API:
  score_chunks_v14_rank(chunks, top_n) -> list[float] in [0.05, 0.95]
  Guarantees: exactly top_n scores >= 0.5 after remap.

NIE używa joblib/pickle. LightGBM Booster loaded z .txt (LightGBM native format).
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from poker44.score.features_v13_safe import chunk_features_v13
from poker44.score.rank_cap_remap import rank_cap_remap


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / 'models' / 'v14_live_stable.txt'
META_PATH = REPO_ROOT / 'models' / 'v14_live_stable_meta.json'

_BOOSTER = None
_META = None
_FEATURE_NAMES: list[str] | None = None
_ISO_X = None
_ISO_Y = None
_SHIFT = 0.0
_BEST_ITER = 0


def _lazy_load() -> bool:
    global _BOOSTER, _META, _FEATURE_NAMES, _ISO_X, _ISO_Y, _SHIFT, _BEST_ITER
    if _BOOSTER is not None:
        return True
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return False
    try:
        import lightgbm as lgb
        _BOOSTER = lgb.Booster(model_file=str(MODEL_PATH))
    except Exception:
        _BOOSTER = None
        return False
    _META = json.loads(META_PATH.read_text())
    _FEATURE_NAMES = list(_META.get('feature_names') or [])
    _ISO_X = np.asarray(_META.get('isotonic_thresholds_x') or [], dtype=float)
    _ISO_Y = np.asarray(_META.get('isotonic_thresholds_y') or [], dtype=float)
    _SHIFT = float(_META.get('score_shift_logit') or 0.0)
    _BEST_ITER = int(_META.get('best_iteration') or 0)
    return True


def _apply_isotonic(values: np.ndarray) -> np.ndarray:
    if _ISO_X is None or len(_ISO_X) == 0:
        return values.clip(0.0, 1.0)
    return np.interp(values, _ISO_X, _ISO_Y).clip(0.0, 1.0)


def _apply_shift(values: np.ndarray) -> np.ndarray:
    if abs(_SHIFT) < 1e-12:
        return values
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) + _SHIFT
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def score_chunks_v14_raw(chunks: Sequence[Sequence[dict]]) -> list[float]:
    """Return calibrated+shifted scores per chunk (without rank cap). Range [0, 1]."""
    if not _lazy_load():
        # Fallback: zero everything (caller should handle absence)
        return [0.5 for _ in chunks]
    assert _FEATURE_NAMES is not None
    X = np.zeros((len(chunks), len(_FEATURE_NAMES)), dtype=np.float32)
    for i, c in enumerate(chunks):
        f = chunk_features_v13(c)
        for j, name in enumerate(_FEATURE_NAMES):
            X[i, j] = float(f.get(name, 0.0))
    raw = _BOOSTER.predict(X, num_iteration=_BEST_ITER)
    cal = _apply_isotonic(np.asarray(raw))
    shifted = _apply_shift(cal)
    return [float(v) for v in shifted]


def score_chunks_v14_rank(chunks: Sequence[Sequence[dict]], top_n: int) -> list[float]:
    """Score + rank-cap remap. Guarantees exactly top_n scores >= 0.5."""
    raw_calibrated = score_chunks_v14_raw(chunks)
    return rank_cap_remap(raw_calibrated, top_n)


def is_available() -> bool:
    return _lazy_load()


def model_meta_summary() -> dict:
    if not _lazy_load():
        return {'available': False}
    return {
        'available': True,
        'feature_count': len(_FEATURE_NAMES or []),
        'best_iteration': _BEST_ITER,
        'score_shift_logit': _SHIFT,
        'model_path': str(MODEL_PATH),
    }
