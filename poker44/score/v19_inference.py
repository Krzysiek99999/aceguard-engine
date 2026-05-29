"""V19 inference wrapper — LambdaRank na shadow-training API + rank_cap_remap.

Wzorowane na v14_inference.py. ZERO joblib/pickle. Lazy load LightGBM .txt + JSON meta.

API:
  score_chunks_v19_raw(chunks)   -> list[float]   (raw ranker output, możliwie >1 lub <0)
  score_chunks_v19_rank(chunks, top_n) -> list[float] in [0.05, 0.95]
  is_available() -> bool
  model_meta_summary() -> dict

V19 vs V14:
  - V14 = binary classifier + isotonic + score_shift; final score w [0,1]
  - V19 = LambdaRank objective; raw score unbounded (no probability semantics)
  - Dlatego v19 inference NIE robi isotonic/shift — od razu rank_cap_remap
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from poker44.score.features_v13_safe import chunk_features_v13
from poker44.score.rank_cap_remap import rank_cap_remap


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / 'models' / 'v19_ranker.txt'
META_PATH = REPO_ROOT / 'models' / 'v19_ranker_meta.json'

_BOOSTER = None
_META = None
_FEATURE_NAMES: list[str] | None = None
_BEST_ITER = 0


def _lazy_load() -> bool:
    global _BOOSTER, _META, _FEATURE_NAMES, _BEST_ITER
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
    _BEST_ITER = int(_META.get('best_iteration') or 0)
    return True


def score_chunks_v19_raw(chunks: Sequence[Sequence[dict]]) -> list[float]:
    """LambdaRank raw output. Unbounded — używaj tylko jako ranking signal."""
    if not _lazy_load():
        # No fallback to v5 — caller decyduje co robić jeśli model missing
        return []
    assert _FEATURE_NAMES is not None
    X = np.zeros((len(chunks), len(_FEATURE_NAMES)), dtype=np.float32)
    for i, c in enumerate(chunks):
        f = chunk_features_v13(c)
        for j, name in enumerate(_FEATURE_NAMES):
            X[i, j] = float(f.get(name, 0.0))
    raw = _BOOSTER.predict(X, num_iteration=_BEST_ITER if _BEST_ITER > 0 else None)
    return [float(v) for v in raw]


def score_chunks_v19_rank(chunks: Sequence[Sequence[dict]], top_n: int) -> list[float]:
    """Score + rank-cap remap. Gwarantuje exactly top_n positives @0.5 threshold."""
    raw = score_chunks_v19_raw(chunks)
    if not raw:
        return []  # caller handles model-missing case
    return rank_cap_remap(raw, top_n)


def is_available() -> bool:
    return _lazy_load()


def model_meta_summary() -> dict:
    if not _lazy_load():
        return {'available': False, 'model_path': str(MODEL_PATH), 'meta_path': str(META_PATH)}
    return {
        'available': True,
        'model_name': _META.get('model_name', 'poker44-v19-lambdarank'),
        'feature_count': len(_FEATURE_NAMES or []),
        'best_iteration': _BEST_ITER,
        'model_path': str(MODEL_PATH),
        'oof_ap': _META.get('metrics_oof', {}).get('ap'),
        'live_std': _META.get('live', {}).get('std'),
    }
