"""V22 inference wrapper — competitor-style LambdaRank + rank_cap_remap.

Same pattern jako v19_inference. Lazy load LightGBM .txt + JSON meta. NO pickle.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from poker44.score.features_v13_safe import chunk_features_v13
from poker44.score.rank_cap_remap import rank_cap_remap

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / 'models' / 'v22_competitor_ranker.txt'
META_PATH = REPO_ROOT / 'models' / 'v22_competitor_ranker_meta.json'

_BOOSTER = None
_META = None
_FEATURE_NAMES: list[str] | None = None
_BEST_ITER = 0


def _lazy_load() -> bool:
    global _BOOSTER, _META, _FEATURE_NAMES, _BEST_ITER
    if _BOOSTER is not None: return True
    if not MODEL_PATH.exists() or not META_PATH.exists(): return False
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


def score_chunks_v22_raw(chunks: Sequence[Sequence[dict]]) -> list[float]:
    if not _lazy_load(): return []
    X = np.zeros((len(chunks), len(_FEATURE_NAMES)), dtype=np.float32)
    for i, c in enumerate(chunks):
        f = chunk_features_v13(c)
        for j, n in enumerate(_FEATURE_NAMES):
            X[i, j] = float(f.get(n, 0.0))
    raw = _BOOSTER.predict(X, num_iteration=_BEST_ITER if _BEST_ITER > 0 else None)
    return [float(v) for v in raw]


def score_chunks_v22_rank(chunks: Sequence[Sequence[dict]], top_n: int) -> list[float]:
    raw = score_chunks_v22_raw(chunks)
    if not raw: return []
    return rank_cap_remap(raw, top_n)


def is_available() -> bool:
    return _lazy_load()


def model_meta_summary() -> dict:
    if not _lazy_load():
        return {'available': False, 'model_path': str(MODEL_PATH)}
    return {
        'available': True,
        'model_name': _META.get('model_name', 'poker44-v22-competitor-ranker'),
        'feature_count': len(_FEATURE_NAMES or []),
        'best_iteration': _BEST_ITER,
        'live_std': _META.get('live', {}).get('std'),
        'holdout_ap': _META.get('metrics', {}).get('holdout', {}).get('ap'),
    }
