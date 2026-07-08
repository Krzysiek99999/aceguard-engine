"""Inference wrapper for locally rebuilt super-v2 stacked artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

_CACHE: dict[str, tuple[float, Any]] = {}


def load_model(path: str | os.PathLike[str]) -> Any:
    p = str(path)
    mtime = os.path.getmtime(p)
    cached = _CACHE.get(p)
    if cached is None or cached[0] != mtime:
        from poker44_ml.inference import Poker44Model

        _CACHE[p] = (mtime, Poker44Model(p))
    return _CACHE[p][1]


def score_from_file(
    chunks: Sequence[Any],
    model_path: str | os.PathLike[str],
) -> list[float]:
    model = load_model(model_path)
    return [float(value) for value in model.predict_chunk_scores(list(chunks))]


def is_available(model_path: str | os.PathLike[str] | None = None) -> bool:
    if model_path is None:
        model_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "models"
            / "v219_rebuilt_superv2"
            / "model.joblib"
        )
    return Path(model_path).exists()
