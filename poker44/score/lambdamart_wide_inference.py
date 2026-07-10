"""Serve single or rank-space-ensemble LambdaMART wide-feature bundles."""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata


def _wide_runtime_dir() -> Path:
    repo = Path(__file__).resolve().parents[2]
    candidates = [
        Path(__file__).resolve().parent / "hg2_runtime",
        repo
        / "data"
        / "observations"
        / "daily_learning"
        / "aceguard_publish_repo"
        / "poker44"
        / "score"
        / "hg2_runtime",
    ]
    override = str(os.getenv("POKER44_HG2_RUNTIME_DIR") or "").strip()
    if override:
        candidates.insert(0, Path(override).expanduser())
    for candidate in candidates:
        if (candidate / "hg_features.py").exists():
            return candidate.resolve()
    raise FileNotFoundError("hg2 wide-feature runtime was not found")


def _load_wide_view():
    runtime_dir = _wide_runtime_dir()
    runtime_text = str(runtime_dir)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    for module_name in ("hg_features", "features_v2", "hg2_features_base"):
        module = sys.modules.get(module_name)
        loaded_from = str(getattr(module, "__file__", "")) if module is not None else ""
        if module is not None and loaded_from and runtime_text not in loaded_from:
            sys.modules.pop(module_name, None)
    from hg_features import wide_view

    return wide_view


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(model_path).open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"{model_path} did not contain a dict bundle")
    return bundle


def _feature_matrix(chunks: Sequence[Any], keys: list[str]) -> np.ndarray:
    wide_view = _load_wide_view()
    rows = [wide_view(chunk or []) for chunk in chunks]
    x = np.asarray(
        [[float(row.get(key, 0.0)) for key in keys] for row in rows],
        dtype=float,
    )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def score_chunks(chunks: Sequence[Any], bundle: dict[str, Any]) -> list[float]:
    if not chunks:
        return []
    keys = list(bundle.get("keys") or [])
    if not keys:
        raise ValueError("LambdaMART-wide bundle has no feature keys")
    x = _feature_matrix(chunks, keys)

    children = list(bundle.get("seed_children") or [])
    if children:
        ranked: list[np.ndarray] = []
        weights: list[float] = []
        for child in children:
            model = child.get("model")
            if model is None:
                raise ValueError("LambdaMART ensemble child has no model")
            raw = np.asarray(model.predict(x), dtype=float)
            ranked.append(rankdata(raw, method="average") / max(len(raw), 1))
            weights.append(float(child.get("weight", 1.0)))
        weight_arr = np.asarray(weights, dtype=float)
        if float(np.sum(weight_arr)) <= 1e-12:
            raise ValueError("LambdaMART ensemble has no positive child weight")
        scores = np.column_stack(ranked) @ (weight_arr / float(np.sum(weight_arr)))
        return [float(value) for value in scores]

    model = bundle.get("model")
    if model is None:
        raise ValueError("LambdaMART-wide bundle has no model or seed_children")
    return [float(value) for value in np.asarray(model.predict(x), dtype=float)]


def score_from_file(chunks: Sequence[Any], model_path: str | os.PathLike[str]) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path))
