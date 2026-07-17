"""Inference for the fixed v402 plus natural multisalt sequence ensemble."""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Load the LightGBM family before importing Torch-backed sequence modules.
from poker44.score.original_multiview_hash_bag_dense_inference import (
    load_bundle as load_v402_bundle,
    score_chunks as score_v402_chunks,
)


FAMILY = "original_v402_natural_multisalt_sequence_ensemble"
VIEW_WEIGHTS = (0.50, 0.50)
TOP_N = 8
SALT_TOKENS = tuple(
    f"aceguard-v394-multisalt-probe-b0-s{index}" for index in range(12)
)
LANES_PER_BLOCK = 4
MODEL_BATCH_SIZE = 32
TORCH_THREADS_DARWIN = 1
TORCH_THREADS_LINUX = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _child_path(model_path: Path, name: Any) -> Path:
    filename = str(name or "")
    if not filename or Path(filename).name != filename:
        raise ValueError("v403 child model filename is unsafe")
    return model_path.parent / filename


def _validate_wrapper(bundle: dict[str, Any]) -> np.ndarray:
    if not isinstance(bundle, dict) or str(bundle.get("family") or "") != FAMILY:
        raise ValueError("v403 family mismatch")
    weights = np.asarray(bundle.get("view_weights") or [], dtype=np.float64)
    if weights.shape != (2,) or np.any(weights < 0.0):
        raise ValueError("v403 weights are invalid")
    if not np.isfinite(weights).all() or not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("v403 weights must be finite and sum to one")
    if not np.allclose(weights, VIEW_WEIGHTS, rtol=0.0, atol=0.0):
        raise ValueError("v403 fixed 50/50 declaration changed")
    if int(bundle.get("top_n") or 0) != TOP_N:
        raise ValueError("v403 requires the fixed top8 operating head")
    if tuple(bundle.get("sequence_salt_tokens") or ()) != SALT_TOKENS:
        raise ValueError("v403 sequence salt declaration changed")
    if int(bundle.get("sequence_lanes_per_block") or 0) != LANES_PER_BLOCK:
        raise ValueError("v403 lane block declaration changed")
    if int(bundle.get("sequence_model_batch_size") or 0) != MODEL_BATCH_SIZE:
        raise ValueError("v403 model batch declaration changed")
    if int(bundle.get("sequence_torch_threads_darwin") or 0) != TORCH_THREADS_DARWIN:
        raise ValueError("v403 macOS Torch thread declaration changed")
    if int(bundle.get("sequence_torch_threads_linux") or 0) != TORCH_THREADS_LINUX:
        raise ValueError("v403 Linux Torch thread declaration changed")
    for prefix in ("v402", "v394"):
        if not str(bundle.get(f"{prefix}_model_file") or ""):
            raise ValueError(f"v403 wrapper has no {prefix} child filename")
        if len(str(bundle.get(f"{prefix}_model_sha256") or "")) != 64:
            raise ValueError(f"v403 wrapper has no valid {prefix} child hash")
    return weights


def load_bundle(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    wrapper_path = Path(model_path)
    with wrapper_path.open("rb") as handle:
        bundle = pickle.load(handle)
    weights = _validate_wrapper(bundle)
    child_paths = {
        prefix: _child_path(wrapper_path, bundle[f"{prefix}_model_file"])
        for prefix in ("v402", "v394")
    }
    for prefix, path in child_paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"v403 {prefix} child model is missing or symlinked")
        if sha256(path) != str(bundle[f"{prefix}_model_sha256"]):
            raise ValueError(f"v403 {prefix} child model hash changed")

    runtime_v402 = load_v402_bundle(child_paths["v402"])
    from poker44.score.original_policy_sequence_inference import load_bundle as load_v394_bundle
    import torch

    torch_threads = TORCH_THREADS_DARWIN if sys.platform == "darwin" else TORCH_THREADS_LINUX
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    runtime_v394 = load_v394_bundle(child_paths["v394"])
    bundle["_runtime_v402"] = runtime_v402
    bundle["_runtime_v394"] = runtime_v394
    bundle["_runtime_weights"] = weights
    return bundle


def combine_component_scores(
    v402_scores: Sequence[float],
    sequence_scores: Sequence[float],
    weights: Sequence[float],
) -> list[float]:
    left = np.asarray(v402_scores, dtype=np.float64)
    right = np.asarray(sequence_scores, dtype=np.float64)
    blend = np.asarray(weights, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("v403 component scores are not aligned")
    if blend.shape != (2,) or not np.isclose(float(blend.sum()), 1.0):
        raise ValueError("v403 component weights are invalid")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("v403 component scores must be finite")
    return [float(value) for value in blend[0] * left + blend[1] * right]


def component_scores(
    chunks: Sequence[Sequence[dict[str, Any]]], bundle: dict[str, Any]
) -> dict[str, Any]:
    weights = _validate_wrapper(bundle)
    runtime_v402 = bundle.get("_runtime_v402")
    runtime_v394 = bundle.get("_runtime_v394")
    if not isinstance(runtime_v402, dict) or not isinstance(runtime_v394, dict):
        raise ValueError("v403 wrapper must be loaded with load_bundle")
    if not chunks:
        return {"v402": [], "sequence": [], "combined": [], "sequence_lanes": 0}

    v402 = score_v402_chunks(chunks, runtime_v402)
    from poker44.score.natural_multisalt_sequence_inference import score_chunks_detailed

    salts = tuple(f"{token}\0".encode() for token in SALT_TOKENS)
    sequence = score_chunks_detailed(
        chunks,
        runtime_v394,
        salts=salts,
        lanes_per_block=LANES_PER_BLOCK,
        model_batch_size=MODEL_BATCH_SIZE,
    )
    sequence_scores = list(sequence["scores"])
    return {
        "v402": [float(value) for value in v402],
        "sequence": [float(value) for value in sequence_scores],
        "combined": combine_component_scores(v402, sequence_scores, weights),
        "sequence_lanes": int(sequence["lanes"]),
    }


def score_chunks(
    chunks: Sequence[Sequence[dict[str, Any]]], bundle: dict[str, Any]
) -> list[float]:
    return list(component_scores(chunks, bundle)["combined"])


def score_from_file(
    chunks: Sequence[Sequence[dict[str, Any]]], model_path: str | os.PathLike[str]
) -> list[float]:
    return score_chunks(chunks, load_bundle(model_path))
