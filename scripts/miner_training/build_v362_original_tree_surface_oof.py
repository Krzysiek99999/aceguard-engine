#!/usr/bin/env python3
"""Build the v362 natural-unit original tree-surface rolling-OOF challenger."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from poker44.score.original_tree_surface_features import chunk_features  # noqa: E402
from poker44.score.original_hash_multiscale_features import (  # noqa: E402
    chunk_features as hash_multiscale_chunk_features,
)
from poker44.score.original_hash_multiscale_inference import (  # noqa: E402
    FEATURE_EXTRACTOR as HASH_MULTISCALE_FEATURE_EXTRACTOR,
    VIEW_MODE as HASH_MULTISCALE_VIEW_MODE,
)
from poker44.score.original_hash_bag_features import (  # noqa: E402
    chunk_features as hash_bag_chunk_features,
)
from poker44.score.original_hash_bag_inference import (  # noqa: E402
    FEATURE_EXTRACTOR as HASH_BAG_FEATURE_EXTRACTOR,
    VIEW_MODE as HASH_BAG_VIEW_MODE,
)
from poker44.score.original_hero_augmented_tree_surface_features import (  # noqa: E402
    chunk_features as hero_augmented_chunk_features,
)
from poker44.score.original_tree_surface_inference import (  # noqa: E402
    BLEND_STRATEGY,
    FEATURE_EXTRACTOR,
    VIEW_MODE,
    blend_component_predictions,
    model_scores,
    percentile_rank,
)
from poker44.score.rank_cap_remap import rank_cap_remap  # noqa: E402
from poker44.score.scoring import reward as official_reward  # noqa: E402
from scripts.miner_training.benchmark_source_units import (  # noqa: E402
    SourceUnit,
    hand_overlap_audit,
    load_source_units,
    source_summary,
)


FROZEN_BEFORE_SOURCE_DATE = "2026-07-12"
EXACT_PREVALENCES = (0.02, 0.05, 0.10, 0.20)
DEFAULT_TOP_NS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)
REWARD_BATCH_SIZE = 100
PREDECLARED_LIVE_TOP_N = 8
HERO_AUGMENTED_FEATURE_EXTRACTOR = (
    "poker44.score.original_hero_augmented_tree_surface_features:chunk_features"
)
COMPONENT_NAMES = (
    "lgbm_contract_ranker",
    "extra_trees_motif_temporal",
    "hist_gradient_selected_all",
)


@dataclass(frozen=True)
class RollingFold:
    name: str
    train_through: str
    tune_date: str
    test_date: str


# Kept literal and identical to build_v361_natural_policy_sequence_oof.DEFAULT_FOLDS.
DEFAULT_FOLDS = (
    RollingFold("f1", "2026-07-05", "2026-07-06", "2026-07-07"),
    RollingFold("f2", "2026-07-06", "2026-07-07", "2026-07-08"),
    RollingFold("f3", "2026-07-07", "2026-07-08", "2026-07-09"),
    RollingFold("f4", "2026-07-08", "2026-07-09", "2026-07-10"),
    RollingFold("f5", "2026-07-09", "2026-07-10", "2026-07-11"),
)


@dataclass(frozen=True)
class SourceMatrix:
    x: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    splits: np.ndarray
    groups: np.ndarray
    source_keys: np.ndarray
    keys: list[str]


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 362
    lgb_estimators: int = 350
    extra_estimators: int = 500
    hgb_iterations: int = 260
    contract_max_features: int = 160
    motif_temporal_max_features: int = 256
    all_max_features: int = 384
    min_samples_leaf: int = 8
    workers: int = 1
    surface_profile: str = "full"


@dataclass(frozen=True)
class SurfaceDefinition:
    name: str
    model_kind: str
    prefixes: tuple[str, ...]
    max_features: int
    score_method: str


def surface_definitions(config: TrainingConfig) -> tuple[SurfaceDefinition, ...]:
    if config.surface_profile not in {
        "full",
        "no_temporal",
        "hero_augmented",
        "hash_multiscale_no_temporal",
        "hash_bag_no_temporal",
    }:
        raise ValueError(f"unsupported surface profile: {config.surface_profile}")
    if config.surface_profile == "hash_bag_no_temporal":
        contract_prefixes = ("contract__",)
        motif_prefixes = ("motif__", "redundancy__")
        all_prefixes = ("contract__", "motif__", "redundancy__")
    elif config.surface_profile == "hash_multiscale_no_temporal":
        stats = ("mean", "std", "min", "max", "range")
        contract_prefixes = ("full__contract__",) + tuple(
            f"bucket__{stat}__contract__" for stat in stats
        )
        motif_prefixes = ("full__motif__", "full__redundancy__") + tuple(
            prefix
            for stat in stats
            for prefix in (
                f"bucket__{stat}__motif__",
                f"bucket__{stat}__redundancy__",
            )
        )
        all_prefixes = ("full__", "bucket__")
    elif config.surface_profile == "no_temporal":
        contract_prefixes = ("contract__",)
        motif_prefixes = ("motif__",)
        all_prefixes = ("contract__", "motif__")
    elif config.surface_profile == "hero_augmented":
        contract_prefixes = (
            "contract__",
            "hero__policy__",
            "hero__token__",
            "hero__signature__",
            "hero__sequence__",
        )
        motif_prefixes = (
            "motif__",
            "temporal__",
            "hero__token__",
            "hero__sequence__",
            "hero__amount__",
        )
        all_prefixes = ("contract__", "motif__", "temporal__", "hero__")
    else:
        contract_prefixes = ("contract__",)
        motif_prefixes = ("motif__", "temporal__")
        all_prefixes = ("contract__", "motif__", "temporal__")
    return (
        SurfaceDefinition(
            COMPONENT_NAMES[0],
            "lgbm_ranker",
            contract_prefixes,
            int(config.contract_max_features),
            "predict",
        ),
        SurfaceDefinition(
            COMPONENT_NAMES[1],
            "extra_trees",
            motif_prefixes,
            int(config.motif_temporal_max_features),
            "predict_proba",
        ),
        SurfaceDefinition(
            COMPONENT_NAMES[2],
            "hist_gradient",
            all_prefixes,
            int(config.all_max_features),
            "predict_proba",
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_key_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(value) for value in values)).encode()).hexdigest()


def assert_source_disjoint(**partitions: Sequence[SourceUnit]) -> None:
    """Reject any SourceUnit appearing in more than one named partition."""
    owner: dict[str, str] = {}
    for partition_name, units in partitions.items():
        for unit in units:
            previous = owner.get(unit.source_key)
            if previous is not None:
                raise ValueError(
                    f"source overlap: {unit.source_key} appears in {previous} and {partition_name}"
                )
            owner[unit.source_key] = partition_name


def fold_units(
    units: Sequence[SourceUnit], fold: RollingFold
) -> tuple[list[SourceUnit], list[SourceUnit], list[SourceUnit]]:
    """Return one v361-compatible rolling fold using only split=train units."""
    if not (fold.train_through < fold.tune_date < fold.test_date):
        raise ValueError(f"non-temporal rolling fold: {fold}")
    eligible = [unit for unit in units if unit.split == "train"]
    train = [unit for unit in eligible if unit.source_date <= fold.train_through]
    tune = [unit for unit in eligible if unit.source_date == fold.tune_date]
    test = [unit for unit in eligible if unit.source_date == fold.test_date]
    if not train or not tune or not test:
        raise ValueError(
            f"empty fold {fold.name}: train={len(train)} tune={len(tune)} test={len(test)}"
        )
    assert_source_disjoint(train=train, tune=tune, test=test)
    return train, tune, test


def benchmark_input_manifest(
    benchmark_dir: Path, units: Sequence[SourceUnit]
) -> dict[str, Any]:
    """Freeze file and sanitized natural-unit hashes for reproducibility."""
    files: list[dict[str, Any]] = []
    for path in sorted(Path(benchmark_dir).glob("chunks_*.json")):
        data = path.read_bytes()
        files.append(
            {
                "path": str(path),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    payload_owners: dict[str, list[str]] = defaultdict(list)
    payload_hashes: dict[str, str] = {}
    for unit in units:
        canonical = json.dumps(unit.chunk, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        payload_hashes[unit.source_key] = digest
        payload_owners[digest].append(unit.source_key)
    repeated = {key: value for key, value in payload_owners.items() if len(value) > 1}
    if repeated:
        raise ValueError(f"duplicate sanitized natural payloads: {len(repeated)}")
    hand_audit = hand_overlap_audit(list(units))
    if hand_audit["cross_date_hashes"]:
        raise ValueError(
            f"model-view hands cross temporal dates: {hand_audit['cross_date_hashes']}"
        )
    return {
        "benchmark_dir": str(benchmark_dir),
        "input_contract": "canonical_api_miner_visible_no_resanitization",
        "files": files,
        "source_units": len(units),
        "source_key_hash": _source_key_hash([unit.source_key for unit in units]),
        "sanitized_payload_hash": hashlib.sha256(
            "\n".join(
                f"{key}:{payload_hashes[key]}" for key in sorted(payload_hashes)
            ).encode()
        ).hexdigest(),
        "duplicate_sanitized_payloads": 0,
        "hand_overlap_audit": hand_audit,
    }


def build_matrix(
    units: Sequence[SourceUnit],
    *,
    extractor: Callable[[list[dict[str, Any]]], dict[str, float]] = chunk_features,
) -> SourceMatrix:
    """Extract exactly one feature row per natural SourceUnit."""
    rows = [extractor(unit.chunk) for unit in units]
    keys = sorted({key for row in rows for key in row})
    forbidden_tokens = (
        "source_key",
        "source_date",
        "group_id",
        "chunkhash",
        "chunkid",
        "player_uid",
        "hand_id",
        "hole_card",
        "outcome",
    )
    leaked = [key for key in keys if any(token in key.lower() for token in forbidden_tokens)]
    if leaked:
        raise RuntimeError(f"private or provenance fields leaked into features: {leaked[:10]}")
    x = np.asarray(
        [[float(row.get(key, 0.0)) for key in keys] for row in rows],
        dtype=np.float32,
    )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return SourceMatrix(
        x=x,
        y=np.asarray([unit.label for unit in units], dtype=np.int8),
        dates=np.asarray([unit.source_date for unit in units]),
        splits=np.asarray([unit.split for unit in units]),
        groups=np.asarray([unit.ranking_group for unit in units]),
        source_keys=np.asarray([unit.source_key for unit in units]),
        keys=keys,
    )


def select_feature_indices(
    matrix: SourceMatrix,
    fit_indices: Sequence[int] | np.ndarray,
    *,
    prefixes: Sequence[str],
    max_features: int,
) -> np.ndarray:
    """Select non-constant features from fit rows only by standardized label gap."""
    indices = np.asarray(fit_indices, dtype=int)
    if indices.ndim != 1 or indices.size == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("feature selection requires unique, non-empty fit indices")
    labels = matrix.y[indices]
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("feature selection requires both classes in fit rows")
    candidates = np.asarray(
        [
            index
            for index, key in enumerate(matrix.keys)
            if any(key.startswith(prefix) for prefix in prefixes)
        ],
        dtype=int,
    )
    if candidates.size == 0:
        raise ValueError(f"no features found for prefixes {tuple(prefixes)}")

    values = np.asarray(matrix.x[indices][:, candidates], dtype=np.float64)
    variance = np.var(values, axis=0)
    positive_mean = np.mean(values[labels == 1], axis=0)
    negative_mean = np.mean(values[labels == 0], axis=0)
    effect = np.abs(positive_mean - negative_mean) / (np.sqrt(variance) + 1e-9)
    eligible = [
        position
        for position in range(len(candidates))
        if variance[position] > 1e-12 and np.isfinite(effect[position])
    ]
    if not eligible:
        raise ValueError(f"all fit features are constant for prefixes {tuple(prefixes)}")
    eligible.sort(
        key=lambda position: (
            -float(effect[position]),
            matrix.keys[int(candidates[position])],
        )
    )
    count = min(max(1, int(max_features)), len(eligible))
    return np.asarray([int(candidates[position]) for position in eligible[:count]], dtype=int)


def group_order_and_sizes(group_ids: np.ndarray) -> tuple[np.ndarray, list[int]]:
    group_ids = np.asarray(group_ids).astype(str)
    order = np.argsort(group_ids, kind="mergesort")
    ordered = group_ids[order]
    sizes: list[int] = []
    start = 0
    for index in range(1, len(ordered) + 1):
        if index == len(ordered) or ordered[index] != ordered[start]:
            sizes.append(index - start)
            start = index
    if sum(sizes) != len(group_ids):
        raise RuntimeError("invalid LambdaRank group packing")
    return order, sizes


def _balanced_sample_weight(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    counts = {value: int(np.sum(labels == value)) for value in (0, 1)}
    if min(counts.values()) <= 0:
        raise ValueError("balanced weights require both classes")
    return np.asarray(
        [len(labels) / (2.0 * counts[int(value)]) for value in labels],
        dtype=float,
    )


def _fit_surface_model(
    definition: SurfaceDefinition,
    x: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    config: TrainingConfig,
) -> Any:
    min_leaf = max(2, int(config.min_samples_leaf))
    if definition.model_kind == "lgbm_ranker":
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
            raise RuntimeError("v362 training requires lightgbm") from exc

        order, group_sizes = group_order_and_sizes(group_ids)
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=[0, 1],
            n_estimators=int(config.lgb_estimators),
            learning_rate=0.025,
            num_leaves=15,
            max_depth=5,
            min_child_samples=min_leaf,
            min_split_gain=0.01,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.50,
            reg_lambda=2.00,
            max_bin=127,
            random_state=int(config.seed),
            bagging_seed=int(config.seed) + 11,
            feature_fraction_seed=int(config.seed) + 12,
            data_random_seed=int(config.seed) + 13,
            deterministic=True,
            force_col_wise=True,
            n_jobs=max(1, int(config.workers)),
            verbosity=-1,
        )
        weights = _balanced_sample_weight(y)
        model.fit(
            x[order],
            y[order],
            group=group_sizes,
            sample_weight=weights[order],
            eval_at=[1, 2, 5, 10, 20],
        )
        return model

    if definition.model_kind == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=int(config.extra_estimators),
            max_depth=12,
            min_samples_leaf=min_leaf,
            max_features=0.55,
            class_weight="balanced",
            random_state=int(config.seed) + 1000,
            n_jobs=max(1, int(config.workers)),
        )
        model.fit(x, y)
        return model

    if definition.model_kind == "hist_gradient":
        model = HistGradientBoostingClassifier(
            max_iter=int(config.hgb_iterations),
            learning_rate=0.035,
            max_leaf_nodes=15,
            max_depth=5,
            min_samples_leaf=min_leaf,
            l2_regularization=2.00,
            max_bins=127,
            early_stopping=False,
            random_state=int(config.seed) + 2000,
        )
        model.fit(x, y, sample_weight=_balanced_sample_weight(y))
        return model
    raise ValueError(f"unsupported surface model kind: {definition.model_kind}")


def fit_surface_components(
    matrix: SourceMatrix,
    fit_indices: Sequence[int] | np.ndarray,
    config: TrainingConfig,
) -> list[dict[str, Any]]:
    """Fit the three fixed surfaces with selectors fitted only on fit_indices."""
    indices = np.asarray(fit_indices, dtype=int)
    components: list[dict[str, Any]] = []
    for definition in surface_definitions(config):
        selected = select_feature_indices(
            matrix,
            indices,
            prefixes=definition.prefixes,
            max_features=definition.max_features,
        )
        keys = [matrix.keys[int(index)] for index in selected]
        model = _fit_surface_model(
            definition,
            matrix.x[indices][:, selected],
            matrix.y[indices],
            matrix.groups[indices],
            config,
        )
        components.append(
            {
                "name": definition.name,
                "model_kind": definition.model_kind,
                "model": model,
                "keys": keys,
                "score_method": definition.score_method,
                "feature_selection": {
                    "method": "fit_rows_standardized_label_gap",
                    "fit_rows": int(len(indices)),
                    "fit_source_key_hash": _source_key_hash(matrix.source_keys[indices]),
                    "candidate_prefixes": list(definition.prefixes),
                    "max_features": int(definition.max_features),
                    "selected_features": int(len(keys)),
                },
            }
        )
    if tuple(component["name"] for component in components) != COMPONENT_NAMES:
        raise RuntimeError("tree-surface component contract changed")
    return components


def predict_components(
    components: Sequence[dict[str, Any]],
    matrix: SourceMatrix,
    row_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Return raw component predictions with shape (components, rows)."""
    rows = np.asarray(row_indices, dtype=int)
    key_to_index = {key: index for index, key in enumerate(matrix.keys)}
    predictions: list[np.ndarray] = []
    for component in components:
        try:
            columns = np.asarray(
                [key_to_index[str(key)] for key in component["keys"]], dtype=int
            )
        except KeyError as exc:
            raise ValueError(f"component feature missing from matrix: {exc}") from exc
        values = model_scores(
            component["model"],
            matrix.x[rows][:, columns],
            str(component["score_method"]),
        )
        predictions.append(values)
    return np.vstack(predictions)


def prevalence_batches(
    labels: np.ndarray,
    prevalences: Sequence[float],
    *,
    samples: int,
    batch_size: int,
    seed: int,
) -> dict[float, list[np.ndarray]]:
    """Sample exact-composition batches without replacing a natural SourceUnit."""
    positives = np.flatnonzero(np.asarray(labels) == 1).tolist()
    negatives = np.flatnonzero(np.asarray(labels) == 0).tolist()
    rng = random.Random(int(seed))
    result: dict[float, list[np.ndarray]] = {}
    for prevalence in prevalences:
        bot_count = max(1, min(batch_size - 1, int(round(float(prevalence) * batch_size))))
        human_count = int(batch_size) - bot_count
        if bot_count > len(positives) or human_count > len(negatives):
            continue
        sampled: list[np.ndarray] = []
        for _ in range(max(1, int(samples))):
            indices = rng.sample(positives, bot_count) + rng.sample(negatives, human_count)
            rng.shuffle(indices)
            sampled.append(np.asarray(indices, dtype=int))
        result[float(prevalence)] = sampled
    return result


def require_exact_prevalences(batches: dict[float, list[np.ndarray]]) -> None:
    missing = sorted(set(EXACT_PREVALENCES) - set(batches))
    if missing:
        raise ValueError(f"not enough natural units for exact batch100 prevalences: {missing}")
    for prevalence in EXACT_PREVALENCES:
        expected = int(round(prevalence * REWARD_BATCH_SIZE))
        if not batches[prevalence]:
            raise ValueError(f"no sampled batches for prevalence {prevalence}")
        if any(len(indices) != REWARD_BATCH_SIZE for indices in batches[prevalence]):
            raise ValueError("reward sampling produced a non-batch100 sample")
        if expected not in {2, 5, 10, 20}:
            raise RuntimeError("exact prevalence contract changed")


def _selection_objective(result: dict[str, Any]) -> float:
    rows = [
        result["by_prevalence"][f"{prevalence:.4f}"]
        for prevalence in EXACT_PREVALENCES
        if f"{prevalence:.4f}" in result.get("by_prevalence", {})
    ]
    if not rows:
        return float("-inf")
    return float(
        np.mean([row["reward_mean"] for row in rows])
        + 0.20 * np.mean([row["reward_p10"] for row in rows])
        - 0.50 * np.mean([row["hard_zero_rate"] for row in rows])
    )


def evaluate_component_predictions(
    component_predictions: np.ndarray,
    labels: np.ndarray,
    batches: dict[float, list[np.ndarray]],
    top_n: int,
) -> dict[str, Any]:
    """Apply the live batch-rank blend and the exact current reward contract."""
    components = np.asarray(component_predictions, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)
    if components.ndim != 2 or components.shape[1] != len(labels):
        raise ValueError("component predictions and labels have incompatible shapes")
    by_prevalence: dict[str, Any] = {}
    for prevalence, sampled in batches.items():
        rows: list[dict[str, float]] = []
        for indices in sampled:
            blended = blend_component_predictions(components[:, indices])
            served = np.asarray(
                rank_cap_remap(blended, min(int(top_n), len(indices))), dtype=float
            )
            reward_value, metrics = official_reward(served, labels[indices])
            rows.append(
                {
                    "reward": float(reward_value),
                    "ap": float(metrics.get("ap_score", 0.0)),
                    "recall": float(metrics.get("bot_recall", 0.0)),
                    "hard_zero": float(float(reward_value) <= 0.0),
                }
            )
        rewards = np.asarray([row["reward"] for row in rows], dtype=float)
        by_prevalence[f"{prevalence:.4f}"] = {
            "batches": int(len(rows)),
            "reward_mean": float(np.mean(rewards)),
            "reward_p10": float(np.quantile(rewards, 0.10)),
            "reward_min": float(np.min(rewards)),
            "hard_zero_rate": float(np.mean([row["hard_zero"] for row in rows])),
            "ap_mean": float(np.mean([row["ap"] for row in rows])),
            "recall_at_fpr05_mean": float(np.mean([row["recall"] for row in rows])),
        }
    result = {
        "top_n": int(top_n),
        "blend_strategy": BLEND_STRATEGY,
        "reward_contract": "poker44.score.scoring:reward",
        "by_prevalence": by_prevalence,
    }
    result["selection_objective"] = _selection_objective(result)
    return result


def operating_point_sweep(
    component_predictions: np.ndarray,
    labels: np.ndarray,
    batches: dict[float, list[np.ndarray]],
    top_ns: Sequence[int],
) -> list[dict[str, Any]]:
    rows = [
        evaluate_component_predictions(component_predictions, labels, batches, int(top_n))
        for top_n in top_ns
    ]
    return sorted(
        rows,
        key=lambda row: (float(row["selection_objective"]), -int(row["top_n"])),
        reverse=True,
    )


def date_local_operating_point_sweep(
    fold_safe_scores: np.ndarray,
    labels: np.ndarray,
    source_dates: Sequence[str],
    top_ns: Sequence[int],
) -> list[dict[str, Any]]:
    """Evaluate each natural OOF date independently without cross-date sampling."""
    scores = np.asarray(fold_safe_scores, dtype=float)
    targets = np.asarray(labels, dtype=np.int8)
    dates = np.asarray([str(value) for value in source_dates])
    if not (scores.shape == targets.shape == dates.shape):
        raise ValueError("date-local operating-point arrays are not aligned")
    rows: list[dict[str, Any]] = []
    for top_n in top_ns:
        by_date: dict[str, Any] = {}
        rewards: list[float] = []
        for source_date in sorted(set(dates.tolist())):
            mask = dates == source_date
            served = np.asarray(
                rank_cap_remap(scores[mask], min(int(top_n), int(np.sum(mask)))),
                dtype=float,
            )
            reward_value, metrics = official_reward(served, targets[mask])
            rewards.append(float(reward_value))
            by_date[source_date] = {
                "rows": int(np.sum(mask)),
                "reward": float(reward_value),
                "ap": float(metrics["ap_score"]),
                "recall_at_fpr05": float(metrics["bot_recall"]),
                "hard_fpr": float(metrics["hard_fpr"]),
                "hard_bot_recall": float(metrics["hard_bot_recall"]),
                "threshold_sanity_quality": float(metrics["threshold_sanity_quality"]),
                "hard_zero": bool(float(reward_value) <= 0.0),
            }
        rows.append(
            {
                "top_n": int(top_n),
                "role": "date_local_diagnostic_not_head_selection",
                "by_date": by_date,
                "reward_mean": float(np.mean(rewards)),
                "reward_min": float(np.min(rewards)),
                "hard_zero_dates": int(
                    sum(bool(row["hard_zero"]) for row in by_date.values())
                ),
            }
        )
    return rows


def continuous_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    result: dict[str, float | int] = {"rows": int(len(labels))}
    if len(labels) == 0 or set(labels.tolist()) != {0, 1}:
        result.update({"ap": 0.0, "auc": 0.0, "recall_at_fpr05": 0.0})
        return result
    ap = float(average_precision_score(labels, scores))
    auc = float(roc_auc_score(labels, scores))
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    recall = np.cumsum(ordered_labels == 1) / positives
    fpr = np.cumsum(ordered_labels == 0) / negatives
    allowed = fpr <= 0.05
    recall_at_fpr = float(np.max(recall[allowed])) if np.any(allowed) else 0.0
    result.update({"ap": ap, "auc": auc, "recall_at_fpr05": recall_at_fpr})
    return result


def prediction_report(
    components: Sequence[dict[str, Any]],
    raw_predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    return {
        "blend": continuous_metrics(labels, blend_component_predictions(raw_predictions)),
        "components": {
            str(component["name"]): continuous_metrics(labels, raw_predictions[index])
            for index, component in enumerate(components)
        },
    }


def fold_safe_component_percentiles(
    component_predictions: np.ndarray,
    fold_names: Sequence[str],
) -> np.ndarray:
    """Normalize each native model only inside the fold that produced it."""
    values = np.asarray(component_predictions, dtype=float)
    folds = np.asarray(fold_names)
    if values.ndim != 2 or values.shape[1] != len(folds):
        raise ValueError("fold-safe component normalization received invalid shapes")
    output = np.empty_like(values, dtype=float)
    for fold in sorted(set(str(value) for value in folds.tolist())):
        indices = np.flatnonzero(folds == fold)
        if indices.size == 0:
            raise ValueError(f"fold {fold} has no predictions")
        for component_index in range(values.shape[0]):
            output[component_index, indices] = percentile_rank(
                values[component_index, indices]
            )
    return output


def ranked_prediction_report(
    component_names: Sequence[str],
    ranked_components: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    blend = np.mean(np.asarray(ranked_components, dtype=float), axis=0)
    return {
        "blend": continuous_metrics(labels, blend),
        "components": {
            str(name): continuous_metrics(labels, ranked_components[index])
            for index, name in enumerate(component_names)
        },
    }


def component_fit_summary(
    components: Sequence[dict[str, Any]], *, include_keys: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in components:
        keys = [str(key) for key in component["keys"]]
        row = {
            "name": str(component["name"]),
            "model_kind": str(component["model_kind"]),
            "model_class": type(component["model"]).__name__,
            "score_method": str(component["score_method"]),
            "feature_count": int(len(keys)),
            "feature_key_hash": _source_key_hash(keys),
            "feature_selection": dict(component["feature_selection"]),
        }
        if include_keys:
            row["keys"] = keys
        rows.append(row)
    return rows


def _indices_for_units(
    units: Sequence[SourceUnit], key_to_index: dict[str, int]
) -> np.ndarray:
    return np.asarray([key_to_index[unit.source_key] for unit in units], dtype=int)


def training_data_statement(training_through_date: str) -> str:
    return (
        "Original AceGuard natural-unit tree-surface challenger trained from scratch only on "
        "official split=train public Poker44 canonical miner-visible API SourceUnits without "
        "reapplying the non-idempotent validator sanitizer. One feature row is one natural "
        "SourceUnit; players, chunks and source "
        "units are never concatenated. The fixed ensemble contains three independently regularized "
        "surfaces: a LightGBM LambdaRank contract surface, an ExtraTrees motif/temporal surface and "
        "a HistGradientBoosting selected surface, with included prefixes recorded by the explicit "
        "surface profile. Feature selection was fitted independently "
        "using only each rolling fold's train rows. Folds f1-f5 exactly match v361; every OOF test "
        "prediction came from models fitted only through the fold's earlier train date. Fold tune "
        "dates are monitoring-only and do not select features, models or top-N. Top-N was selected "
        "once on pooled natural OOF component predictions using per-batch component percentile-rank "
        "mean, exact batch100 prevalences 2/5/10/20 and the current Poker44 reward. Official "
        "split=validation was excluded from fitting, feature selection and top-N selection and is "
        "reported only as a non-blind regression because repository experiments had already inspected "
        "that split. The frozen components were refitted on split=train through "
        f"{training_through_date}. Private live payloads, identities, cards, outcomes, wallets, IPs, "
        "pseudo-labels, competitor code and competitor weights were not used. Frozen before sourceDate "
        f"{FROZEN_BEFORE_SOURCE_DATE}; a future sourceDate blind is still required before deployment."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default=str(REPO / "data/benchmark_v112"))
    parser.add_argument("--first-date", default="2026-05-26")
    parser.add_argument("--last-date", default="2026-07-11")
    parser.add_argument("--seed", type=int, default=362)
    parser.add_argument("--lgb-estimators", type=int, default=350)
    parser.add_argument("--extra-estimators", type=int, default=500)
    parser.add_argument("--hgb-iterations", type=int, default=260)
    parser.add_argument("--contract-max-features", type=int, default=160)
    parser.add_argument("--motif-temporal-max-features", type=int, default=256)
    parser.add_argument("--all-max-features", type=int, default=384)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument(
        "--surface-profile",
        choices=(
            "full",
            "no_temporal",
            "hero_augmented",
            "hash_multiscale_no_temporal",
            "hash_bag_no_temporal",
        ),
        default="full",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, int(os.cpu_count() or 1)))
    )
    parser.add_argument("--top-n", nargs="+", type=int, default=list(DEFAULT_TOP_NS))
    parser.add_argument("--oof-samples", type=int, default=1000)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--family", default="v362_original_tree_surface_oof")
    parser.add_argument(
        "--out-dir",
        default=str(REPO / "data/models/v362_original_tree_surface_oof"),
    )
    args = parser.parse_args()

    config = TrainingConfig(
        seed=int(args.seed),
        lgb_estimators=int(args.lgb_estimators),
        extra_estimators=int(args.extra_estimators),
        hgb_iterations=int(args.hgb_iterations),
        contract_max_features=int(args.contract_max_features),
        motif_temporal_max_features=int(args.motif_temporal_max_features),
        all_max_features=int(args.all_max_features),
        min_samples_leaf=int(args.min_samples_leaf),
        workers=max(1, int(args.workers)),
        surface_profile=str(args.surface_profile),
    )
    benchmark_dir = Path(args.benchmark_dir)
    units = load_source_units(
        benchmark_dir,
        first_date=str(args.first_date),
        last_date=str(args.last_date),
        sanitize=False,
    )
    if not units:
        raise ValueError("no natural SourceUnits loaded")
    if any(unit.source_date >= FROZEN_BEFORE_SOURCE_DATE for unit in units):
        raise ValueError(
            f"input contains sourceDate at or after frozen boundary {FROZEN_BEFORE_SOURCE_DATE}"
        )
    manifest = benchmark_input_manifest(benchmark_dir, units)
    if config.surface_profile == "hero_augmented":
        extractor = hero_augmented_chunk_features
        feature_extractor_name = HERO_AUGMENTED_FEATURE_EXTRACTOR
        view_mode_name = VIEW_MODE
    elif config.surface_profile == "hash_multiscale_no_temporal":
        extractor = hash_multiscale_chunk_features
        feature_extractor_name = HASH_MULTISCALE_FEATURE_EXTRACTOR
        view_mode_name = HASH_MULTISCALE_VIEW_MODE
    elif config.surface_profile == "hash_bag_no_temporal":
        extractor = hash_bag_chunk_features
        feature_extractor_name = HASH_BAG_FEATURE_EXTRACTOR
        view_mode_name = HASH_BAG_VIEW_MODE
    else:
        extractor = chunk_features
        feature_extractor_name = FEATURE_EXTRACTOR
        view_mode_name = VIEW_MODE
    view_score_order = (
        "one_hash_multiscale_row_per_chunk_then_batch_percentile_rank_mean"
        if config.surface_profile in {"hash_multiscale_no_temporal", "hash_bag_no_temporal"}
        else "reduce_per_component_then_batch_percentile_rank_mean"
    )
    matrix = build_matrix(units, extractor=extractor)
    key_to_index = {unit.source_key: index for index, unit in enumerate(units)}

    fold_reports: list[dict[str, Any]] = []
    oof_component_rows: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []
    oof_source_keys: list[str] = []
    oof_source_dates: list[str] = []
    oof_fold_names: list[str] = []
    seen_test_keys: set[str] = set()
    for fold in DEFAULT_FOLDS:
        train_units, tune_units, test_units = fold_units(units, fold)
        test_keys = {unit.source_key for unit in test_units}
        overlap = seen_test_keys & test_keys
        if overlap:
            raise ValueError(f"OOF test source repeated across folds: {sorted(overlap)[:3]}")
        seen_test_keys |= test_keys
        train_indices = _indices_for_units(train_units, key_to_index)
        tune_indices = _indices_for_units(tune_units, key_to_index)
        test_indices = _indices_for_units(test_units, key_to_index)
        print(
            f"{fold.name}: fit={len(train_indices)} tune={len(tune_indices)} "
            f"test={len(test_indices)}",
            flush=True,
        )
        components = fit_surface_components(matrix, train_indices, config)
        tune_raw = predict_components(components, matrix, tune_indices)
        test_raw = predict_components(components, matrix, test_indices)
        tune_labels = matrix.y[tune_indices]
        test_labels = matrix.y[test_indices]
        fold_reports.append(
            {
                "fold": asdict(fold),
                "train_summary": source_summary(train_units),
                "tune_summary": source_summary(tune_units),
                "test_summary": source_summary(test_units),
                "feature_selection_fit_partition": "fold_train_only",
                "tune_role": "monitoring_only_not_selection",
                "components": component_fit_summary(components, include_keys=False),
                "tune_predictions": prediction_report(components, tune_raw, tune_labels),
                "test_predictions": prediction_report(components, test_raw, test_labels),
            }
        )
        oof_component_rows.append(test_raw)
        oof_labels.append(test_labels)
        oof_source_keys.extend(str(value) for value in matrix.source_keys[test_indices])
        oof_source_dates.extend(str(value) for value in matrix.dates[test_indices])
        oof_fold_names.extend([fold.name] * len(test_indices))

    pooled_components = np.concatenate(oof_component_rows, axis=1)
    pooled_labels = np.concatenate(oof_labels)
    if pooled_components.shape != (len(COMPONENT_NAMES), len(pooled_labels)):
        raise RuntimeError("pooled OOF component prediction shape is invalid")
    if len(oof_source_keys) != len(set(oof_source_keys)):
        raise RuntimeError("pooled OOF contains duplicate natural SourceUnits")
    fold_safe_components = fold_safe_component_percentiles(
        pooled_components, oof_fold_names
    )
    fold_safe_blend = np.mean(fold_safe_components, axis=0)
    operating_points = date_local_operating_point_sweep(
        fold_safe_blend,
        pooled_labels,
        oof_source_dates,
        list(args.top_n),
    )
    selected_top_n = PREDECLARED_LIVE_TOP_N

    final_train = [unit for unit in units if unit.split == "train"]
    official_validation = [unit for unit in units if unit.split == "validation"]
    if not final_train or not official_validation:
        raise ValueError("final split=train or official split=validation is empty")
    assert_source_disjoint(final_train=final_train, official_validation=official_validation)
    final_indices = _indices_for_units(final_train, key_to_index)
    validation_indices = _indices_for_units(official_validation, key_to_index)
    final_components = fit_surface_components(matrix, final_indices, config)
    validation_components = predict_components(final_components, matrix, validation_indices)
    validation_labels = matrix.y[validation_indices]
    validation_blend = np.empty(len(validation_labels), dtype=float)
    validation_dates = np.asarray(matrix.dates[validation_indices]).astype(str)
    for source_date in sorted(set(validation_dates.tolist())):
        mask = validation_dates == source_date
        validation_blend[mask] = blend_component_predictions(validation_components[:, mask])
    validation_reward = date_local_operating_point_sweep(
        validation_blend,
        validation_labels,
        validation_dates,
        [selected_top_n],
    )

    training_through_date = max(unit.source_date for unit in final_train)
    statement = training_data_statement(training_through_date)
    statement += (
        f" Surface profile: {config.surface_profile}; "
        + {
            "no_temporal": "the diagnostic no-temporal profile excludes every temporal__ feature.",
            "full": "the full profile includes contract__, motif__ and temporal__ features.",
            "hero_augmented": (
                "the hero-augmented profile adds an independently implemented HERO-only conditional "
                "policy, motif, amount and cross-hand strategy surface."
            ),
            "hash_multiscale_no_temporal": (
                "the label-blind hash-multiscale profile erases source hand order, preserves one "
                "full canonical hand bag and mean/std/min/max/range across three disjoint subsets, "
                "and excludes every temporal__ feature."
            ),
            "hash_bag_no_temporal": (
                "the full hash-bag profile erases source hand order, caps the canonical bag at "
                "60 hands, adds exact/near redundancy and conditional-policy features, and "
                "excludes every temporal__ feature."
            ),
        }[config.surface_profile]
    )
    statement += (
        f" The live batch100 head is predeclared as top{PREDECLARED_LIVE_TOP_N}; "
        "OOF source dates are never mixed to manufacture prevalence stress, and "
        "date-local served-head rewards are diagnostic only."
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    family = str(args.family)
    version = f"{family}_{stamp}"
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": 1,
        "family": family,
        "version": version,
        "feature_extractor": feature_extractor_name,
        "components": final_components,
        "component_names": list(COMPONENT_NAMES),
        "blend_strategy": BLEND_STRATEGY,
        "view_mode": view_mode_name,
        "view_score_order": view_score_order,
        "top_n": selected_top_n,
        "training_split": "train",
        "training_through_date": training_through_date,
        "frozen_before_source_date": FROZEN_BEFORE_SOURCE_DATE,
        "training_data_statement": statement,
        "surface_profile": config.surface_profile,
    }
    model_path = out_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    oof_path = out_dir / "oof_component_predictions.npz"
    np.savez_compressed(
        oof_path,
        source_keys=np.asarray(oof_source_keys),
        source_dates=np.asarray(oof_source_dates),
        fold_names=np.asarray(oof_fold_names),
        labels=pooled_labels.astype(np.int8),
        component_names=np.asarray(COMPONENT_NAMES),
        component_predictions=pooled_components.T.astype(np.float64),
        descriptive_pooled_blend=blend_component_predictions(pooled_components),
        fold_safe_component_percentiles=fold_safe_components.T.astype(np.float64),
        fold_safe_blend=fold_safe_blend.astype(np.float64),
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "family": family,
        "version": version,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "oof_predictions_path": str(oof_path),
        "oof_predictions_sha256": sha256_file(oof_path),
        "benchmark_manifest": manifest,
        "source_summary": source_summary(list(units)),
        "fold_contract": [asdict(fold) for fold in DEFAULT_FOLDS],
        "training_config": asdict(config),
        "training_representation": "one_feature_row_per_natural_source_unit_no_player_join",
        "feature_extractor": feature_extractor_name,
        "blend_strategy": BLEND_STRATEGY,
        "view_mode": view_mode_name,
        "view_score_order": view_score_order,
        "folds": fold_reports,
        "oof_source_units": int(len(pooled_labels)),
        "oof_source_key_hash": _source_key_hash(oof_source_keys),
        "oof_predictions": ranked_prediction_report(
            COMPONENT_NAMES, fold_safe_components, pooled_labels
        ),
        "oof_raw_cross_fold_scale_diagnostic": prediction_report(
            final_components, pooled_components, pooled_labels
        ),
        "oof_score_space": "within-fold average-tie component percentiles",
        "oof_operating_points": operating_points,
        "oof_operating_points_role": "date_local_diagnostic_not_head_selection",
        "selected_top_n": selected_top_n,
        "selected_top_n_role": "predeclared_live_batch100_head",
        "selection_partition": "none_predeclared_before_future_blind",
        "reward_contract": "poker44.score.scoring:reward",
        "reward_batch_size": REWARD_BATCH_SIZE,
        "reward_prevalences": "not_manufactured_from_cross_date_oof",
        "final_train_summary": source_summary(final_train),
        "final_components": component_fit_summary(final_components, include_keys=True),
        "official_validation": {
            "role": "non_blind_regression_only",
            "used_for_fit_feature_selection_or_top_n": False,
            "summary": source_summary(official_validation),
            "predictions": prediction_report(
                final_components, validation_components, validation_labels
            ),
            "reward": validation_reward,
        },
        "training_data_statement": statement,
        "evidence_status": "rolling_oof_plus_nonblind_validation_regression_not_future_blind",
        "frozen_before_source_date": FROZEN_BEFORE_SOURCE_DATE,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (Path(args.out_dir) / "latest_report.json").write_text(report_path.read_text())
    print(
        json.dumps(
            {
                "model": str(model_path),
                "model_sha256": report["model_sha256"],
                "oof_predictions": str(oof_path),
                "selected_top_n": selected_top_n,
                "oof": report["oof_predictions"],
                "official_validation": report["official_validation"],
                "report": str(report_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
