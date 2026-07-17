"""Helpers for publishing and validating Poker44 miner model manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

MIN_REQUIRED_MANIFEST_FIELDS = [
    "open_source",
    "repo_url",
    "repo_commit",
    "model_name",
    "model_version",
    "training_data_statement",
    "private_data_attestation",
]
REFERENCE_MINER_MODEL_NAME = "poker44-reference-heuristic"
REFERENCE_REPO_URL = "https://github.com/Poker44/Poker44-subnet"
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


V403_PUBLIC_MODEL = {
    "variant": "v403_v402_natural_multisalt60_sequence",
    "family": "original_v402_natural_multisalt_sequence_ensemble",
    "description": (
        "Frozen public-data-only 50/50 v402 hash-bag and natural multisalt "
        "sequence ensemble; immutable Jul15 blind PASS."
    ),
    "default_top_n": 8,
    "model_file": "data/models/v403_v402_natural_multisalt60_sequence/model.pkl",
    "model_sha256": "f5b4ea0fed33a31c540cc8f2b8afdd0f934f740b3570fa85b6195d948c76c905",
    "model_version": "v403_20260712_123231",
    "frozen_before_source_date": "2026-07-15",
    "blind_report_sha256": "5d4df982c92d5e26eb75d5784c0fe6c645fe3b7a7d575b19fe7199d6dfb5e772",
    "training_refresh": "v403_frozen_blind_passed_2026-07-15",
    "training_data_sources": (
        "https://api.poker44.net/api/v1/benchmark",
        "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-12",
    ),
    "training_data_statement": (
        "Original AceGuard fixed rank ensemble trained only on official split=train Poker44 "
        "canonical miner-visible API SourceUnits through sourceDate 2026-07-12 with "
        "sanitize=False. It combines frozen v402 balanced hash-bag/dense ranks and frozen "
        "v394 hierarchical natural action-policy sequence ranks at fixed 0.50/0.50. "
        "Live-length chunks use twelve deterministic hash-balanced salts with five 40-hand "
        "views each; hash selection controls coverage while every selected subsequence is "
        "restored to natural hand order. Public 30/40-hand source units therefore remain "
        "exactly identical to the rolling-OOF predictor. No private labels, pseudo-labels, "
        "identities, cards, outcomes, competitor code or weights, wallets, hotkeys or IP "
        "addresses were used for training or selection. Private live payloads were used "
        "only for label-blind shape, stability and runtime checks. This local candidate "
        "requires a frozen future sourceDate blind and has no deployment authorization."
    ),
    "package_paths": (
        'data/models/v403_v402_natural_multisalt60_sequence/model.pkl',
        'data/models/v403_v402_natural_multisalt60_sequence/report.json',
        'data/models/v403_v402_natural_multisalt60_sequence/v394_model.pkl',
        'data/models/v403_v402_natural_multisalt60_sequence/v402_model.pkl',
        'poker44/__init__.py',
        'poker44/score/__init__.py',
        'poker44/score/balanced_hash_views.py',
        'poker44/score/chunk_sequence_model.py',
        'poker44/score/chunk_view_aggregation.py',
        'poker44/score/model_view_hand_hash.py',
        'poker44/score/natural_multisalt_sequence_inference.py',
        'poker44/score/natural_order_balanced_views.py',
        'poker44/score/original_hash_bag_features.py',
        'poker44/score/original_multiview_hash_bag_dense_inference.py',
        'poker44/score/original_multiview_hash_bag_inference.py',
        'poker44/score/original_numpy_dense_mlp.py',
        'poker44/score/original_policy_sequence_inference.py',
        'poker44/score/original_policy_sequence_model.py',
        'poker44/score/original_redundancy_features.py',
        'poker44/score/original_schema_contract_features.py',
        'poker44/score/original_tree_surface_features.py',
        'poker44/score/original_tree_surface_inference.py',
        'poker44/score/original_v402_natural_multisalt_sequence_inference.py',
        'poker44/score/scoring.py',
        'poker44/score/sequence_schema.py',
        'poker44/score/temporal_consistency_features.py',
    ),
}


CONTRACT_PUBLIC_RUNTIME_PATHS = (
    "poker44/__init__.py",
    "poker44/score/__init__.py",
    "poker44/score/action_anomaly_features.py",
    "poker44/score/rank_cap_remap.py",
    "poker44/score/robust_schema/__init__.py",
    "poker44/score/robust_schema/features.py",
    "poker44/score/scoring.py",
    "poker44/score/sequence_schema.py",
    "poker44/score/statistical_v25.py",
    "poker44/score/temporal_consistency_features.py",
    "poker44/score/v112_super_inference.py",
)


V415_PUBLIC_MODEL = {
    "variant": "v415_canonical_contract_schema_fit80",
    "family": "v415_canonical_contract_schema_fit80",
    "description": (
        "Frozen canonical public-contract fit80 ranker; immutable Jul16 blind PASS."
    ),
    "default_top_n": 20,
    "model_file": "data/models/v415_canonical_contract_schema_fit80/model.pkl",
    "model_sha256": "7840708c2284cf2eeea4420429692714db1d35b013baa65dab1b5b5a3eedd1f6",
    "model_version": "v415_canonical_contract_schema_fit80_20260712_preblind",
    "frozen_before_source_date": "2026-07-16",
    "blind_report_sha256": "7e8b002971b2fece0b4eb7315d2b2170fbca9074da9b293c742e6d441c04c0e5",
    "training_refresh": "v415_frozen_blind_passed_2026-07-16",
    "training_data_sources": (
        "https://api.poker44.net/api/v1/benchmark",
        "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-12",
    ),
    "training_data_statement": (
        "Trained only on public Poker44 benchmark labels through 2026-07-12. "
        "No wallets, hotkeys, IP addresses, forward-audit labels, validator-private "
        "labels, pseudo-labels, or competitor model weights are used."
    ),
    "package_paths": (
        "data/models/v415_canonical_contract_schema_fit80/model.pkl",
        "data/models/v415_canonical_contract_schema_fit80/report.json",
        *CONTRACT_PUBLIC_RUNTIME_PATHS,
    ),
}


V417_PUBLIC_MODEL = {
    "variant": "v417_contract_hedged_rank_ensemble",
    "family": "v417_contract_hedged_rank_ensemble",
    "description": (
        "Frozen equal-rank canonical and symmetric augmentation ensemble; "
        "immutable Jul17 blind PASS."
    ),
    "default_top_n": 15,
    "model_file": "data/models/v417_contract_hedged_rank_ensemble/model.pkl",
    "model_sha256": "f29c882d75837511baeb9598348b14c0a3211057d1f8c75bf9ce5c6390524dcc",
    "model_version": "v417_contract_hedged_rank_ensemble_20260712_preblind",
    "frozen_before_source_date": "2026-07-17",
    "blind_report_sha256": "62909a6f56a479867d8cc47b06193aa768f4e848c1dea769cad38402ceff6d7a",
    "training_refresh": "v417_frozen_blind_passed_2026-07-17",
    "training_data_sources": (
        "https://api.poker44.net/api/v1/benchmark",
        "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-12",
    ),
    "training_data_statement": (
        "Trained only on public Poker44 benchmark labels through 2026-07-12. "
        "No private validator labels, pseudo-labels, live payload labels, competitor "
        "weights, wallets, hotkeys, or IP addresses are used."
    ),
    "package_paths": (
        "data/models/v417_contract_hedged_rank_ensemble/model.pkl",
        "data/models/v417_contract_hedged_rank_ensemble/report.json",
        *CONTRACT_PUBLIC_RUNTIME_PATHS,
    ),
}

def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _sha256_for_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((p.resolve() for p in paths), key=lambda p: str(p)):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def build_local_model_manifest(
    *,
    repo_root: Path,
    implementation_files: Iterable[Path],
    defaults: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a serializable manifest for the miner's current implementation."""
    implementation_paths = [path.resolve() for path in implementation_files]
    implementation_sha256 = _sha256_for_files(implementation_paths)
    default_values = dict(defaults or {})

    manifest: Dict[str, Any] = {
        "schema_version": "1",
        "open_source": _parse_bool(
            os.getenv("POKER44_MODEL_OPEN_SOURCE"),
            default=bool(default_values.get("open_source", True)),
        ),
        "model_name": os.getenv(
            "POKER44_MODEL_NAME",
            str(default_values.get("model_name", "poker44-reference-heuristic")),
        ),
        "model_version": os.getenv(
            "POKER44_MODEL_VERSION",
            str(default_values.get("model_version", "dev")),
        ),
        "framework": os.getenv(
            "POKER44_MODEL_FRAMEWORK",
            str(default_values.get("framework", "python-heuristic")),
        ),
        "license": os.getenv(
            "POKER44_MODEL_LICENSE",
            str(default_values.get("license", "MIT")),
        ),
        "repo_url": os.getenv(
            "POKER44_MODEL_REPO_URL",
            str(default_values.get("repo_url", "")),
        ).strip(),
        "repo_commit": os.getenv(
            "POKER44_MODEL_REPO_COMMIT",
            str(default_values.get("repo_commit", "")),
        ).strip(),
        "artifact_url": os.getenv(
            "POKER44_MODEL_ARTIFACT_URL",
            str(default_values.get("artifact_url", "")),
        ).strip(),
        "artifact_sha256": os.getenv(
            "POKER44_MODEL_ARTIFACT_SHA256",
            str(default_values.get("artifact_sha256", "")),
        ).strip(),
        "model_card_url": os.getenv(
            "POKER44_MODEL_CARD_URL",
            str(default_values.get("model_card_url", "")),
        ).strip(),
        "training_data_statement": os.getenv(
            "POKER44_MODEL_TRAINING_DATA_STATEMENT",
            str(default_values.get("training_data_statement", "")),
        ).strip(),
        "training_data_sources": [
            item.strip()
            for item in os.getenv(
                "POKER44_MODEL_TRAINING_DATA_SOURCES",
                ",".join(default_values.get("training_data_sources", [])),
            ).split(",")
            if item.strip()
        ],
        "private_data_attestation": os.getenv(
            "POKER44_MODEL_PRIVATE_DATA_ATTESTATION",
            str(default_values.get("private_data_attestation", "")),
        ).strip(),
        "inference_mode": os.getenv(
            "POKER44_MODEL_INFERENCE_MODE",
            str(default_values.get("inference_mode", "remote")),
        ).strip(),
        "implementation_sha256": implementation_sha256,
        "implementation_files": [
            str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)
            for path in implementation_paths
        ],
        "notes": os.getenv(
            "POKER44_MODEL_NOTES",
            str(default_values.get("notes", "")),
        ).strip(),
    }
    return normalize_model_manifest(manifest)


def normalize_model_manifest(manifest: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a compact, JSON-stable manifest dictionary."""
    if not manifest:
        return {}

    normalized: Dict[str, Any] = {}
    for key, value in manifest.items():
        if value is None:
            continue
        if isinstance(value, bool):
            normalized[key] = value
            continue
        if isinstance(value, (int, float)):
            normalized[key] = value
            continue
        if isinstance(value, (list, tuple)):
            cleaned_list: List[Any] = []
            for item in value:
                if item is None:
                    continue
                cleaned_item = str(item).strip()
                if cleaned_item:
                    cleaned_list.append(cleaned_item)
            if cleaned_list:
                normalized[key] = cleaned_list
            continue

        cleaned = str(value).strip()
        if cleaned:
            normalized[key] = cleaned

    if "open_source" in manifest:
        raw = manifest.get("open_source")
        if isinstance(raw, bool):
            normalized["open_source"] = raw
        else:
            normalized["open_source"] = _parse_bool(str(raw), default=False)

    return normalized


def manifest_digest(manifest: Optional[Mapping[str, Any]]) -> str:
    """Return a stable digest for change detection."""
    normalized = normalize_model_manifest(manifest)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _uses_reference_repo(manifest: Mapping[str, Any]) -> bool:
    return str(manifest.get("repo_url", "")).strip().rstrip("/") == REFERENCE_REPO_URL


def _is_reference_miner_manifest(manifest: Mapping[str, Any]) -> bool:
    return str(manifest.get("model_name", "")).strip() == REFERENCE_MINER_MODEL_NAME


def _has_implementation_files(manifest: Mapping[str, Any]) -> bool:
    value = manifest.get("implementation_files")
    if not isinstance(value, (list, tuple)):
        return False
    return any(str(item).strip() for item in value)


def _looks_like_git_commit(value: Any) -> bool:
    return bool(GIT_COMMIT_RE.fullmatch(str(value).strip()))


def evaluate_manifest_compliance(manifest: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Classify whether a manifest meets the current transparent-miner standard."""
    if not manifest:
        return {
            "status": "opaque",
            "missing_fields": list(MIN_REQUIRED_MANIFEST_FIELDS),
            "required_fields": list(MIN_REQUIRED_MANIFEST_FIELDS),
            "open_source": False,
            "policy_violations": [],
        }

    missing_fields: List[str] = []
    for field in MIN_REQUIRED_MANIFEST_FIELDS:
        value = manifest.get(field)
        if field == "open_source":
            if not bool(value):
                missing_fields.append(field)
            continue
        if value is None:
            missing_fields.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing_fields.append(field)
            continue

    if not _has_implementation_files(manifest):
        missing_fields.append("implementation_files")
    if not str(manifest.get("implementation_sha256", "")).strip():
        missing_fields.append("implementation_sha256")

    policy_violations: List[str] = []
    if not _looks_like_git_commit(manifest.get("repo_commit", "")):
        policy_violations.append("repo_commit_invalid")
    if _uses_reference_repo(manifest) and not _is_reference_miner_manifest(manifest):
        policy_violations.append("repo_url_must_point_to_model_repo")

    status = "transparent" if not missing_fields and not policy_violations else "opaque"
    return {
        "status": status,
        "missing_fields": missing_fields,
        "required_fields": list(MIN_REQUIRED_MANIFEST_FIELDS),
        "open_source": bool(manifest.get("open_source", False)),
        "policy_violations": policy_violations,
    }
