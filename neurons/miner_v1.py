"""Clean public Poker44 miner entrypoint for AceGuard model transparency.

This file intentionally supports only the active public model families:

- v5_statistical
- v8_markov
- v10_mild
- v10_sharp
- v11_ensemble
- v112_super_<strategy>_top<N>
- v113_daily_<strategy>_top<N>
- v115_short_<strategy>_top<N>
- v118_live_<strategy>_top<N>
- v118_stable75_<strategy>_top<N>
- v118_seg35_<strategy>_top<N>
- v125_topk_<strategy>_top<N>

Deployment secrets, wallet names, host details, audit logs, and private run
scripts belong outside the public model repository.
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import bittensor as bt
import numpy as np

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

REPO_ROOT = Path(__file__).resolve().parents[1]


def _audit_enabled() -> bool:
    return os.getenv("POKER44_FORWARD_AUDIT", "0").strip().lower() in {"1", "true", "yes"}


def _audit_full_chunks_enabled() -> bool:
    return os.getenv("POKER44_FORWARD_AUDIT_FULL_CHUNKS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _audit_dedupe_full_chunks_enabled() -> bool:
    return os.getenv("POKER44_FORWARD_AUDIT_DEDUPE_FULL_CHUNKS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _audit_dir() -> Path:
    return Path(os.getenv("POKER44_AUDIT_DIR", str(REPO_ROOT / "data" / "forward_audit")))


_AUDIT_SEEN_FULL_CHUNKS: set[str] | None = None


def _audit_seen_full_chunks(out_dir: Path) -> set[str]:
    global _AUDIT_SEEN_FULL_CHUNKS
    if _AUDIT_SEEN_FULL_CHUNKS is None:
        seen_path = out_dir / "seen_full_chunk_fingerprints.txt"
        if seen_path.exists():
            _AUDIT_SEEN_FULL_CHUNKS = {
                line.strip() for line in seen_path.read_text(errors="replace").splitlines() if line.strip()
            }
        else:
            _AUDIT_SEEN_FULL_CHUNKS = set()
    return _AUDIT_SEEN_FULL_CHUNKS


def _audit_mark_full_chunks_seen(out_dir: Path, fingerprint: str) -> None:
    seen = _audit_seen_full_chunks(out_dir)
    if fingerprint in seen:
        return
    seen.add(fingerprint)
    with (out_dir / "seen_full_chunk_fingerprints.txt").open("a", encoding="utf-8") as f:
        f.write(f"{fingerprint}\n")


def _audit_chunk_fingerprint(chunks: list[list[dict[str, Any]]]) -> str:
    payload = json.dumps(chunks, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _audit_chunk_meta(chunk: list[dict[str, Any]]) -> dict[str, Any]:
    action_count = 0
    streets: set[str] = set()
    for hand in chunk:
        actions = hand.get("actions") if isinstance(hand, dict) else None
        if isinstance(actions, list):
            action_count += len(actions)
            for action in actions:
                if isinstance(action, dict) and action.get("street") is not None:
                    streets.add(str(action.get("street")))
    return {
        "hands": len(chunk),
        "actions": action_count,
        "streets": sorted(streets),
    }


def _write_forward_audit(
    *,
    variant: str,
    family: str,
    manifest_digest_value: str,
    chunks: list[list[dict[str, Any]]],
    raw_scores: list[float],
    final_scores: list[float],
    predictions: list[bool],
) -> None:
    if not _audit_enabled():
        return
    try:
        out_dir = _audit_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = _audit_chunk_fingerprint(chunks)
        record: dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "variant": variant,
            "family": family,
            "manifest_digest": manifest_digest_value,
            "batch_size": len(chunks),
            "batch_fingerprint": fingerprint,
            "chunk_meta": [_audit_chunk_meta(chunk) for chunk in chunks],
            "raw_scores": [round(float(v), 8) for v in raw_scores],
            "final_scores": [round(float(v), 8) for v in final_scores],
            "predictions": [bool(v) for v in predictions],
            "positive_count": int(sum(bool(v) for v in predictions)),
        }
        if _audit_full_chunks_enabled():
            if not _audit_dedupe_full_chunks_enabled() or fingerprint not in _audit_seen_full_chunks(out_dir):
                record["chunks"] = chunks
                record["full_chunks_stored"] = True
                _audit_mark_full_chunks_seen(out_dir, fingerprint)
            else:
                record["full_chunks_stored"] = False
                record["chunks_deduped"] = True
        path = out_dir / f"forward_{variant}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except Exception as exc:
        try:
            bt.logging.debug(f"forward audit write failed: {exc}")
        except Exception:
            pass


def _unwrap_chunks(chunks: list[Any]) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            chunk = chunk.get("hands", chunk.get("chunks", chunk))
        out.append(chunk if isinstance(chunk, list) else [])
    return out


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _repo_commit() -> str:
    value = os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip()
    if value:
        return value
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _variant_config(name: str) -> dict[str, Any]:
    if name == "v5_statistical":
        return {
            "family": "v5",
            "description": "Deterministic behavioral scorer.",
            "default_top_n": 3,
        }
    if name == "v10_mild":
        return {
            "family": "v10",
            "description": "Type-aware deterministic scorer with mild calibration.",
            "stage2_mode": "mild",
            "default_top_n": 2,
        }
    if name == "v10_sharp":
        return {
            "family": "v10",
            "description": "Type-aware deterministic scorer with sharp calibration.",
            "stage2_mode": "sharp",
            "default_top_n": 3,
        }
    if name == "v8_markov":
        return {
            "family": "v8_markov",
            "description": "Deterministic Markov sequence scorer.",
            "default_top_n": 2,
        }
    if name == "v11_ensemble":
        return {
            "family": "v11",
            "description": "Deterministic type-aware ensemble scorer.",
            "default_top_n": 2,
        }

    daily = False
    short_v115 = False
    live_sized = False
    stable75 = False
    seg35 = False
    v125_topk = False
    prefix = "v112_super_"
    if name.startswith("v113_daily_"):
        prefix = "v113_daily_"
        daily = True
    elif name.startswith("v115_short_"):
        prefix = "v115_short_"
        short_v115 = True
    elif name.startswith("v118_live_"):
        prefix = "v118_live_"
        live_sized = True
    elif name.startswith("v118_stable75_"):
        prefix = "v118_stable75_"
        live_sized = True
        stable75 = True
    elif name.startswith("v118_seg35_"):
        prefix = "v118_seg35_"
        live_sized = True
        seg35 = True
    elif name.startswith("v125_topk_"):
        prefix = "v125_topk_"
        live_sized = True
        v125_topk = True
    elif name.startswith("v131_behav_"):
        prefix = "v131_behav_"
        live_sized = True
        v131_behav = True

    if name.startswith(prefix):
        tail = name[len(prefix) :]
        try:
            strategy_part, top_part = tail.rsplit("_top", 1)
            top_n = int(top_part)
        except ValueError:
            strategy_part, top_n = "rank", 1

        strategy_aliases = {
            "rank": "rank_mean",
            "rank_mean": "rank_mean",
            "cat": "cat",
            "et": "et",
            "lgb": "lgb",
            "linear": "linear",
            "xgb": "xgb",
            "stack": "stack",
            "avg": "avg",
        }
        if strategy_part.startswith(("avg_no_", "avg_w", "blend_", "seg")) or strategy_part in {
            "v125_weighted",
        }:
            strategy = strategy_part
        else:
            strategy = strategy_aliases.get(strategy_part, "rank_mean")
        family = (
            "v131_behav"
            if locals().get("v131_behav", False)
            else (
            "v125_topk"
            if v125_topk
            else (
            "v118_seg35"
            if seg35
            else (
                "v118_stable75"
                if stable75
                else (
                    "v118_live"
                    if live_sized
                    else ("v115_short" if short_v115 else ("v113_daily" if daily else "v112_super"))
                )
            )
            )
            )
        )
        return {
            "family": family,
            "description": (
                (
                    "Live-sized behavioural rank stack with batch-normalized schema, sequence, pot-geometry, and temporal consistency features."
                    if locals().get("v131_behav", False)
                    else "Live-sized weighted top-K ET segment scorer with drift-pruned schema and sequence features."
                    if v125_topk
                    else "Live-sized segment scorer with drift-pruned non-money features."
                    if seg35
                    else (
                        "Live-sized supervised scorer with stricter benchmark-to-live stability filtering."
                        if stable75
                        else "Live-sized supervised schema and sequence scorer trained on merged miner-visible benchmark chunks."
                    )
                )
                if live_sized
                else (
                    "Short-unit supervised schema and sequence scorer trained on miner-visible benchmark chunks."
                    if short_v115
                    else (
                    "Daily refreshed supervised schema scorer trained on current miner-visible benchmark views."
                    if daily
                    else "Supervised schema scorer trained on miner-visible benchmark views."
                    )
                )
            ),
            "strategy": strategy,
            "default_top_n": max(1, min(5, top_n)),
            "model_file": (
                "data/models/v131_behav_mix/model.pkl"
                if locals().get("v131_behav", False)
                else (
                "data/models/v125_topk64_et3_weighted/model.pkl"
                if v125_topk
                else (
                "data/models/v118_stableall75/model.pkl"
                if stable75
                else (
                    "data/models/v118_seg35_nomoney/model.pkl"
                    if seg35
                    else (
                        "data/models/v118_livesized_chunks/model.pkl"
                        if live_sized
                        else (
                            "data/models/v115_short/model.pkl"
                            if short_v115
                            else (
                                "data/models/v113_daily/model.pkl"
                                if daily
                                else "data/models/v112_super/model.pkl"
                            )
                        )
                    )
                )
                )
                )
            ),
        }

    bt.logging.warning(
        f"Unknown POKER44_V1_VARIANT={name!r}; falling back to v5_statistical."
    )
    return _variant_config("v5_statistical")


class Miner(BaseMinerNeuron):
    """Miner serving the clean public AceGuard Poker44 models."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.variant = os.getenv("POKER44_V1_VARIANT", "v5_statistical").strip()
        self.variant_cfg = _variant_config(self.variant)
        self._last_raw_scores: list[float] = []
        self.model_manifest = self._build_manifest()
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(
            f"Clean Poker44 miner variant={self.variant} "
            f"family={self.variant_cfg['family']} "
            f"manifest={self.manifest_compliance['status']} "
            f"digest={self.manifest_digest}"
        )

    def _implementation_files(self) -> list[Path]:
        files = [
            Path(__file__).resolve(),
            REPO_ROOT / "poker44" / "utils" / "model_manifest.py",
            REPO_ROOT / "poker44" / "validator" / "synapse.py",
            REPO_ROOT / "poker44" / "score" / "calibration.py",
            REPO_ROOT / "poker44" / "score" / "rank_cap_remap.py",
            REPO_ROOT / "poker44" / "score" / "scoring.py",
        ]

        family = self.variant_cfg["family"]
        if family == "v5":
            files.extend(
                [
                    REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                ]
            )
        elif family == "v10":
            files.extend(
                [
                    REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                    REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                    REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                    REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                    REPO_ROOT / "poker44" / "score" / "features_pot_geometry.py",
                    REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                    REPO_ROOT / "poker44" / "score" / "stage2_calibration.py",
                ]
            )
        elif family == "v8_markov":
            files.extend(
                [
                    REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                    REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                ]
            )
        elif family == "v11":
            files.extend(
                [
                    REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                    REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                    REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                    REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                    REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                    REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                    REPO_ROOT / "poker44" / "score" / "features_pot_geometry.py",
                    REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                ]
            )
        elif family in {"v112_super", "v113_daily", "v115_short", "v118_live", "v118_stable75", "v118_seg35", "v125_topk"}:
            files.extend(
                [
                    REPO_ROOT / "poker44" / "score" / "v112_super_inference.py",
                    REPO_ROOT / "poker44" / "score" / "robust_schema" / "__init__.py",
                    REPO_ROOT / "poker44" / "score" / "robust_schema" / "features.py",
                    REPO_ROOT / "poker44" / "score" / "statistical_v25.py",
                    REPO_ROOT / self.variant_cfg["model_file"],
                ]
            )
        return [path for path in files if path.exists()]

    def _build_manifest(self) -> dict[str, Any]:
        family = self.variant_cfg["family"]
        if family in {"v112_super", "v113_daily", "v115_short", "v118_live", "v118_stable75", "v118_seg35", "v125_topk"}:
            training_statement = (
                "Model trained on public Poker44 benchmark releases using "
                "miner-visible payload views only."
            )
            framework = "python+scikit-learn"
        else:
            training_statement = (
                "Deterministic behavioral scorer using miner-visible hand-history fields only."
            )
            framework = "python-heuristic"

        manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=self._implementation_files(),
            defaults={
                "model_name": os.getenv("POKER44_MODEL_NAME", f"aceguard-{self.variant}"),
                "model_version": os.getenv("POKER44_MODEL_VERSION", "2026.06.17"),
                "framework": framework,
                "license": "MIT",
                "repo_url": os.getenv("POKER44_MODEL_REPO_URL", ""),
                "repo_commit": _repo_commit(),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": training_statement,
                "private_data_attestation": (
                    "No external private user data and no validator-private labels were used."
                ),
                "notes": (
                    f"{self.variant_cfg['description']} "
                    f"variant={self.variant}; "
                    f"family={family}; "
                    f"strategy={self.variant_cfg.get('strategy', family)}; "
                    f"top_n={self.variant_cfg.get('default_top_n')}"
                ),
            },
        )
        manifest["model_variant"] = self.variant
        manifest["model_family"] = family
        manifest["selection_strategy"] = str(self.variant_cfg.get("strategy", family))
        manifest["selection_top_n"] = int(self.variant_cfg.get("default_top_n", 0))
        if family == "v113_daily":
            manifest["training_refresh"] = "daily_candidate_2026-06-18"
        if family == "v115_short":
            manifest["training_refresh"] = "short_unit_sequence_candidate_2026-07-05"
        if family == "v118_live":
            manifest["training_refresh"] = "live_sized_candidate_2026-07-04"
        if family == "v118_stable75":
            manifest["training_refresh"] = "live_sized_stable75_candidate_2026-07-05"
        if family == "v118_seg35":
            manifest["training_refresh"] = "live_sized_seg35_nomoney_candidate_2026-07-06"
        if family == "v125_topk":
            manifest["training_refresh"] = "live_sized_topk_weighted_segment_candidate_2026-07-06"
        return manifest

    def _score_v5(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.calibration import rank_based_calibrate
        from poker44.score.statistical_v5 import score_chunks_v5

        raw_scores = score_chunks_v5(chunks)
        self._last_raw_scores = [float(v) for v in raw_scores]
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        bot_ratio = min(max(top_n / max(len(raw_scores), 1), 0.0), 0.1)
        return [round(float(v), 6) for v in rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)]

    def _score_v10(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.calibration import rank_based_calibrate
        from poker44.score.stage2_calibration import stage2_calibrate, stage2_max_n_adaptive
        from poker44.score.statistical_v9 import score_chunks_v9

        raw_scores, _types = score_chunks_v9(chunks)
        mode = str(self.variant_cfg.get("stage2_mode", "mild"))
        calibrated_scores, mode_top_n, _score_floor = stage2_calibrate(
            raw_scores, mode=mode
        )
        if mode == "sharp":
            mode_top_n = stage2_max_n_adaptive(raw_scores, mode=mode)
        self._last_raw_scores = [float(v) for v in calibrated_scores]
        top_n = _env_int("POKER44_MAX_N", mode_top_n)
        bot_ratio = min(max(top_n / max(len(calibrated_scores), 1), 0.0), 0.1)
        return [
            round(float(v), 6)
            for v in rank_based_calibrate(calibrated_scores, bot_ratio=bot_ratio)
        ]

    def _score_v8_markov(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.calibration import rank_based_calibrate
        from poker44.score.sequence_v8_markov import score_chunks_v8_combined

        raw_scores = score_chunks_v8_combined(chunks)
        self._last_raw_scores = [float(v) for v in raw_scores]
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        bot_ratio = min(max(top_n / max(len(raw_scores), 1), 0.0), 0.1)
        return [round(float(v), 6) for v in rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)]

    def _score_v11(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.calibration import rank_based_calibrate
        from poker44.score.ensemble_v11 import score_chunks_v11

        raw_scores, _telemetry, _types = score_chunks_v11(chunks)
        self._last_raw_scores = [float(v) for v in raw_scores]
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        bot_ratio = min(max(top_n / max(len(raw_scores), 1), 0.0), 0.1)
        return [round(float(v), 6) for v in rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)]

    def _score_schema_model(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.rank_cap_remap import rank_cap_remap
        from poker44.score.v112_super_inference import score_from_file

        if self.variant_cfg["family"] == "v113_daily":
            env_name = "POKER44_V113_DAILY_MODEL_PATH"
        elif self.variant_cfg["family"] == "v115_short":
            env_name = "POKER44_V115_SHORT_MODEL_PATH"
        elif self.variant_cfg["family"] == "v118_live":
            env_name = "POKER44_V118_MODEL_PATH"
        elif self.variant_cfg["family"] == "v118_stable75":
            env_name = "POKER44_V118_STABLE75_MODEL_PATH"
        elif self.variant_cfg["family"] == "v118_seg35":
            env_name = "POKER44_V118_SEG35_MODEL_PATH"
        elif self.variant_cfg["family"] == "v125_topk":
            env_name = "POKER44_V125_MODEL_PATH"
        elif self.variant_cfg["family"] == "v131_behav":
            env_name = "POKER44_V131_MODEL_PATH"
        else:
            env_name = "POKER44_V112_SUPER_MODEL_PATH"
        model_file = os.getenv(env_name, str(REPO_ROOT / self.variant_cfg["model_file"]))
        model_path = Path(model_file)
        if not model_path.exists():
            bt.logging.error(f"{self.variant_cfg['family']} model missing: {model_path}")
            return [0.49 for _ in chunks]

        strategy = self.variant_cfg.get("strategy", "rank_mean")
        raw_scores = score_from_file(chunks, model_path, strategy=strategy)
        self._last_raw_scores = [float(v) for v in raw_scores]
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        scores = rank_cap_remap(raw_scores, top_n)
        try:
            bt.logging.info(
                f"{self.variant_cfg['family']} strategy={strategy} top_n={top_n} "
                f"raw_std={float(np.std(raw_scores)):.4f} "
                f"positives={sum(1 for v in scores if v >= 0.5)}/{len(scores)}"
            )
        except Exception:
            pass
        return [round(float(v), 6) for v in scores]

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = _unwrap_chunks(list(synapse.chunks or []))
        if not chunks:
            synapse.risk_scores = []
            synapse.predictions = []
            synapse.model_manifest = dict(self.model_manifest)
            return synapse

        family = self.variant_cfg["family"]
        try:
            self._last_raw_scores = []
            if family == "v5":
                scores = self._score_v5(chunks)
            elif family == "v10":
                scores = self._score_v10(chunks)
            elif family == "v8_markov":
                scores = self._score_v8_markov(chunks)
            elif family == "v11":
                scores = self._score_v11(chunks)
            elif family in {"v112_super", "v113_daily", "v115_short", "v118_live", "v118_stable75", "v118_seg35", "v125_topk", "v131_behav"}:
                scores = self._score_schema_model(chunks)
            else:
                scores = [0.49 for _ in chunks]
        except Exception as exc:
            bt.logging.error(f"Scoring failed for variant={self.variant}: {exc}")
            scores = [0.49 for _ in chunks]

        synapse.risk_scores = scores
        synapse.predictions = [float(score) >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)
        _write_forward_audit(
            variant=self.variant,
            family=family,
            manifest_digest_value=self.manifest_digest,
            chunks=chunks,
            raw_scores=self._last_raw_scores or scores,
            final_scores=scores,
            predictions=synapse.predictions,
        )
        bt.logging.info(
            f"Scored {len(chunks)} chunks variant={self.variant} "
            f"positives={sum(s >= 0.5 for s in scores)}/{len(scores)} "
            f"min={min(scores):.3f} max={max(scores):.3f}"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info(f"Poker44 miner running variant={miner.variant}")
        while True:
            bt.logging.info("Poker44 miner heartbeat")
            time.sleep(5 * 60)
