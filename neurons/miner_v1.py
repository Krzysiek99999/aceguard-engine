"""Clean public Poker44 miner entrypoint for AceGuard model transparency.

This file intentionally supports only the active public model families:

- v5_statistical
- v10_mild
- v112_super_<strategy>_top<N>

Deployment secrets, wallet names, host details, audit logs, and private run
scripts belong outside the public model repository.
"""

import os
import subprocess
import time
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
            "default_top_n": 2,
        }

    prefix = "v112_super_"
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
            "xgb": "xgb",
            "stack": "stack",
        }
        strategy = strategy_aliases.get(strategy_part, "rank_mean")
        return {
            "family": "v112_super",
            "description": "Supervised schema scorer trained on miner-visible benchmark views.",
            "strategy": strategy,
            "default_top_n": max(1, min(5, top_n)),
            "model_file": "data/models/v112_super/model.pkl",
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
        elif family == "v112_super":
            files.extend(
                [
                    REPO_ROOT / "poker44" / "score" / "v112_super_inference.py",
                    REPO_ROOT / "poker44" / "score" / "robust_schema" / "__init__.py",
                    REPO_ROOT / "poker44" / "score" / "robust_schema" / "features.py",
                    REPO_ROOT / "poker44" / "score" / "statistical_v25.py",
                    REPO_ROOT / "data" / "models" / "v112_super" / "model.pkl",
                ]
            )
        return [path for path in files if path.exists()]

    def _build_manifest(self) -> dict[str, Any]:
        family = self.variant_cfg["family"]
        if family == "v112_super":
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

        return build_local_model_manifest(
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
                "notes": self.variant_cfg["description"],
            },
        )

    def _score_v5(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.calibration import rank_based_calibrate
        from poker44.score.statistical_v5 import score_chunks_v5

        raw_scores = score_chunks_v5(chunks)
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        bot_ratio = min(max(top_n / max(len(raw_scores), 1), 0.0), 0.1)
        return [round(float(v), 6) for v in rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)]

    def _score_v10(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.calibration import rank_based_calibrate
        from poker44.score.stage2_calibration import stage2_calibrate
        from poker44.score.statistical_v9 import score_chunks_v9

        raw_scores, _types = score_chunks_v9(chunks)
        calibrated_scores, mode_top_n, _score_floor = stage2_calibrate(
            raw_scores, mode="mild"
        )
        top_n = _env_int("POKER44_MAX_N", mode_top_n)
        bot_ratio = min(max(top_n / max(len(calibrated_scores), 1), 0.0), 0.1)
        return [
            round(float(v), 6)
            for v in rank_based_calibrate(calibrated_scores, bot_ratio=bot_ratio)
        ]

    def _score_v112_super(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.rank_cap_remap import rank_cap_remap
        from poker44.score.v112_super_inference import score_from_file

        model_file = os.getenv(
            "POKER44_V112_SUPER_MODEL_PATH",
            str(REPO_ROOT / self.variant_cfg["model_file"]),
        )
        model_path = Path(model_file)
        if not model_path.exists():
            bt.logging.error(f"v112_super model missing: {model_path}")
            return [0.49 for _ in chunks]

        strategy = self.variant_cfg.get("strategy", "rank_mean")
        raw_scores = score_from_file(chunks, model_path, strategy=strategy)
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        scores = rank_cap_remap(raw_scores, top_n)
        try:
            bt.logging.info(
                f"v112_super strategy={strategy} top_n={top_n} "
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
            if family == "v5":
                scores = self._score_v5(chunks)
            elif family == "v10":
                scores = self._score_v10(chunks)
            elif family == "v112_super":
                scores = self._score_v112_super(chunks)
            else:
                scores = [0.49 for _ in chunks]
        except Exception as exc:
            bt.logging.error(f"Scoring failed for variant={self.variant}: {exc}")
            scores = [0.49 for _ in chunks]

        synapse.risk_scores = scores
        synapse.predictions = [float(score) >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)
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
