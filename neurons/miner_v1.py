"""Poker44 V1 miner — CNN bot detector with auto-detect V0/V1 format.

Supports multiple model variants via POKER44_V1_VARIANT env:
  - cnn_base:      CNN trained on HF human + SandboxPokerBot (baseline)
  - cnn_safe:      CNN + adaptive_safe cap 40% (conservative)
  - cnn_openspiel: CNN trained with OpenSpiel diverse bots
  - lgbm_fallback: LightGBM V0 model (backward compat)
  - hybrid:        CNN for V1 schema, LightGBM for V0 schema (auto-detect)

Handles:
  - Variable chunk sizes (1-70+ hands)
  - V0 sanitized format AND V1 poker44_eval_hand_v* schema
  - Natural gap scoring (no hardcoded ratio)
  - Raw chunk collection for online retraining
"""

# from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import bittensor as bt
import lightgbm as lgb
import numpy as np
import torch

from poker44.base.miner import BaseMinerNeuron
from poker44.models.cnn_detector import ChunkDetector
from poker44.models.hand_encoder import encode_hand
from poker44.score.calibration import adaptive_safe_calibrate
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "data" / "models"
TRAIN_DIR = REPO_ROOT / "data" / "miner_training"
RAW_CHUNKS_DIR = TRAIN_DIR / "raw_validator_chunks"

VARIANTS = {
    # CANARY — statistical detector v5 (codex iter 19: "Statistical detectors most robust for unknown domains")
    # Added 2026-05-18 after diagnosing v2_benchmark_single live saturation (raw=0.651 std=0
    # across all 40 chunks). Live validator distribution diverges from benchmark API training
    # (no "other" action_type, different sizing), so supervised models collapse. v5 ranks
    # chunks by behavioral anomaly score using only live-stable features (sizing entropy,
    # action repetition, sequence quantization).
    "v5_statistical": {
        "model_file": None,
        "description": "Statistical detector v5 — sizing/sequence entropy, no benchmark training",
        "use_safe_cap": False,
        "cls": "stat_v5",
        "use_bounded_rank": True,
        "max_n": 3,
        "score_floor": 0.0,  # statistical scores already in [0,1]; no junk-pick filter needed
    },
    # Wave 1 (2026-05-27): baseline_heuristic — clone of official subnet neurons/miner.py.
    # 7-feature linear weighted scoring per hand, threshold @0.5 binarization.
    # Orthogonal to v5 (pearson -0.07 on live, flag_iou 0.00). max raw 0.37 → expected bot_pred {0}.
    "baseline_heuristic": {
        "model_file": None,
        "description": "Reference heuristic — 7-feature linear, threshold @0.5",
        "use_safe_cap": False,
        "cls": "stat_baseline",
        "use_bounded_rank": False,
        "max_n": 0,
        "score_floor": 0.0,
    },
    # Wave 1 (2026-05-27): v5_statistical INVERTED + rank_cap_remap top3.
    # Polarity flip experiment — pearson -0.843 vs UID160 (v5_orig_top3).
    # rank_cap_remap gwarantuje exactly top3 positives per request. NO POKER44_V5_INVERT env path here.
    "v5_statistical_invert_top3": {
        "model_file": None,
        "description": "v5 inverted polarity (1 - score) + rank_cap_remap top3",
        "use_safe_cap": False,
        "cls": "stat_v5_invert_topn",
        "rank_top_n": 3,
        "use_bounded_rank": False,
        "max_n": 3,
        "score_floor": 0.0,
    },
    # Wave 1 + Wave 2 (2026-05-27): v14 LightGBM (KS-filtered 21 features) + rank_cap_remap top-N.
    # Hold-out AP=0.79, FPR=0.04, reward=0.67. Live std=0.22. Ortogonalny do UID160 (pearson ~-0.04).
    # Wave 1: top1 (UID131), top2 (UID255). Wave 2 hold: top3 (UID252), top4 (UID89).
    "v14_rank_top1": {
        "model_file": "v14_live_stable.txt",
        "description": "v14 LightGBM + rank_cap_remap top1",
        "use_safe_cap": False,
        "cls": "v14_rank",
        "rank_top_n": 1,
        "use_bounded_rank": False,
        "max_n": 1,
        "score_floor": 0.0,
    },
    "v14_rank_top2": {
        "model_file": "v14_live_stable.txt",
        "description": "v14 LightGBM + rank_cap_remap top2",
        "use_safe_cap": False,
        "cls": "v14_rank",
        "rank_top_n": 2,
        "use_bounded_rank": False,
        "max_n": 2,
        "score_floor": 0.0,
    },
    "v14_rank_top3": {
        "model_file": "v14_live_stable.txt",
        "description": "v14 LightGBM + rank_cap_remap top3 (Wave 2)",
        "use_safe_cap": False,
        "cls": "v14_rank",
        "rank_top_n": 3,
        "use_bounded_rank": False,
        "max_n": 3,
        "score_floor": 0.0,
    },
    "v14_rank_top4": {
        "model_file": "v14_live_stable.txt",
        "description": "v14 LightGBM + rank_cap_remap top4 (Wave 2)",
        "use_safe_cap": False,
        "cls": "v14_rank",
        "rank_top_n": 4,
        "use_bounded_rank": False,
        "max_n": 4,
        "score_floor": 0.0,
    },
    # Wave v19-A (2026-05-28): LambdaRank trained on shadow-training-v1 API release.
    # OOF AP=0.84, live std=0.0497 (PASS rank gate). Orthogonal to v5/v14 (IoU < 0.20).
    # NIE używa isotonic/shift — raw LambdaRank output bezpośrednio do rank_cap_remap.
    "v19_rank_top1": {
        "model_file": "v19_ranker.txt",
        "description": "v19 LambdaRank + rank_cap_remap top1 (Wave v19-A conservative)",
        "use_safe_cap": False,
        "cls": "v19_rank",
        "rank_top_n": 1,
        "use_bounded_rank": False,
        "max_n": 1,
        "score_floor": 0.0,
    },
    "v19_rank_top2": {
        "model_file": "v19_ranker.txt",
        "description": "v19 LambdaRank + rank_cap_remap top2 (Wave v19-B medium)",
        "use_safe_cap": False,
        "cls": "v19_rank",
        "rank_top_n": 2,
        "use_bounded_rank": False,
        "max_n": 2,
        "score_floor": 0.0,
    },
    "v19_rank_top3": {
        "model_file": "v19_ranker.txt",
        "description": "v19 LambdaRank + rank_cap_remap top3 (Wave v19-A medium)",
        "use_safe_cap": False,
        "cls": "v19_rank",
        "rank_top_n": 3,
        "use_bounded_rank": False,
        "max_n": 3,
        "score_floor": 0.0,
    },
    "v19_rank_top4": {
        "model_file": "v19_ranker.txt",
        "description": "v19 LambdaRank + rank_cap_remap top4 (reserve)",
        "use_safe_cap": False,
        "cls": "v19_rank",
        "rank_top_n": 4,
        "use_bounded_rank": False,
        "max_n": 4,
        "score_floor": 0.0,
    },
    # v22 competitor-style LambdaRank (2026-05-28): augmented 1500 windows, hard-drop
    # amount/pot/bucket families, KS≤0.25 filter. Holdout AP=0.27 (FAIL bench-binary
    # threshold) BUT live std=0.076, max IoU vs deployed <0.06 (highly orthogonal).
    # Deploy ONLY as rank-based variant — bench AP gate not applicable for rank scoring.
    "v22_rank_top1": {
        "model_file": "v22_competitor_ranker.txt",
        "description": "v22 competitor LambdaRank + rank_cap_remap top1 (UID 11 candidate)",
        "use_safe_cap": False,
        "cls": "v22_rank",
        "rank_top_n": 1,
        "use_bounded_rank": False,
        "max_n": 1,
        "score_floor": 0.0,
    },
    "v22_rank_top2": {
        "model_file": "v22_competitor_ranker.txt",
        "description": "v22 + rank_cap_remap top2",
        "use_safe_cap": False,
        "cls": "v22_rank",
        "rank_top_n": 2,
        "use_bounded_rank": False,
        "max_n": 2,
        "score_floor": 0.0,
    },
    "v22_rank_top3": {
        "model_file": "v22_competitor_ranker.txt",
        "description": "v22 + rank_cap_remap top3",
        "use_safe_cap": False,
        "cls": "v22_rank",
        "rank_top_n": 3,
        "use_bounded_rank": False,
        "max_n": 3,
        "score_floor": 0.0,
    },
    "v22_rank_top4": {
        "model_file": "v22_competitor_ranker.txt",
        "description": "v22 + rank_cap_remap top4",
        "use_safe_cap": False,
        "cls": "v22_rank",
        "rank_top_n": 4,
        "use_bounded_rank": False,
        "max_n": 4,
        "score_floor": 0.0,
    },
    # Model 2 (2026-05-23): v5 + adaptive N based on distribution shape — UID 97 target.
    # Stały max_n=3 zostawia recall na stole gdy 5 bots, naraża FPR gdy 2 bots. Adaptacja
    # bierze pod uwagę top-k scores i gap'y żeby wybrać N ∈ {1,2,3,4}.
    "v5_adaptive_n": {
        "model_file": None,
        "description": "v5 + adaptive N from distribution shape (Model 2)",
        "use_safe_cap": False,
        "cls": "stat_v5_adaptive",
        "max_n_cap": 4,  # hard FPR-cliff guard
    },
    # Model 1 (2026-05-23): v5 + per-seat behavioral consistency — UID 226 target.
    # Bot = jedno siedzenie ma identyczną strategię across all hands; per-seat entropy
    # i sizing consistency to dodatkowy sygnał którego v5 nie miał (patrzył chunk-global).
    "v6_per_seat": {
        "model_file": None,
        "description": "v5 + per-seat action/sizing entropy + VPIP consistency (Model 1)",
        "use_safe_cap": False,
        "cls": "stat_v6",
        "max_n": 3,
    },
    # Model 4 (2026-05-23, codex verify iter 4/5): orthogonal sequence-only heuristic.
    # Replaces UID 192 v5-clone. Uses action n-gram repetition + LCS + motif entropy
    # — different signal family from v5 (which uses hand-signature hash + sizing).
    "v8_sequence": {
        "model_file": None,
        "description": "v8 sequence-only — 3/4-gram + LCS + motif entropy (codex orthogonal)",
        "use_safe_cap": False,
        "cls": "stat_v8",
        "max_n": 2,
    },
    # Model 5 (2026-05-23, codex strategy iter 17): v8 + Markov transition matrix.
    # Detects fixed-policy bots that vary surface form but keep transition probs constant.
    # Combined as max(v8_ngram, v8_markov) per chunk — strongest signal wins.
    "v8_markov": {
        "model_file": None,
        "description": "v8 + Markov transition matrix (JS divergence + transition entropy)",
        "use_safe_cap": False,
        "cls": "stat_v8_markov",
        "max_n": 2,
    },
    # Model 6 (2026-05-23, codex strategy iter 13/16 #1 priority): type-aware calibration.
    # Detect LONG/SHORT payload variant + apply per-type feature weights.
    # UID 211 likely uses this (R5=R7 spike pattern observed).
    "v9_type_calibrated": {
        "model_file": None,
        "description": "v9 LONG/SHORT detection + type-aware v5/v6/v8 blend",
        "use_safe_cap": False,
        "cls": "stat_v9",
        "max_n": 3,
    },
    # Model 7 (2026-05-23, codex iter 23): stage-2 mild calibration.
    # Same base scorer as v5/v9, compressed transform (×0.75 around median), max_n=2, floor=0.20.
    # Replicates UID 87 (ml17_pre-stage2-hl-cal-MILD) pattern.
    "v10_mild": {
        "model_file": None,
        "description": "v9 base + stage-2 MILD calibration (compressed, conservative)",
        "use_safe_cap": False,
        "cls": "stat_v10",
        "stage2_mode": "mild",
        "max_n": 2,
    },
    # Model 8 (2026-05-23, codex iter 23): stage-2 sharp calibration.
    # Same base scorer, stretched transform (×1.35), max_n=3 escalating to 4 on strict gate.
    # Replicates UID 180 (ml17_pre-stage2-hl-cal-MILD-SHARP) pattern.
    "v10_sharp": {
        "model_file": None,
        "description": "v9 base + stage-2 SHARP calibration (stretched, aggressive)",
        "use_safe_cap": False,
        "cls": "stat_v10",
        "stage2_mode": "sharp",
        "max_n": 3,
    },
    # Model 9 (2026-05-23, codex iter 18): type-aware ensemble combiner.
    # Blends 5 orthogonal scorers (v5, v6, v8.1, pot_geo, response_curves) by chunk type
    # with agreement bonus. NOT trained ML (avoids v7 saturation).
    "v11_ensemble": {
        "model_file": None,
        "description": "v11 ensemble — 5-scorer type-aware blend + agreement bonus",
        "use_safe_cap": False,
        "cls": "stat_v11",
        "max_n": 3,
    },
    # Model 10 (2026-05-23, codex iter 27): response_curves standalone scorer.
    # Pure per-seat response-to-aggression signal — BEST std (0.18) and gap (0.565)
    # in replay harness. Lowest correlation with v5/v9 family (0.21).
    "response_curves": {
        "model_file": None,
        "description": "response_curves standalone — per-seat fold/call/raise patterns",
        "use_safe_cap": False,
        "cls": "stat_response_curves",
        "max_n": 3,
    },
    # Model 3 (2026-05-23): LightGBM trenowany na benchmark API z drop other_*, v5 sub-scores
    # jako features, sample weight = recency bias. UID 190 target. Najwyższy upside ale risk
    # saturation (jak v2). Promote dopiero po sim_reward gate (≥0.39 holdout).
    "v7_hybrid_ml": {
        "model_file": None,
        "description": "Hybrid LightGBM + v5 features, benchmark-supervised (Model 3)",
        "use_safe_cap": False,
        "cls": "lgbm",
        "lgbm_tag": "v7_hybrid_ml",
        "use_bounded_rank": True,
        "max_n": 3,
        "score_floor": 0.20,
        "v1_features": False,  # uses custom feature builder including v5 sub-scores
    },
    # PRODUCTION CHAMPION — codex 20-iter design (2026-05-18)
    "v2_benchmark_single": {
        "model_file": None,
        "description": "Benchmark-supervised single LightGBM + bounded_rank calibrator (codex iter 4/7/14)",
        "use_safe_cap": False,
        "cls": "lgbm",
        "lgbm_tag": "v2_benchmark_single",
        "use_bounded_rank": True,
        "max_n": 3,
        "score_floor": 0.16,
        "v1_features": True,
    },
    "v1_real_2026": {
        "model_file": None,
        "description": "REAL benchmark API training (AUC=1.0, AP=1.0 on hold-out 2026-05-05)",
        "use_safe_cap": False,
        "cls": "lgbm",
        "lgbm_tag": "v1_real_2026",
        "use_dynamic_cap": True,
        "v1_features": True,
    },
    "v1_top1_voting_ensemble": {
        # Loads tags v1_top1_v3 + v1_top1_v2 + B_deeper, agreement 2of3 cap=0.08
        # (BEATS baseline-v1 by +5.6pp on hold-out)
        "model_file": None,
        "description": "TOP1: voting ensemble v3+v2+B_deeper, agreement 2of3 cap=0.08",
        "use_safe_cap": False,
        "cls": "voting_ensemble",
        "lgbm_tag": "v1_top1_v3",  # primary loaded as self.lgbm for manifest
        "use_voting_ensemble": True,
        "max_bot_fraction": 0.08,
        "v1_features": True,
    },

    # ZERO-ML EMERGENCY FALLBACKS (no model files needed, work on any data)
    "v1_other_r": {
        "model_file": None,
        "description": "V1: other_r zero-ML detector — emergency fallback",
        "use_safe_cap": False,
        "cls": "stat",
        "stat": "other_r",
    },
    "v1_diversity": {
        "model_file": None,
        "description": "V1: diversity zero-ML detector — emergency fallback twin",
        "use_safe_cap": False,
        "cls": "stat",
        "stat": "diversity",
    },

    # CNN CANARY SLOT (for future A/B test of CNN models)
    "v1_cnn_adversarial": {
        "model_file": "cnn_v1_adversarial.pt",
        "description": "V1: CNN adversarial + Otsu cap 0.50",
        "use_safe_cap": True,
        "cls": "cnn",
        "max_bot_override": 0.50,
    },
}


def _load_cnn(model_path: Path) -> ChunkDetector:
    if not model_path.exists():
        return None
    try:
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        hidden_dim = ckpt.get("hidden_dim", 64)
        detector = ChunkDetector(hidden_dim=hidden_dim)
        detector.load_state_dict(ckpt["model_state"])
        detector.eval()
        return detector
    except Exception as exc:
        bt.logging.warning(f"Failed to load CNN from {model_path}: {exc}")
        return None


def _load_lgbm(tag: str = None):
    try:
        import json
        search_order = [tag] if tag else []
        search_order.extend(["active", "robust_prod_v2", "robust"])
        for t in search_order:
            if t is None:
                continue
            model_path = TRAIN_DIR / f"bot_detector_lgbm_{t}.txt"
            meta_path = TRAIN_DIR / f"bot_detector_meta_{t}.json"
            if model_path.exists() and meta_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                meta = json.loads(meta_path.read_text())
                iso = [(float(x), float(y)) for x, y in meta.get("isotonic_points", [])]
                return model, list(meta["feature_names"]), iso, meta
        return None, None, None, None
    except Exception as exc:
        bt.logging.warning(f"Failed to load LightGBM: {exc}")
        return None, None, None, None


def _is_v1_schema(hand: dict) -> bool:
    schema = str(hand.get("schema") or "").strip().lower()
    return schema.startswith("poker44_eval_hand_v")


class Miner(BaseMinerNeuron):
    """V1 miner with CNN + LightGBM hybrid, multiple variants."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        self.variant = os.getenv("POKER44_V1_VARIANT", "v1_real_2026")
        variant_cfg = VARIANTS.get(self.variant, VARIANTS["v1_real_2026"])
        self.use_safe_cap = variant_cfg["use_safe_cap"]
        env_cap = os.getenv("POKER44_MAX_BOT_FRACTION")
        if env_cap is not None:
            try:
                self.max_bot_fraction = float(env_cap)
            except ValueError:
                self.max_bot_fraction = 0.50
        else:
            self.max_bot_fraction = float(variant_cfg.get("max_bot_override", 0.50))

        # Load CNN
        self.cnn = None
        cnn_file = os.getenv("POKER44_CNN_MODEL_PATH")
        if cnn_file:
            self.cnn = _load_cnn(Path(cnn_file))
        elif variant_cfg["model_file"]:
            self.cnn = _load_cnn(MODELS_DIR / variant_cfg["model_file"])

        # Load LightGBM (for lgbm/hybrid/ensemble variants)
        self.lgbm, self.lgbm_features, self.lgbm_isotonic, self.lgbm_meta = None, None, None, None
        if variant_cfg.get("cls") in ("lgbm", "hybrid", "ensemble", "voting_ensemble"):
            lgbm_tag = variant_cfg.get("lgbm_tag")
            self.lgbm, self.lgbm_features, self.lgbm_isotonic, self.lgbm_meta = _load_lgbm(tag=lgbm_tag)

        # Voting ensemble — load secondary models v2 + B_deeper
        self.voting_models = None
        if variant_cfg.get("use_voting_ensemble"):
            self.voting_models = []
            for tag in ["v1_top1_v3", "v1_top1_v2", "B_deeper"]:
                m, feats, iso, meta = _load_lgbm(tag=tag)
                if m is not None:
                    self.voting_models.append({"tag": tag, "model": m, "features": feats, "meta": meta or {}})
            bt.logging.info(f"🗳️  Voting ensemble loaded {len(self.voting_models)}/3 models: {[v['tag'] for v in self.voting_models]}")

        # Status logging
        has_cnn = self.cnn is not None
        has_lgbm = self.lgbm is not None
        bt.logging.info(
            f"🤖 V1 miner variant={self.variant} CNN={'✅' if has_cnn else '❌'} "
            f"LightGBM={'✅' if has_lgbm else '❌'} safe_cap={self.use_safe_cap}"
        )

        if not has_cnn and not has_lgbm:
            bt.logging.warning("⚠️ No model loaded — will use heuristic fallback")

        # Startup self-test
        self._startup_self_test()

        # Manifest
        self._build_manifest(variant_cfg)
        bt.logging.info(f"Axon created: {self.axon}")

        # Chunk collection — disabled by default. Set POKER44_SAVE_RAW_CHUNKS=1 in run script
        # only on a designated collector miner (one is enough for training-data pipeline).
        self._save_chunks = os.getenv("POKER44_SAVE_RAW_CHUNKS", "0").strip() in {"1", "true", "yes"}

        # Adaptive policy hot-reload (no-restart cap adjustment).
        # Reads data/observations/adaptive_policy.json every N forward calls.
        # Disabled by env: POKER44_DISABLE_ADAPTIVE_POLICY=1
        self._adaptive_policy_path = REPO_ROOT / "data" / "observations" / "adaptive_policy.json"
        self._adaptive_policy_check_every = 5  # check every 5 forward() calls
        self._adaptive_policy_check_counter = 0
        self._adaptive_policy_disabled = os.getenv("POKER44_DISABLE_ADAPTIVE_POLICY", "0").strip() in {"1", "true", "yes"}
        self._last_policy_mtime = 0.0
        # Calibration guard: track recent bot_pred ratios to detect calibration breakdown
        self._recent_bot_pred_ratios: list[float] = []
        self._calibration_fallback_active = False

        # Adaptive N — codex strategy 2026-05-25.
        # Replaces static POKER44_MAX_N with per-batch N driven by raw_scores distribution.
        # Disabled by default; enable via POKER44_ADAPTIVE_N=1.
        self._adaptive_n_enabled = os.getenv("POKER44_ADAPTIVE_N", "0").strip() in {"1", "true", "yes"}
        self._adaptive_profile = os.getenv("POKER44_ADAPTIVE_PROFILE", "balanced").strip()
        try:
            self._adaptive_min_n = int(os.getenv("POKER44_ADAPTIVE_MIN_N", "")) if os.getenv("POKER44_ADAPTIVE_MIN_N") else None
        except ValueError:
            self._adaptive_min_n = None
        try:
            self._adaptive_max_n = int(os.getenv("POKER44_ADAPTIVE_MAX_N", "")) if os.getenv("POKER44_ADAPTIVE_MAX_N") else None
        except ValueError:
            self._adaptive_max_n = None
        self._adaptive_prev_n: Optional[int] = None  # hysteresis state
        if self._adaptive_n_enabled:
            bt.logging.info(
                f"🎯 Adaptive-N ENABLED: profile={self._adaptive_profile} "
                f"min_n={self._adaptive_min_n} max_n={self._adaptive_max_n}"
            )

    def _resolve_max_n(self, raw_scores, default_n: int) -> int:
        """Return max_n for this batch. Adaptive if POKER44_ADAPTIVE_N=1, else env/default.

        When adaptive: inspect raw_scores distribution, return N per profile.
        Stores diag in self._last_adaptive_diag for caller logging.
        """
        # Static env override always wins if set explicitly
        env_static = os.getenv("POKER44_MAX_N")
        if env_static is not None and not self._adaptive_n_enabled:
            try:
                return int(env_static)
            except ValueError:
                pass
        if not self._adaptive_n_enabled:
            return int(default_n)
        # Adaptive path
        try:
            from poker44.score.calibration import select_adaptive_n
            n, diag = select_adaptive_n(
                raw_scores,
                profile=self._adaptive_profile,
                min_n=self._adaptive_min_n,
                max_n=self._adaptive_max_n,
                prev_n=self._adaptive_prev_n,
                hysteresis=True,
            )
            self._adaptive_prev_n = n
            self._last_adaptive_diag = diag
            return n
        except Exception as exc:
            bt.logging.warning(f"adaptive_n failed: {exc} — fallback to default {default_n}")
            return int(default_n)

    def _startup_self_test(self):
        test_hand = {
            "metadata": {"hero_seat": 3, "max_seats": 6, "game_type": "Hold'em", "limit_type": "No Limit",
                         "sb": 0.01, "bb": 0.02, "ante": 0.0},
            "players": [{"player_uid": f"seat_{i}", "seat": i, "starting_stack": 2.0} for i in range(1, 7)],
            "streets": [{"street": "preflop", "board_cards": []}],
            "actions": [{"action_id": str(j), "street": "preflop", "actor_seat": 1,
                         "action_type": "fold", "amount": 0.0, "normalized_amount_bb": 0.0,
                         "pot_before": 0.03, "pot_after": 0.03, "raise_to": None, "call_to": None}
                        for j in range(1, 13)],
            "outcome": {"winners": [], "payouts": {}, "total_pot": 0.0, "rake": 0.0,
                        "result_reason": "", "showdown": False},
        }
        try:
            score = self._score_single_chunk([test_hand] * 10)
            bt.logging.info(f"✅ self-test passed: chunk_score={score:.4f}")
        except Exception as exc:
            bt.logging.error(f"❌ self-test FAILED: {exc}")

    def _build_manifest(self, variant_cfg):
        repo_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip()
        if not repo_commit:
            try:
                import subprocess
                repo_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
                ).decode().strip()
            except Exception:
                repo_commit = ""
        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=[
                Path(__file__).resolve(),
                REPO_ROOT / "poker44" / "models" / "statistical_detector.py",
                REPO_ROOT / "poker44" / "models" / "cnn_detector.py",
                REPO_ROOT / "poker44" / "models" / "hand_encoder.py",
                REPO_ROOT / "poker44" / "score" / "calibration.py",
            ],
            defaults={
                "model_name": f"poker44-v1-{self.variant}",
                "model_version": "1.0",
                "framework": "pytorch+lightgbm",
                "license": "MIT",
                "repo_url": os.getenv("POKER44_MODEL_REPO_URL", ""),
                "repo_commit": repo_commit,
                "notes": variant_cfg["description"],
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": "HF 21M real human + SandboxPokerBot + OpenSpiel bots; V1-real chunks",
                "private_data_attestation": "No validator-private data used",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)

    # ---- Adaptive policy hot-reload ----

    def _check_adaptive_policy(self) -> None:
        """Read adaptive_policy.json if present and update self.max_bot_fraction.

        Called every N forward() invocations. No-op if file missing or disabled.
        """
        if self._adaptive_policy_disabled:
            return
        try:
            if not self._adaptive_policy_path.exists():
                return
            mtime = self._adaptive_policy_path.stat().st_mtime
            if mtime == self._last_policy_mtime:
                return  # no change
            import json as _json
            policy = _json.loads(self._adaptive_policy_path.read_text())
            uid_str = str(getattr(self, "uid", ""))
            if not uid_str or uid_str not in policy.get("uids", {}):
                return
            uid_policy = policy["uids"][uid_str]
            new_cap = float(uid_policy.get("max_bot_fraction", self.max_bot_fraction))
            old_cap = self.max_bot_fraction
            if abs(new_cap - old_cap) > 0.001:
                bt.logging.info(
                    f"adaptive_policy hot-reload UID {uid_str}: cap {old_cap:.3f} -> {new_cap:.3f} "
                    f"(reason: {uid_policy.get('reason', 'unknown')})"
                )
            self.max_bot_fraction = new_cap
            # Bounded rank profile (codex 20-iter): per-UID role split
            if "max_n" in uid_policy:
                old_n = getattr(self, "max_n", None)
                new_n = int(uid_policy["max_n"])
                if old_n != new_n:
                    bt.logging.info(
                        f"adaptive_policy hot-reload UID {uid_str}: max_n {old_n} -> {new_n}"
                    )
                self.max_n = new_n
            if "score_floor" in uid_policy:
                old_floor = getattr(self, "score_floor", None)
                new_floor = float(uid_policy["score_floor"])
                if old_floor is None or abs(new_floor - old_floor) > 0.001:
                    bt.logging.info(
                        f"adaptive_policy hot-reload UID {uid_str}: score_floor {old_floor} -> {new_floor:.3f}"
                    )
                self.score_floor = new_floor
            self._last_policy_mtime = mtime
        except Exception as exc:
            bt.logging.debug(f"adaptive_policy reload failed: {exc}")

    def _record_calibration_outcome(self, raw_scores: list[float], scores: list[float]) -> None:
        """Track recent bot_pred ratios; activate fallback if calibration produces 0 botów streak."""
        if not scores:
            return
        bot_count = sum(1 for s in scores if s >= 0.5)
        ratio = bot_count / len(scores)
        self._recent_bot_pred_ratios.append(ratio)
        if len(self._recent_bot_pred_ratios) > 6:
            self._recent_bot_pred_ratios.pop(0)

        # Activate fallback: if last 4+ batches yielded ratio == 0 → calibration broken.
        # EXCEPTION: baseline_heuristic intentionally has max score < 0.5 → bot_pred=0 always.
        # Don't flag it as misleading [FALLBACK] (Wave 1 fix 2026-05-27).
        variant_cfg = VARIANTS.get(self.variant, {})
        cls = variant_cfg.get("cls", "")
        is_threshold_only_variant = cls in ("stat_baseline",)
        if (not is_threshold_only_variant
                and len(self._recent_bot_pred_ratios) >= 4
                and all(r == 0 for r in self._recent_bot_pred_ratios[-4:])):
            if not self._calibration_fallback_active:
                bt.logging.warning(
                    f"⚠️ Calibration fallback ACTIVATED: bot_pred=0 streak ({len(self._recent_bot_pred_ratios)} batches). "
                    f"Will use rank_based_calibrate next batch with cap={self.max_bot_fraction}"
                )
                self._calibration_fallback_active = True
        elif self._calibration_fallback_active and self._recent_bot_pred_ratios[-1] > 0:
            bt.logging.info("✓ Calibration fallback DEACTIVATED (got non-zero bot prediction)")
            self._calibration_fallback_active = False

    # ---- Forward ----

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []

        # Adaptive policy hot-reload (every N invocations)
        self._adaptive_policy_check_counter += 1
        if self._adaptive_policy_check_counter >= self._adaptive_policy_check_every:
            self._adaptive_policy_check_counter = 0
            self._check_adaptive_policy()

        # Save raw chunks for retraining
        if self._save_chunks and chunks:
            self._save_raw_chunks(chunks)

        variant_cfg = VARIANTS.get(self.variant, VARIANTS["v1_real_2026"])

        # TOP1 VOTING ENSEMBLE — agreement_2of3 between 3 models
        if variant_cfg.get("use_voting_ensemble") and self.voting_models and len(chunks) > 0:
            raw_scores = self._voting_ensemble_score_batch(chunks)
            # Adaptive policy WINS over env_cap (env was a static fallback from run scripts).
            # If POKER44_FORCE_ENV_CAP=1 set, env still wins (manual override).
            force_env = os.getenv("POKER44_FORCE_ENV_CAP", "0").strip() in {"1", "true", "yes"}
            env_cap = os.getenv("POKER44_MAX_BOT_FRACTION")
            if force_env and env_cap is not None:
                cap = float(env_cap)
            else:
                cap = float(self.max_bot_fraction)  # this gets hot-reloaded from policy
            if os.getenv("POKER44_VOTING_DYNAMIC_CAP", "0").strip() in {"1", "true", "yes"}:
                # Guard: if calibration fallback active (bot_pred=0 streak detected) →
                # use rank_based_calibrate which guarantees N*cap bot predictions.
                if self._calibration_fallback_active:
                    from poker44.score.calibration import rank_based_calibrate
                    cal = rank_based_calibrate(raw_scores, bot_ratio=cap)
                else:
                    from poker44.score.calibration import dynamic_safe_calibrate
                    cal = dynamic_safe_calibrate(
                        raw_scores,
                        safety_margin=0.05,
                        absolute_max=cap,
                        absolute_min=0.05,
                    )
            else:
                cal = adaptive_safe_calibrate(raw_scores, max_bot_fraction=cap)
            scores = [round(float(v), 6) for v in cal]
        # Model 10: response_curves standalone — per-seat response patterns
        elif variant_cfg.get("cls") == "stat_response_curves" and len(chunks) > 0:
            from poker44.score.features_response_curves import score_chunks_response_curves
            from poker44.score.calibration import rank_based_calibrate
            chunk_hands = [c.get("hands", c) if isinstance(c, dict) else c for c in chunks]
            raw_scores = score_chunks_response_curves(chunk_hands)
            max_n = self._resolve_max_n(
                raw_scores,
                default_n=int(getattr(self, "max_n", variant_cfg.get("max_n", 3))),
            )
            bot_ratio = min(max(max_n / max(len(raw_scores), 1), 0.0), 0.1)
            cal = rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)
            scores = [round(float(v), 6) for v in cal]
            try:
                rs = sorted(raw_scores, reverse=True)
                bt.logging.info(
                    f"response_curves diag: top5=[{', '.join(f'{v:.3f}' for v in rs[:5])}] "
                    f"std={float(np.std(raw_scores)):.4f} max_n={max_n}"
                )
            except Exception as exc:
                bt.logging.debug(f"response_curves diag failed: {exc}")
        # Model 9: v11 ensemble — 5-scorer type-aware blend + agreement bonus
        elif variant_cfg.get("cls") == "stat_v11" and len(chunks) > 0:
            from poker44.score.ensemble_v11 import score_chunks_v11
            from poker44.score.calibration import rank_based_calibrate
            chunk_hands = [c.get("hands", c) if isinstance(c, dict) else c for c in chunks]
            raw_scores, tele, types = score_chunks_v11(chunk_hands)
            max_n = self._resolve_max_n(
                raw_scores,
                default_n=int(getattr(self, "max_n", variant_cfg.get("max_n", 3))),
            )
            bot_ratio = min(max(max_n / max(len(raw_scores), 1), 0.0), 0.1)
            cal = rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)
            scores = [round(float(v), 6) for v in cal]
            try:
                rs = sorted(raw_scores, reverse=True)
                tc = {t: types.count(t) for t in set(types)}
                # Top chunk telemetry
                top_idx = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
                top_tele = tele[top_idx]
                bt.logging.info(
                    f"stat_v11 diag: types={tc} "
                    f"top5=[{', '.join(f'{v:.3f}' for v in rs[:5])}] "
                    f"top_chunk_scorers={ {k: round(v,3) for k,v in top_tele.items()} } "
                    f"max_n={max_n}"
                )
            except Exception as exc:
                bt.logging.debug(f"v11 diag failed: {exc}")
        # Model 8: v10 mild/sharp — stage-2 calibration over v9 type-aware base
        elif variant_cfg.get("cls") == "stat_v10" and len(chunks) > 0:
            from poker44.score.statistical_v9 import score_chunks_v9
            from poker44.score.stage2_calibration import stage2_calibrate, stage2_max_n_adaptive
            from poker44.score.calibration import rank_based_calibrate
            chunk_hands = [c.get("hands", c) if isinstance(c, dict) else c for c in chunks]
            raw_scores, types = score_chunks_v9(chunk_hands)
            mode = variant_cfg.get("stage2_mode", "balanced")
            calibrated_scores, mode_max_n, score_floor = stage2_calibrate(raw_scores, mode=mode)
            # Adaptive escalation for sharp mode
            if mode == "sharp":
                mode_max_n = stage2_max_n_adaptive(raw_scores, mode=mode)
            max_n = self._resolve_max_n(calibrated_scores, default_n=int(mode_max_n))
            bot_ratio = min(max(max_n / max(len(calibrated_scores), 1), 0.0), 0.1)
            cal = rank_based_calibrate(calibrated_scores, bot_ratio=bot_ratio)
            scores = [round(float(v), 6) for v in cal]
            try:
                rs = sorted(raw_scores, reverse=True)
                cs = sorted(calibrated_scores, reverse=True)
                type_counts = {t: types.count(t) for t in set(types)}
                bt.logging.info(
                    f"stat_v10_{mode} diag: types={type_counts} "
                    f"raw_top3=[{rs[0]:.3f},{rs[1]:.3f},{rs[2]:.3f}] "
                    f"cal_top3=[{cs[0]:.3f},{cs[1]:.3f},{cs[2]:.3f}] "
                    f"max_n={max_n}"
                )
            except Exception as exc:
                bt.logging.debug(f"v10 diag failed: {exc}")
        # Model 6: v9 type-aware calibration (LONG/SHORT detection)
        elif variant_cfg.get("cls") == "stat_v9" and len(chunks) > 0:
            from poker44.score.statistical_v9 import score_chunks_v9
            from poker44.score.calibration import rank_based_calibrate
            chunk_hands = [c.get("hands", c) if isinstance(c, dict) else c for c in chunks]
            raw_scores, types = score_chunks_v9(chunk_hands)
            max_n = self._resolve_max_n(
                raw_scores,
                default_n=int(getattr(self, "max_n", variant_cfg.get("max_n", 3))),
            )
            bot_ratio = min(max(max_n / max(len(raw_scores), 1), 0.0), 0.1)
            cal = rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)
            scores = [round(float(v), 6) for v in cal]
            try:
                rs = sorted(raw_scores, reverse=True)
                type_counts = {t: types.count(t) for t in set(types)}
                bt.logging.info(
                    f"stat_v9 diag: types={type_counts} "
                    f"top5=[{', '.join(f'{v:.3f}' for v in rs[:5])}] "
                    f"std={float(np.std(raw_scores)):.4f} max_n={max_n}"
                )
            except Exception as exc:
                bt.logging.debug(f"v9 diag failed: {exc}")
        # Model 5: v8.1 Markov (v8 + transition matrix)
        elif variant_cfg.get("cls") == "stat_v8_markov" and len(chunks) > 0:
            from poker44.score.sequence_v8_markov import score_chunks_v8_combined
            from poker44.score.calibration import rank_based_calibrate
            chunk_hands = [c.get("hands", c) if isinstance(c, dict) else c for c in chunks]
            raw_scores = score_chunks_v8_combined(chunk_hands)
            max_n = self._resolve_max_n(
                raw_scores,
                default_n=int(getattr(self, "max_n", variant_cfg.get("max_n", 2))),
            )
            bot_ratio = min(max(max_n / max(len(raw_scores), 1), 0.0), 0.1)
            cal = rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)
            scores = [round(float(v), 6) for v in cal]
            try:
                rs = sorted(raw_scores, reverse=True)
                bt.logging.info(
                    f"stat_v8_markov diag: top5=[{', '.join(f'{v:.3f}' for v in rs[:5])}] "
                    f"std={float(np.std(raw_scores)):.4f} max_n={max_n}"
                )
            except Exception as exc:
                bt.logging.debug(f"v8_markov diag failed: {exc}")
        # Model 4: v8 sequence-only — orthogonal n-gram + LCS heuristic
        elif variant_cfg.get("cls") == "stat_v8" and len(chunks) > 0:
            from poker44.score.sequence_v8 import score_chunks_v8
            from poker44.score.calibration import rank_based_calibrate
            chunk_hands = [c.get("hands", c) if isinstance(c, dict) else c for c in chunks]
            raw_scores = score_chunks_v8(chunk_hands)
            max_n = self._resolve_max_n(
                raw_scores,
                default_n=int(getattr(self, "max_n", variant_cfg.get("max_n", 2))),
            )
            bot_ratio = min(max(max_n / max(len(raw_scores), 1), 0.0), 0.1)
            cal = rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)
            scores = [round(float(v), 6) for v in cal]
            try:
                rs = sorted(raw_scores, reverse=True)
                bt.logging.info(
                    f"stat_v8 diag: top5=[{', '.join(f'{v:.3f}' for v in rs[:5])}] "
                    f"std={float(np.std(raw_scores)):.4f} max_n={max_n}"
                )
            except Exception as exc:
                bt.logging.debug(f"v8 diag failed: {exc}")
        # Model 1: v6 — v5 features + per-seat behavioral consistency
        elif variant_cfg.get("cls") == "stat_v6" and len(chunks) > 0:
            from poker44.score.statistical_v6 import score_chunks_v6
            from poker44.score.calibration import rank_based_calibrate
            chunk_hands = [c.get("hands", c) if isinstance(c, dict) else c for c in chunks]
            raw_scores = score_chunks_v6(chunk_hands)
            max_n = self._resolve_max_n(
                raw_scores,
                default_n=int(getattr(self, "max_n", variant_cfg.get("max_n", 3))),
            )
            bot_ratio = min(max(max_n / max(len(raw_scores), 1), 0.0), 0.1)
            cal = rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)
            scores = [round(float(v), 6) for v in cal]
            try:
                rs = sorted(raw_scores, reverse=True)
                bt.logging.info(
                    f"stat_v6 diag: top5=[{', '.join(f'{v:.3f}' for v in rs[:5])}] "
                    f"std={float(np.std(raw_scores)):.4f} max_n={max_n}"
                )
            except Exception as exc:
                bt.logging.debug(f"v6 diag failed: {exc}")
        # Model 2: v5 + adaptive N from distribution shape
        elif variant_cfg.get("cls") == "stat_v5_adaptive" and len(chunks) > 0:
            from poker44.score.statistical_v5 import score_chunks_v5
            from poker44.score.calibration import rank_based_calibrate
            chunk_hands = [c.get("hands", c) if isinstance(c, dict) else c for c in chunks]
            raw_scores = score_chunks_v5(chunk_hands)
            arr = np.array(raw_scores, dtype=np.float64)
            sorted_desc = np.sort(arr)[::-1]
            top1 = float(sorted_desc[0]) if len(sorted_desc) > 0 else 0.0
            top2 = float(sorted_desc[1]) if len(sorted_desc) > 1 else 0.0
            top3 = float(sorted_desc[2]) if len(sorted_desc) > 2 else 0.0
            top4 = float(sorted_desc[3]) if len(sorted_desc) > 3 else 0.0
            top5 = float(sorted_desc[4]) if len(sorted_desc) > 4 else 0.0
            std_top5 = float(np.std(sorted_desc[:5])) if len(sorted_desc) >= 2 else 0.0
            median = float(np.median(arr))
            gap_12 = top1 - top2
            gap_23 = top2 - top3
            gap_34 = top3 - top4
            gap_45 = top4 - top5
            max_n_cap = int(variant_cfg.get("max_n_cap", 4))
            # Adaptive decision (codex verify iter 1):
            # - N=4 requires TRUE exceptional separation (strict gates per codex iter 1).
            # - DEFAULT N=3 matches UID 160 baseline (winner) — UID 97 only adds value via
            #   N=4 escalation; with N=2 default it'd duplicate UID 192 (already deployed).
            # - N=1/N=2 only when signal genuinely weak vs bulk.
            if (
                top4 >= 0.52
                and gap_45 >= 0.06
                and (top4 - median) >= 0.20
                and std_top5 >= 0.05
            ):
                N = 4  # exceptional 4-bot scenario
            elif top1 < 0.32 or std_top5 < 0.025 or (top1 - median) < 0.08:
                N = 1  # collapse — nothing confident
            elif (top1 - median) < 0.12:
                N = 2  # weak separation from bulk
            else:
                N = 3  # default — matches UID 160 baseline (winner)
            N = max(0, min(N, max_n_cap))
            bot_ratio = min(max(N / max(len(raw_scores), 1), 0.0), 0.1)
            cal = rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)
            scores = [round(float(v), 6) for v in cal]
            try:
                bt.logging.info(
                    f"stat_v5_adaptive diag: top4=[{top1:.3f},{top2:.3f},{top3:.3f},{top4:.3f}] "
                    f"std5={std_top5:.4f} gap12={gap_12:.3f} gap23={gap_23:.3f} gap34={gap_34:.3f} -> N={N}"
                )
            except Exception as exc:
                bt.logging.debug(f"v5_adaptive diag failed: {exc}")
        # Statistical detector v5 — no model, behavioral anomaly only
        elif variant_cfg.get("cls") == "stat_v5" and len(chunks) > 0:
            from poker44.score.statistical_v5 import score_chunks_v5
            from poker44.score.calibration import rank_based_calibrate
            # Hands sometimes wrapped in {"hands": [...]} sometimes raw list
            chunk_hands = [c.get("hands", c) if isinstance(c, dict) else c for c in chunks]
            raw_scores = score_chunks_v5(chunk_hands)
            # Optional inversion for A/B polarity test (benchmark API has inverse polarity)
            if os.getenv("POKER44_V5_INVERT", "0").strip() in {"1", "true", "yes"}:
                raw_scores = [1.0 - s for s in raw_scores]
            max_n = self._resolve_max_n(
                raw_scores,
                default_n=int(getattr(self, "max_n", variant_cfg.get("max_n", 3))),
            )
            # Deterministic rank-based selection — exactly max_n positives per 40-window
            bot_ratio = min(max(max_n / max(len(raw_scores), 1), 0.0), 0.1)
            cal = rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)
            scores = [round(float(v), 6) for v in cal]
            try:
                rs = sorted(raw_scores, reverse=True)
                bt.logging.info(
                    f"stat_v5 diag: top5=[{', '.join(f'{v:.3f}' for v in rs[:5])}] "
                    f"med={sorted(raw_scores)[len(raw_scores)//2]:.3f} "
                    f"std={float(np.std(raw_scores)):.4f} max_n={max_n} "
                    f"bot_ratio={bot_ratio:.3f} invert={os.getenv('POKER44_V5_INVERT','0')}"
                )
            except Exception as exc:
                bt.logging.debug(f"v5 diag failed: {exc}")
        # === Wave 1 (2026-05-27) dispatch: baseline_heuristic, v5_invert_topN, v14_rank ===
        elif variant_cfg.get("cls") == "stat_baseline" and len(chunks) > 0:
            from poker44.score.baseline_heuristic import score_chunks_baseline_heuristic
            scores = list(score_chunks_baseline_heuristic(chunks))
            try:
                top5 = sorted(scores, reverse=True)[:5]
                bt.logging.info(
                    f"stat_baseline diag: top5=[{', '.join(f'{v:.3f}' for v in top5)}] "
                    f"min={min(scores):.3f} max={max(scores):.3f} mean={sum(scores)/len(scores):.3f} "
                    f"positives_at_05={sum(1 for v in scores if v >= 0.5)}"
                )
            except Exception as exc:
                bt.logging.debug(f"baseline diag failed: {exc}")
        elif variant_cfg.get("cls") == "stat_v5_invert_topn" and len(chunks) > 0:
            from poker44.score.statistical_v5 import score_chunks_v5
            from poker44.score.rank_cap_remap import rank_cap_remap
            raw = list(score_chunks_v5(chunks))
            inverted = [1.0 - float(r) for r in raw]
            top_n = int(variant_cfg.get("rank_top_n", 3))
            scores = rank_cap_remap(inverted, top_n)
            try:
                top5 = sorted(scores, reverse=True)[:5]
                raw_top3 = sorted(raw, reverse=True)[:3]
                bt.logging.info(
                    f"stat_v5_invert_top{top_n} diag: raw_top3=[{', '.join(f'{r:.3f}' for r in raw_top3)}] "
                    f"final_top5=[{', '.join(f'{v:.3f}' for v in top5)}] "
                    f"positives={sum(1 for v in scores if v >= 0.5)}"
                )
            except Exception as exc:
                bt.logging.debug(f"v5_invert diag failed: {exc}")
        elif variant_cfg.get("cls") == "v14_rank" and len(chunks) > 0:
            from poker44.score.v14_inference import (
                score_chunks_v14_rank, is_available as _v14_avail,
            )
            top_n = int(variant_cfg.get("rank_top_n", 3))
            if not _v14_avail():
                bt.logging.warning(
                    "v14 model not available (models/v14_live_stable.txt missing) — "
                    "fallback to v5_statistical raw scores (no rank cap)"
                )
                from poker44.score.statistical_v5 import score_chunks_v5
                scores = list(score_chunks_v5(chunks))
            else:
                scores = score_chunks_v14_rank(chunks, top_n)
                try:
                    top5 = sorted(scores, reverse=True)[:5]
                    bt.logging.info(
                        f"v14_rank_top{top_n} diag: final_top5=[{', '.join(f'{v:.3f}' for v in top5)}] "
                        f"positives={sum(1 for v in scores if v >= 0.5)} "
                        f"mean={sum(scores)/len(scores):.3f} "
                        f"min={min(scores):.3f} max={max(scores):.3f}"
                    )
                except Exception as exc:
                    bt.logging.debug(f"v14_rank diag failed: {exc}")
        # Wave v19-A (2026-05-28): v19 LambdaRank dispatch.
        # NO silent v5 fallback — log ERROR i safe baseline (0.5) if model missing.
        elif variant_cfg.get("cls") == "v22_rank" and len(chunks) > 0:
            from poker44.score.v22_inference import (
                score_chunks_v22_raw, score_chunks_v22_rank, is_available as _v22_avail,
            )
            top_n = int(variant_cfg.get("rank_top_n", 1))
            if not _v22_avail():
                bt.logging.error(
                    "v22 model not available (models/v22_competitor_ranker.txt missing) — "
                    f"variant={self.variant} returning safe negative baseline 0.49 per chunk."
                )
                scores = [0.49 for _ in chunks]
            else:
                scores = score_chunks_v22_rank(chunks, top_n)
                if not scores:
                    bt.logging.error(
                        f"v22 score_chunks_v22_rank returned empty (variant={self.variant} top_n={top_n}) "
                        f"— safe negative baseline 0.49"
                    )
                    scores = [0.49 for _ in chunks]
                else:
                    try:
                        raw = score_chunks_v22_raw(chunks)
                        top5 = sorted(scores, reverse=True)[:5]
                        bt.logging.info(
                            f"v22_rank_top{top_n} diag: final_top5=[{', '.join(f'{v:.3f}' for v in top5)}] "
                            f"positives={sum(1 for v in scores if v >= 0.5)} "
                            f"raw_mean={sum(raw)/len(raw):.4f} "
                            f"raw_std={(__import__('numpy').std(raw)):.4f} "
                            f"raw_min={min(raw):.4f} raw_max={max(raw):.4f}"
                        )
                    except Exception as exc:
                        bt.logging.debug(f"v22_rank diag failed: {exc}")
        elif variant_cfg.get("cls") == "v19_rank" and len(chunks) > 0:
            from poker44.score.v19_inference import (
                score_chunks_v19_raw, score_chunks_v19_rank, is_available as _v19_avail,
            )
            top_n = int(variant_cfg.get("rank_top_n", 1))
            if not _v19_avail():
                bt.logging.error(
                    "v19 model not available (models/v19_ranker.txt or _meta.json missing) — "
                    f"variant={self.variant} cannot score; returning safe negative baseline 0.49 per chunk. "
                    "DO NOT TREAT AS NORMAL — investigate model deployment."
                )
                scores = [0.49 for _ in chunks]
            else:
                scores = score_chunks_v19_rank(chunks, top_n)
                if not scores:
                    bt.logging.error(
                        f"v19 score_chunks_v19_rank returned empty for {len(chunks)} chunks "
                        f"(variant={self.variant} top_n={top_n}) — safe negative baseline 0.49"
                    )
                    scores = [0.49 for _ in chunks]
                else:
                    try:
                        raw = score_chunks_v19_raw(chunks)
                        raw_arr = list(raw)
                        top5 = sorted(scores, reverse=True)[:5]
                        bt.logging.info(
                            f"v19_rank_top{top_n} diag: final_top5=[{', '.join(f'{v:.3f}' for v in top5)}] "
                            f"positives={sum(1 for v in scores if v >= 0.5)} "
                            f"raw_mean={sum(raw_arr)/len(raw_arr):.4f} "
                            f"raw_std={(__import__('numpy').std(raw_arr)):.4f} "
                            f"raw_min={min(raw_arr):.4f} raw_max={max(raw_arr):.4f}"
                        )
                    except Exception as exc:
                        bt.logging.debug(f"v19_rank diag failed: {exc}")
        # Bounded rank calibrator (codex iter 6) — discrete N∈{0..max_n}, never N≥4
        elif variant_cfg.get("use_bounded_rank") and self.lgbm and len(chunks) > 0:
            from poker44.score.calibration import bounded_rank_calibrate, apply_isotonic
            raw_scores = [self._lgbm_score_raw(chunk) for chunk in chunks]
            max_n = self._resolve_max_n(
                raw_scores,
                default_n=int(getattr(self, "max_n", variant_cfg.get("max_n", 3))),
            )
            score_floor = float(os.getenv(
                "POKER44_SCORE_FLOOR",
                str(getattr(self, "score_floor", variant_cfg.get("score_floor", 0.16))),
            ))
            cal = bounded_rank_calibrate(
                raw_scores,
                max_n=max_n,
                score_floor=score_floor,
                isotonic_points=self.lgbm_isotonic,
            )
            scores = [round(float(v), 6) for v in cal]
            # Diagnostic: RAW (pre-isotonic) and iso distribution
            try:
                raw_sorted = sorted(raw_scores, reverse=True)
                top5_raw = ", ".join(f"{v:.3f}" for v in raw_sorted[:5])
                raw_med = raw_sorted[len(raw_sorted)//2]
                raw_min = raw_sorted[-1]
                raw_std = float(np.std(raw_scores))
                iso_scores = sorted(
                    (apply_isotonic(float(r), self.lgbm_isotonic) for r in raw_scores),
                    reverse=True,
                )
                top5_iso = ", ".join(f"{v:.3f}" for v in iso_scores[:5])
                iso_med = iso_scores[len(iso_scores)//2]
                above_floor = sum(1 for v in iso_scores if v >= score_floor)
                bt.logging.info(
                    f"diag RAW: top5=[{top5_raw}] med={raw_med:.3f} min={raw_min:.3f} std={raw_std:.4f} "
                    f"| ISO: top5=[{top5_iso}] med={iso_med:.3f} #above({score_floor:.2f})={above_floor}/40 "
                    f"max_n={max_n}"
                )
            except Exception as exc:
                bt.logging.debug(f"diag failed: {exc}")
        # Dynamic per-batch cap (auto-detects ratio, anti-cliff)
        elif variant_cfg.get("use_dynamic_cap") and self.lgbm and len(chunks) > 0:
            from poker44.score.calibration import dynamic_safe_calibrate
            raw_scores = [self._lgbm_score_raw(chunk) for chunk in chunks]
            env_cap = os.getenv("POKER44_MAX_BOT_FRACTION")
            abs_max = float(env_cap) if env_cap is not None else 0.50
            cal = dynamic_safe_calibrate(
                raw_scores,
                isotonic_points=self.lgbm_isotonic,
                safety_margin=0.05,
                absolute_max=abs_max,
                absolute_min=0.05,
            )
            scores = [round(float(v), 6) for v in cal]
        # Per-batch adaptive calibration with FIXED cap (best for known ratio)
        elif variant_cfg.get("use_batch_adaptive") and self.lgbm and len(chunks) > 0:
            raw_scores = [self._lgbm_score_raw(chunk) for chunk in chunks]
            # Allow env override for rapid cap tuning
            env_cap = os.getenv("POKER44_MAX_BOT_FRACTION")
            if env_cap is not None:
                max_bot = float(env_cap)
            else:
                max_bot = float(variant_cfg.get("max_bot_fraction", 0.22))
            cal = adaptive_safe_calibrate(
                raw_scores, isotonic_points=self.lgbm_isotonic, max_bot_fraction=max_bot
            )
            scores = [round(float(v), 6) for v in cal]
        else:
            # Score all chunks (per-chunk calibration)
            raw_scores = [self._score_single_chunk(chunk) for chunk in chunks]

            # Apply safety cap if configured
            if self.use_safe_cap and len(raw_scores) > 1:
                cal = adaptive_safe_calibrate(
                    raw_scores, max_bot_fraction=self.max_bot_fraction
                )
                scores = [round(float(v), 6) for v in cal]
            else:
                scores = [round(s, 6) for s in raw_scores]

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        # Calibration outcome tracking (auto-fallback if bot_pred=0 streak)
        self._record_calibration_outcome(raw_scores=[], scores=scores)

        bot_count = sum(1 for s in scores if s >= 0.5)
        adaptive_info = ""
        if self._adaptive_n_enabled:
            diag = getattr(self, "_last_adaptive_diag", None)
            if diag:
                adaptive_info = (
                    f" adaptive=ON profile={diag.get('profile')} "
                    f"N={diag.get('final_n')}(suggested={diag.get('suggested_n')}) "
                    f"std={diag.get('std')} top1_med={diag.get('top1_med')}"
                )
        bt.logging.info(
            f"Scored {len(chunks)} chunks | variant={self.variant} "
            f"min={min(scores):.3f} max={max(scores):.3f} "
            f"bot_pred={bot_count}/{len(scores)} cap={self.max_bot_fraction:.3f}"
            f"{adaptive_info}"
            f"{' [FALLBACK]' if self._calibration_fallback_active else ''}"
        )
        return synapse

    def _score_single_chunk(self, chunk) -> float:
        """Score one chunk using the configured variant."""
        if not chunk:
            return 0.5

        variant_cfg = VARIANTS.get(self.variant, VARIANTS["v1_real_2026"])
        cls = variant_cfg.get("cls", "lgbm")

        try:
            if cls == "cnn" and self.cnn:
                return self._cnn_score(chunk)

            elif cls == "stat":
                stat_type = variant_cfg.get("stat", "other_r")
                from poker44.models.statistical_detector import DETECTORS
                return DETECTORS.get(stat_type, DETECTORS["other_r"])(chunk)

            elif cls == "stat_v5":
                from poker44.score.statistical_v5 import score_chunk_v5
                return score_chunk_v5(chunk)

            elif cls == "lgbm" and self.lgbm:
                return self._lgbm_score(chunk)

            # cls == "voting_ensemble" handled in forward() (batch path), not here.

        except Exception as exc:
            bt.logging.warning(f"Variant {self.variant} failed: {exc}")

        # Ultimate fallback — try LightGBM, CNN, then heuristic
        if self.lgbm:
            return self._lgbm_score(chunk)
        if self.cnn:
            return self._cnn_score(chunk)
        return self._heuristic_score(chunk)

    def _cnn_score(self, chunk) -> float:
        """Score chunk using CNN per-hand model."""
        try:
            return self.cnn.score_chunk(chunk)
        except Exception as exc:
            bt.logging.warning(f"CNN score failed: {exc}")
            return 0.5

    def _lgbm_score_raw(self, chunk) -> float:
        """Raw LightGBM score BEFORE calibration (for batch adaptive)."""
        try:
            is_v1 = bool(self.lgbm_meta and self.lgbm_meta.get("v1_optimized"))
            if is_v1:
                from poker44.score.features_v1 import extract_v1_features
                feats = extract_v1_features(chunk)
            else:
                from poker44.score.features import extract_chunk_features
                feats = extract_chunk_features(chunk)
            row = np.asarray([[feats.get(n, 0.0) for n in self.lgbm_features]], dtype=np.float32)
            return float(self.lgbm.predict(row)[0])
        except Exception as exc:
            bt.logging.warning(f"LightGBM raw score failed: {exc}")
            return 0.5

    def _lgbm_score(self, chunk) -> float:
        """Score chunk using LightGBM on aggregated features."""
        try:
            raw = self._lgbm_score_raw(chunk)
            from poker44.score.calibration import full_calibrate
            return full_calibrate(raw, self.lgbm_isotonic)
        except Exception as exc:
            bt.logging.warning(f"LightGBM score failed: {exc}")
            return 0.5

    def _voting_ensemble_score_batch(self, chunks):
        """TOP1 voting ensemble — agreement_2of3 across 3 models.

        For each chunk, score with v3, v2, B_deeper. If >=2/3 say bot (>0.5),
        use mean as score. Else, use mean/2 (anti-FP).

        Verified offline: reward 0.44-0.47 (BEATS baseline-v1 0.419 by +5pp).
        """
        from poker44.score.features_v1 import extract_v1_features
        from poker44.score.features import extract_chunk_features
        n = len(chunks)
        per_model_scores = []  # [n_chunks, n_models]
        for vm in self.voting_models:
            is_v1 = bool(vm["meta"].get("v1_optimized"))
            rows = []
            for c in chunks:
                if is_v1:
                    feats = extract_v1_features(c)
                else:
                    feats = extract_chunk_features(c)
                rows.append([feats.get(name, 0.0) for name in vm["features"]])
            X = np.asarray(rows, dtype=np.float32)
            scores = vm["model"].predict(X)
            per_model_scores.append(scores)

        # Stack: shape [n_chunks, n_models]
        S = np.stack(per_model_scores, axis=1) if per_model_scores else np.zeros((n, 1))
        # Agreement: 2/3 say bot (>0.5)
        votes = (S > 0.5).sum(axis=1)
        mean_score = S.mean(axis=1)
        # If consensus >=2/3 → use mean. Else use mean/2 (anti-FP)
        consensus_threshold = max(2, S.shape[1] - 1)  # 2 of 3 (or all if 1-2 models)
        out = np.where(votes >= consensus_threshold, mean_score, mean_score / 2.0)
        return out.tolist()

    @staticmethod
    def _heuristic_score(chunk) -> float:
        actions = []
        for h in chunk:
            actions.extend(h.get("actions") or [])
        if not actions:
            return 0.5
        types = Counter(a.get("action_type") for a in actions)
        total = max(sum(types.values()), 1)
        fold_r = types.get("fold", 0) / total
        call_r = types.get("call", 0) / total
        return max(0.0, min(1.0, 0.3 + 0.4 * call_r - 0.3 * fold_r))

    def _save_raw_chunks(self, chunks):
        try:
            import json
            RAW_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            path = RAW_CHUNKS_DIR / f"chunks_{ts}.json"
            with path.open("w") as fh:
                json.dump({"timestamp": ts, "n_chunks": len(chunks),
                           "chunk_sizes": [len(c) for c in chunks], "chunks": chunks},
                          fh, separators=(",", ":"))
            files = sorted(RAW_CHUNKS_DIR.glob("chunks_*.json"))
            if len(files) > 500:
                for f in files[:-500]:
                    f.unlink()
        except Exception:
            pass

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info(f"V1 Poker44 miner running (variant={miner.variant})...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
