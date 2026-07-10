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
- v136_live_<strategy>_top<N>
- v140_multi_<strategy>_top<N>
- v142_rankblend_<strategy>_top<N>
- v173_<strategy>_top<N>
- v175_<strategy>_top<N>
- v179_<strategy>_top<N>
- v181_<strategy>_top<N>
- v183_<strategy>_top<N>
- v184_<strategy>_top<N>
- v193_<strategy>_top<N>
- v190_<strategy>_top<N>
- v209_<strategy>_top<N>
- v216_<strategy>_top<N>
- v217_<strategy>_top<N>
- v218_<strategy>_top<N>
- v219_rebuilt_superv2_top<N>
- v220_v11lock2_v219rest_top<N>
- v223_<strategy>_top<N>
- v228_<strategy>_top<N>
- v234_<strategy>_top<N>
- v237_<strategy>_top<N>
- v241_<strategy>_top<N>
- v245_<strategy>_top<N>
- v248_<strategy>_top<N>
- v249_<strategy>_top<N>
- v252_<strategy>_top<N>
- v253_<strategy>_top<N>
- v255_<strategy>_top<N>
- v257_<strategy>_top<N>
- v260_<strategy>_top<N>
- v262_w95_<strategy>_top<N>
- v264_<strategy>_top<N>
- v270_<strategy>_top<N>
- v288_<strategy>_top<N>
- v292_<strategy>_top<N>
- v289_<strategy>_top<N>
- v290_runtime_top<N>
- v287_<strategy>_top<N>
- v297_<strategy>_top<N>
- v298_<strategy>_top<N>
- v274_<strategy>_top<N>
- v276_<strategy>_top<N>
- v277_<strategy>_top<N>
- v280_<strategy>_top<N>
- v294_<strategy>_top<N>
- v200_<strategy>_top<N>

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
    score_extra: dict[str, Any] | None = None,
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
        if score_extra:
            record["score_extra"] = score_extra
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


def _banded_rank_remap(
    raw_scores: list[float] | np.ndarray,
    top_n: int,
    *,
    positive_hi: float = 0.509,
    positive_lo: float = 0.501,
    negative_hi: float = 0.49,
    negative_lo: float = 0.0,
) -> list[float]:
    """Top-k threshold-safe remap with a narrow positive score band.

    This keeps rank order unchanged while matching the current public-leader
    operational shape: exactly top_n chunks cross 0.5, but positives stay close
    to the threshold and negatives occupy [0, 0.49].
    """
    arr = np.asarray(raw_scores, dtype=float)
    n = int(arr.size)
    if n <= 0:
        return []
    top_n = max(0, min(int(top_n), n))
    order = np.argsort(-arr, kind="mergesort")
    out = np.zeros(n, dtype=float)
    if top_n <= 0:
        for rank, idx in enumerate(order):
            frac = rank / max(n - 1, 1)
            out[int(idx)] = negative_hi - frac * (negative_hi - negative_lo)
        return [float(np.clip(v, 0.0, 1.0)) for v in out]

    positive_span = max(top_n - 1, 1)
    negative_count = n - top_n
    for rank, idx in enumerate(order):
        if rank < top_n:
            frac = rank / positive_span
            out[int(idx)] = positive_hi - frac * (positive_hi - positive_lo)
        else:
            frac = (rank - top_n) / max(negative_count - 1, 1)
            out[int(idx)] = negative_hi - frac * (negative_hi - negative_lo)
    return [float(np.clip(v, 0.0, 1.0)) for v in out]


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
    v131_behav = False
    v132_ngram = False
    v133_split = False
    v136_live = False
    v140_multi = False
    v142_rankblend = False
    v173_actioncap8 = False
    v175_actioncap8_ks075 = False
    v179_actioncap8_livehand89_ks060 = False
    v181_actioncap8_livehand89_maxks075_fullheads = False
    v183_v11lock1_v181rest = False
    v184_v11lock2_v181rest = False
    v193_v11lock1_v145rank_rest = False
    v190_contract60_80_100_ks050_livesized = False
    v209_served_rankcap_ks055 = False
    v216_served_rankcap_ks060 = False
    v217_v11lock2_v216rest = False
    v218_withinbatch_behav = False
    v219_rebuilt_superv2 = False
    v220_v11lock2_v219rest = False
    v221_v11lock2_nightlyrest = False
    v223_withinbatch_behav_refresh = False
    v228_shaped_v223 = False
    v234_behav_mix_v11 = False
    v237_v11lock1_v234rest = False
    v241_v11lock1_v118rest = False
    v245_v11lock1_v244rest = False
    v248_batchrank_schema = False
    v249_batchrank_behavmix_v11 = False
    v252_clean_top1recipe_schema = False
    v253_oldwindow_schema = False
    v255_oldwindow_top1schema = False
    v257_trainonly_top1schema = False
    v260_fit80_sanitized_top1schema = False
    v262_v260w95_v11w5_rankblend = False
    v264_v260w80_v263w20_rankblend = False
    v270_v260w98_v263w01_v265w01_rankblend = False
    v291_v263_rankmean_latest = False
    v288_top1style_schema = False
    v292_v26090_v26105_v11_rankblend = False
    v289_v270w90_v288397avgw10_rankblend = False
    v290_v289w90_v11w10_runtime = False
    v287_shape_adaptive_v11_v270 = False
    v297_shape_adaptive_v296_v11_v270 = False
    v298_lambdamart_wide_ranker = False
    v306_lambdamart_temporal_seed_ensemble = False
    v315_original_lambdamart_top10 = False
    v271_v11lock1_v268rest = False
    v274_v11lock1_v273rest = False
    v276_livesized6080100_v273 = False
    v277_livesized6080100_temporal = False
    v280_livesized6080100_temporal_consistency = False
    v294_hg2_rebuild = False
    v200_stackseq_last3 = False
    v201_stackseq_wide8 = False
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
    elif name.startswith("v132_ngram_"):
        prefix = "v132_ngram_"
        live_sized = True
        v132_ngram = True
    elif name.startswith("v133_split_"):
        prefix = "v133_split_"
        live_sized = True
        v133_split = True
    elif name.startswith("v136_live_"):
        prefix = "v136_live_"
        live_sized = True
        v136_live = True
    elif name.startswith("v140_multi_"):
        prefix = "v140_multi_"
        live_sized = True
        v140_multi = True
    elif name.startswith("v142_rankblend_"):
        prefix = "v142_rankblend_"
        live_sized = True
        v142_rankblend = True
    elif name.startswith("v173_"):
        prefix = "v173_"
        live_sized = True
        v173_actioncap8 = True
    elif name.startswith("v175_"):
        prefix = "v175_"
        live_sized = True
        v175_actioncap8_ks075 = True
    elif name.startswith("v179_"):
        prefix = "v179_"
        live_sized = True
        v179_actioncap8_livehand89_ks060 = True
    elif name.startswith("v181_"):
        prefix = "v181_"
        live_sized = True
        v181_actioncap8_livehand89_maxks075_fullheads = True
    elif name.startswith("v183_"):
        prefix = "v183_"
        live_sized = True
        v183_v11lock1_v181rest = True
    elif name.startswith("v184_"):
        prefix = "v184_"
        live_sized = True
        v184_v11lock2_v181rest = True
    elif name.startswith("v193_"):
        prefix = "v193_"
        live_sized = True
        v193_v11lock1_v145rank_rest = True
    elif name.startswith("v190_"):
        prefix = "v190_"
        live_sized = True
        v190_contract60_80_100_ks050_livesized = True
    elif name.startswith("v209_"):
        prefix = "v209_"
        live_sized = True
        v209_served_rankcap_ks055 = True
    elif name.startswith("v216_"):
        prefix = "v216_"
        live_sized = True
        v216_served_rankcap_ks060 = True
    elif name.startswith("v217_"):
        prefix = "v217_"
        live_sized = True
        v217_v11lock2_v216rest = True
    elif name.startswith("v218_"):
        prefix = "v218_"
        live_sized = True
        v218_withinbatch_behav = True
    elif name.startswith("v219_"):
        prefix = "v219_"
        live_sized = True
        v219_rebuilt_superv2 = True
    elif name.startswith("v220_"):
        prefix = "v220_"
        live_sized = True
        v220_v11lock2_v219rest = True
    elif name.startswith("v221_"):
        prefix = "v221_"
        live_sized = True
        v221_v11lock2_nightlyrest = True
    elif name.startswith("v223_"):
        prefix = "v223_"
        live_sized = True
        v223_withinbatch_behav_refresh = True
    elif name.startswith("v228_"):
        prefix = "v228_"
        live_sized = True
        v228_shaped_v223 = True
    elif name.startswith("v234_"):
        prefix = "v234_"
        live_sized = True
        v234_behav_mix_v11 = True
    elif name.startswith("v237_"):
        prefix = "v237_"
        live_sized = True
        v237_v11lock1_v234rest = True
    elif name.startswith("v241_"):
        prefix = "v241_"
        live_sized = True
        v241_v11lock1_v118rest = True
    elif name.startswith("v245_"):
        prefix = "v245_"
        live_sized = True
        v245_v11lock1_v244rest = True
    elif name.startswith("v248_"):
        prefix = "v248_"
        live_sized = True
        v248_batchrank_schema = True
    elif name.startswith("v249_"):
        prefix = "v249_"
        live_sized = True
        v249_batchrank_behavmix_v11 = True
    elif name.startswith("v252_"):
        prefix = "v252_"
        live_sized = True
        v252_clean_top1recipe_schema = True
    elif name.startswith("v253_"):
        prefix = "v253_"
        live_sized = True
        v253_oldwindow_schema = True
    elif name.startswith("v255_"):
        prefix = "v255_"
        live_sized = True
        v255_oldwindow_top1schema = True
    elif name.startswith("v257_"):
        prefix = "v257_"
        live_sized = True
        v257_trainonly_top1schema = True
    elif name.startswith("v260_"):
        prefix = "v260_"
        live_sized = True
        v260_fit80_sanitized_top1schema = True
    elif name.startswith("v262_w95_"):
        prefix = "v262_w95_"
        live_sized = True
        v262_v260w95_v11w5_rankblend = True
    elif name.startswith("v264_"):
        prefix = "v264_"
        live_sized = True
        v264_v260w80_v263w20_rankblend = True
    elif name.startswith("v270_"):
        prefix = "v270_"
        live_sized = True
        v270_v260w98_v263w01_v265w01_rankblend = True
    elif name.startswith("v291_"):
        prefix = "v291_"
        live_sized = True
        v291_v263_rankmean_latest = True
    elif name.startswith("v288_"):
        prefix = "v288_"
        live_sized = True
        v288_top1style_schema = True
    elif name.startswith("v292_"):
        prefix = "v292_"
        live_sized = True
        v292_v26090_v26105_v11_rankblend = True
    elif name.startswith("v289_"):
        prefix = "v289_"
        live_sized = True
        v289_v270w90_v288397avgw10_rankblend = True
    elif name.startswith("v290_"):
        prefix = "v290_"
        live_sized = True
        v290_v289w90_v11w10_runtime = True
    elif name.startswith("v287_"):
        prefix = "v287_"
        live_sized = True
        v287_shape_adaptive_v11_v270 = True
    elif name.startswith("v297_"):
        prefix = "v297_"
        live_sized = True
        v297_shape_adaptive_v296_v11_v270 = True
    elif name.startswith("v298_"):
        prefix = "v298_"
        live_sized = True
        v298_lambdamart_wide_ranker = True
    elif name.startswith("v306_"):
        prefix = "v306_"
        live_sized = True
        v306_lambdamart_temporal_seed_ensemble = True
    elif name.startswith("v315_"):
        prefix = "v315_"
        live_sized = True
        v315_original_lambdamart_top10 = True
    elif name.startswith("v271_"):
        prefix = "v271_"
        live_sized = True
        v271_v11lock1_v268rest = True
    elif name.startswith("v274_"):
        prefix = "v274_"
        live_sized = True
        v274_v11lock1_v273rest = True
    elif name.startswith("v276_"):
        prefix = "v276_"
        live_sized = True
        v276_livesized6080100_v273 = True
    elif name.startswith("v277_"):
        prefix = "v277_"
        live_sized = True
        v277_livesized6080100_temporal = True
    elif name.startswith("v280_"):
        prefix = "v280_"
        live_sized = True
        v280_livesized6080100_temporal_consistency = True
    elif name.startswith("v294_"):
        prefix = "v294_"
        live_sized = True
        v294_hg2_rebuild = True
    elif name.startswith("v200_"):
        prefix = "v200_"
        live_sized = True
        v200_stackseq_last3 = True
    elif name.startswith("v201_"):
        prefix = "v201_"
        live_sized = True
        v201_stackseq_wide8 = True

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
            "cat_shape": "cat_shape",
            "lgb_shape": "lgb_shape",
            "xgb_shape": "xgb_shape",
            "et_shape": "et_shape",
            "avg": "avg",
            "raw": "raw",
            "precal": "raw",
            "cal": "calibrated",
            "calibrated": "calibrated",
            "final": "final",
            "hg2": "hg2_rank_budget",
            "runtime": "v290_runtime",
        }
        if strategy_part.startswith(("avg_no_", "avg_w", "blend_", "seg", "ladder_", "banded_")) or strategy_part in {
            "v125_weighted",
        }:
            strategy = strategy_part
        else:
            strategy = strategy_aliases.get(strategy_part, "rank_mean")
        family = (
            "v184_v11lock2_v181rest"
            if v184_v11lock2_v181rest
            else "v193_v11lock1_v145rank_rest"
            if v193_v11lock1_v145rank_rest
            else "v190_contract60_80_100_ks050_livesized"
            if v190_contract60_80_100_ks050_livesized
            else "v209_served_rankcap_ks055"
            if v209_served_rankcap_ks055
            else "v216_served_rankcap_ks060"
            if v216_served_rankcap_ks060
            else "v217_v11lock2_v216rest"
            if v217_v11lock2_v216rest
            else "v218_withinbatch_behav"
            if v218_withinbatch_behav
            else "v219_rebuilt_superv2"
            if v219_rebuilt_superv2
            else "v220_v11lock2_v219rest"
            if v220_v11lock2_v219rest
            else "v221_v11lock2_nightlyrest"
            if v221_v11lock2_nightlyrest
            else "v223_withinbatch_behav_refresh"
            if v223_withinbatch_behav_refresh
            else "v228_shaped_v223"
            if v228_shaped_v223
            else "v234_behav_mix_v11"
            if v234_behav_mix_v11
            else "v237_v11lock1_v234rest"
            if v237_v11lock1_v234rest
            else "v241_v11lock1_v118rest"
            if v241_v11lock1_v118rest
            else "v245_v11lock1_v244rest"
            if v245_v11lock1_v244rest
            else "v248_batchrank_schema"
            if v248_batchrank_schema
            else "v249_batchrank_behavmix_v11"
            if v249_batchrank_behavmix_v11
            else "v252_clean_top1recipe_schema"
            if v252_clean_top1recipe_schema
            else "v253_oldwindow_schema"
            if v253_oldwindow_schema
            else "v255_oldwindow_top1schema"
            if v255_oldwindow_top1schema
            else "v257_trainonly_top1schema"
            if v257_trainonly_top1schema
            else "v260_fit80_sanitized_top1schema"
            if v260_fit80_sanitized_top1schema
            else "v262_v260w95_v11w5_rankblend"
            if v262_v260w95_v11w5_rankblend
            else "v264_v260w80_v263w20_rankblend"
            if v264_v260w80_v263w20_rankblend
            else "v270_v260w98_v263w01_v265w01_rankblend"
            if v270_v260w98_v263w01_v265w01_rankblend
            else "v291_v263_rankmean_latest"
            if v291_v263_rankmean_latest
            else "v288_top1style_schema"
            if v288_top1style_schema
            else "v292_v26090_v26105_v11_rankblend"
            if v292_v26090_v26105_v11_rankblend
            else "v289_v270w90_v288397avgw10_rankblend"
            if v289_v270w90_v288397avgw10_rankblend
            else "v290_v289w90_v11w10_runtime"
            if v290_v289w90_v11w10_runtime
            else "v287_shape_adaptive_v11_v270"
            if v287_shape_adaptive_v11_v270
            else "v297_shape_adaptive_v296_v11_v270"
            if v297_shape_adaptive_v296_v11_v270
            else "v298_lambdamart_wide_ranker"
            if v298_lambdamart_wide_ranker
            else "v306_lambdamart_temporal_seed_ensemble"
            if v306_lambdamart_temporal_seed_ensemble
            else "v315_original_lambdamart_top10"
            if v315_original_lambdamart_top10
            else "v271_v11lock1_v268rest"
            if v271_v11lock1_v268rest
            else "v274_v11lock1_v273rest"
            if v274_v11lock1_v273rest
            else "v276_livesized6080100_v273"
            if v276_livesized6080100_v273
            else "v277_livesized6080100_temporal"
            if v277_livesized6080100_temporal
            else "v280_livesized6080100_temporal_consistency"
            if v280_livesized6080100_temporal_consistency
            else "v294_hg2_rebuild"
            if v294_hg2_rebuild
            else "v200_stackseq_last3"
            if v200_stackseq_last3
            else "v201_stackseq_wide8"
            if v201_stackseq_wide8
            else "v183_v11lock1_v181rest"
            if v183_v11lock1_v181rest
            else "v181_actioncap8_livehand89_maxks075_fullheads"
            if v181_actioncap8_livehand89_maxks075_fullheads
            else "v179_actioncap8_livehand89_ks060"
            if v179_actioncap8_livehand89_ks060
            else "v175_actioncap8_ks075"
            if v175_actioncap8_ks075
            else "v173_actioncap8"
            if v173_actioncap8
            else "v142_rankblend"
            if v142_rankblend
            else "v140_multi"
            if v140_multi
            else "v136_live"
            if v136_live
            else "v133_split"
            if v133_split
            else "v132_ngram"
            if v132_ngram
            else "v131_behav"
            if v131_behav
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
                    if v131_behav
                    else "Live-shaped full-head behavioural n-gram rank ensemble with segment-capable serve strategy, max-8 action parity, and worst-live-payload KS<=0.75 feature filtering."
                    if v181_actioncap8_livehand89_maxks075_fullheads
                    else "Live-shaped behavioural n-gram ranker trained on public benchmark chunks with 60/80/100 chunk contracts, max-8 action parity, and worst-live-payload KS<=0.50 feature filtering."
                    if v190_contract60_80_100_ks050_livesized
                    else "Served-rankcap-selected live-shaped behavioural n-gram ranker trained on public benchmark chunks with 60/80/100 chunk contracts and worst-live-payload KS<=0.55 feature filtering."
                    if v209_served_rankcap_ks055
                    else "Served-rankcap-selected live-shaped behavioural n-gram ranker trained on public benchmark chunks with 60/80/100 chunk contracts and worst-live-payload KS<=0.60 feature filtering."
                    if v216_served_rankcap_ks060
                    else "v11 top-2 locked behavioural anchor with v216 served-rankcap KS<=0.60 ranker ordering the remaining chunks."
                    if v217_v11lock2_v216rest
                    else "Within-batch behavioural ranker trained on public benchmark v1.13 live-sized units and selected by served rank-cap official reward."
                    if v218_withinbatch_behav
                    else "Local rebuild of public super-v2 stacked architecture trained on Poker44 public v1.13 miner-visible benchmark data."
                    if v219_rebuilt_superv2
                    else "v11 top-2 locked behavioural anchor with rebuilt super-v2 ranking the remaining chunks."
                    if v220_v11lock2_v219rest
                    else "v11 top-2 locked behavioural anchor with the 2026-07-09 super-sequence stack ranking the remaining chunks."
                    if v221_v11lock2_nightlyrest
                    else "Daily refreshed within-batch behavioural ranker trained on public benchmark v1.13 live-sized units through 2026-07-09 and selected by served rank-cap reward."
                    if v223_withinbatch_behav_refresh
                    else "Daily refreshed within-batch behavioural ranker with a monotone shaped score head for the validator's fixed 0.5 cutoff."
                    if v228_shaped_v223
                    else "Daily refreshed within-batch behavioural ranker augmented with deterministic v11 telemetry features and served through a rank-ladder strategy selected by live-sized stress robustness."
                    if v234_behav_mix_v11
                    else "v11 top-1 locked behavioural anchor with v234 behavioural telemetry rank-ladder ordering the remaining chunks."
                    if v237_v11lock1_v234rest
                    else "v11 top-1 locked behavioural anchor with the v118 stable live-sized ranker ordering the remaining chunks."
                    if v241_v11lock1_v118rest
                    else "v11 top-1 locked behavioural anchor with the v244 clean-restart schema ranker ordering the remaining chunks."
                    if v245_v11lock1_v244rest
                    else "Within-batch feature-rank schema ensemble trained on public v1.13 benchmark chunks with the same 100-chunk batch-rank transform at serve time."
                    if v248_batchrank_schema
                    else "Batch-rank behav-mix-v11 schema ensemble trained on public benchmark chunks with the same 100-chunk batch-rank transform at serve time."
                    if v249_batchrank_behavmix_v11
                    else "Clean-restart robust schema ranker trained on latest public benchmark chunks and served with rank-space ordering over 100-chunk validator payloads."
                    if v252_clean_top1recipe_schema
                    else "Old-window robust schema ranker trained on the stable 2026-06-03..2026-07-02 public benchmark window and served with rank-space ordering over validator payloads."
                    if v253_oldwindow_schema
                    else "Old-window 293-feature schema ranker trained with top1-style tree hyperparameters on the stable 2026-06-03..2026-07-02 public benchmark window and served with rank-space ordering over validator payloads."
                    if v255_oldwindow_top1schema
                    else "Train-split-only 293-feature schema ranker trained with top1-style tree hyperparameters on 2026-06-03..2026-06-30, holding 2026-07-01..2026-07-02 out, and served with a rank-ladder strategy over validator payloads."
                    if v257_trainonly_top1schema
                    else "Fit-split 293-feature schema ranker trained on miner-visible public benchmark data with 2026-07-01..2026-07-02 held out and served through a rank-ladder strategy over validator payloads."
                    if v260_fit80_sanitized_top1schema
                    else "Rank-space blend of the fit-split 293-feature schema ranker and the independent v11 behavioural scorer, served through a rank-ladder strategy over validator payloads."
                    if v262_v260w95_v11w5_rankblend
                    else "Rank-space blend of v260 ET and the latest-public-benchmark v263 schema ranker, served through a rank-ladder strategy over validator payloads."
                    if v264_v260w80_v263w20_rankblend
                    else "Rank-space blend of v260 ET with small v263 and v265 schema-anomaly components, selected by live-topology replay against current public leaders."
                    if v270_v260w98_v263w01_v265w01_rankblend
                    else "Fresh 397-feature robust-schema top1-style ranker trained on public v1.13 benchmark sourceDates through 2026-07-10 and served with a disclosed top-k threshold-safe strategy over validator payloads."
                    if v288_top1style_schema
                    else "Rank-space blend of v260 ET, independent-seed v261 ET, and a small deterministic v11 behavioural stabilizer selected by live-payload topology replay against current public leaders."
                    if v292_v26090_v26105_v11_rankblend
                    else "Rank-space blend of the v270 UID99-like schema blend and a small v288 top1-style robust-schema branch, selected by live-payload topology replay."
                    if v289_v270w90_v288397avgw10_rankblend
                    else "Runtime rank-space blend of v289 UID99-like schema scoring and the independent v11 behavioural scorer, selected by held-out stress replay."
                    if v290_v289w90_v11w10_runtime
                    else "Shape-adaptive scorer that routes lower-preflop validator batches to the deterministic v11 behavioural top-2 anchor and high-preflop batches to the v270 rank-ladder top-20 schema blend."
                    if v287_shape_adaptive_v11_v270
                    else "Three-branch shape-adaptive scorer selected by live topology replay: v296 wide per-batch-rank PCA/MLP for D8-like batches, v11 behavioural anchor for middle-low-preflop batches, and v270 UID99-like schema rank-ladder for high-preflop batches."
                    if v297_shape_adaptive_v296_v11_v270
                    else "LightGBM LambdaMART wide-feature ranker trained on current public v1.13 benchmark sourceDates through 2026-07-10 with miner-visible HG2 wide features and served through a narrow top-k threshold-safe ranking head."
                    if v298_lambdamart_wide_ranker
                    else "Temporal seed ensemble of two independently trained LambdaMART wide-feature rankers; weights selected on July 6-7 and verified on untouched July 8-10 batches."
                    if v306_lambdamart_temporal_seed_ensemble
                    else "Independent AceGuard LambdaMART ranker using original identity-free action, response, pot, position, stack, and order-invariant distribution features with a low-prevalence-selected top-10 head."
                    if v315_original_lambdamart_top10
                    else "v11 top-1 locked behavioural anchor with the fresh 2026-07-10 v268 robust-schema ranker ordering the remaining chunks."
                    if v271_v11lock1_v268rest
                    else "v11 top-1 locked behavioural anchor with a v273 live-sized 60/80/100 public-benchmark ranker ordering the remaining chunks."
                    if v274_v11lock1_v273rest
                    else "Live-sized 60/80/100 public-benchmark super_seq ranker served as a narrow top1 candidate."
                    if v276_livesized6080100_v273
                    else "Live-sized 60/80/100 public-benchmark super_seq ranker with temporal lag, trend, action-bigram, and street-share schema features."
                    if v277_livesized6080100_temporal
                    else "Live-sized 60/80/100 public-benchmark super_seq_temporal ranker with temporal consistency, quartile drift, action-bigram, street-share, and bet/pot clustering features."
                    if v280_livesized6080100_temporal_consistency
                    else "HG2 weighted-rank blend trained on public v1.13 benchmark releases through 2026-07-10 with live-size pooled augmentation, monotone GBM, stacked tree ensemble, PCA-MLP member, and rank-preserving safety budget."
                    if v294_hg2_rebuild
                    else "Wider stacked tree and chunk-sequence model trained on latest public benchmark releases with miner-visible payload fields only."
                    if v201_stackseq_wide8
                    else "Stacked tree and chunk-sequence model trained on latest public benchmark releases with miner-visible payload fields only."
                    if v200_stackseq_last3
                    else "v11 top-2 locked behavioural anchor with v181 segment ranker ordering the remaining chunks."
                    if v184_v11lock2_v181rest
                    else "v11 top-1 locked behavioural anchor with v145 human-corpus ranker ordering the remaining chunks."
                    if v193_v11lock1_v145rank_rest
                    else "v11 top-1 locked behavioural anchor with v181 segment ranker ordering the remaining chunks."
                    if v183_v11lock1_v181rest
                    else "Live-shaped action-capped behavioural n-gram ranker with KS<=0.60 multi-live stability filtering and 80-100 hand training units."
                    if v179_actioncap8_livehand89_ks060
                    else "Live-sized action-capped behavioural n-gram ranker with KS<=0.75 multi-live stability filtering and max-8 action parity."
                    if v175_actioncap8_ks075
                    else "Live-sized action-capped behavioural n-gram ranker trained on public v1.13 benchmark chunks with train/serve max-8 action parity."
                    if v173_actioncap8
                    else "Rank-space blend of independently gated live-sized benchmark rankers."
                    if v142_rankblend
                    else "Multi-seed live-sized behavioural n-gram ranker trained on public v1.13 benchmark chunks through 2026-07-07."
                    if v140_multi
                    else "Live-sized v1.13 supervised schema model trained through 2026-07-07 with live-shape stability gating."
                    if v136_live
                    else "Poker44 v1.13 split-aware behavioural n-gram stack trained on public train split with validation held out."
                    if v133_split
                    else "Live-sized behavioural rank stack with sparse action n-gram side learner."
                    if v132_ngram
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
            "strategy": (
                "shape_adaptive"
                if v297_shape_adaptive_v296_v11_v270
                else "ladder_rank_mean"
                if v287_shape_adaptive_v11_v270
                else strategy
            ),
            "default_top_n": max(1, top_n),
            "model_file": (
                "data/models/v142_rankblend/model.pkl"
                if v142_rankblend
                else "data/models/v184_v11lock2_v181rest/model.pkl"
                if v184_v11lock2_v181rest
                else "data/models/v193_v11lock1_v145rank_rest/model.pkl"
                if v193_v11lock1_v145rank_rest
                else "data/models/v190_contract60_80_100_ks050_livesized/model.pkl"
                if v190_contract60_80_100_ks050_livesized
                else "data/models/v209_served_rankcap_ks055/model.pkl"
                if v209_served_rankcap_ks055
                else "data/models/v216_served_rankcap_ks060/model.pkl"
                if v216_served_rankcap_ks060
                else "data/models/v217_v11lock2_v216rest/model.pkl"
                if v217_v11lock2_v216rest
                else "data/models/v218_withinbatch_behav/model.pkl"
                if v218_withinbatch_behav
                else "data/models/v219_rebuilt_superv2/model.joblib"
                if v219_rebuilt_superv2
                else "data/models/v219_rebuilt_superv2/model.joblib"
                if v220_v11lock2_v219rest
                else "data/models/v221_v11lock2_nightlyrest/model.pkl"
                if v221_v11lock2_nightlyrest
                else "data/models/v223_withinbatch_behav_refresh/model.pkl"
                if v223_withinbatch_behav_refresh
                else "data/models/v228_shaped_v223/model.pkl"
                if v228_shaped_v223
                else "data/models/v234_behav_mix_v11_shaped/model.pkl"
                if v234_behav_mix_v11
                else "data/models/v237_v11lock1_v234rest/model.pkl"
                if v237_v11lock1_v234rest
                else "data/models/v241_v11lock1_v118rest/model.pkl"
                if v241_v11lock1_v118rest
                else "data/models/v245_v11lock1_v244rest/model.pkl"
                if v245_v11lock1_v244rest
                else "data/models/v248_batchrank_schema/model.pkl"
                if v248_batchrank_schema
                else "data/models/v249_batchrank_behavmix_v11/model.pkl"
                if v249_batchrank_behavmix_v11
                else "data/models/v252_clean_top1recipe_schema/model.pkl"
                if v252_clean_top1recipe_schema
                else "data/models/v253_oldwindow_schema/model.pkl"
                if v253_oldwindow_schema
                else "data/models/v255_oldwindow_top1schema/model.pkl"
                if v255_oldwindow_top1schema
                else "data/models/v257_trainonly_top1schema/model.pkl"
                if v257_trainonly_top1schema
                else "data/models/v260_fit80_sanitized_top1schema/model.pkl"
                if v260_fit80_sanitized_top1schema
                else "data/models/v262_v260w95_v11w5_rankblend/model.pkl"
                if v262_v260w95_v11w5_rankblend
                else "data/models/v264_v260w80_v263w20_rankblend/model.pkl"
                if v264_v260w80_v263w20_rankblend
                else "data/models/v270_v260w98_v263w01_v265w01_rankblend/model.pkl"
                if v270_v260w98_v263w01_v265w01_rankblend
                else "data/models/v291_v263_rankmean_latest/model.pkl"
                if v291_v263_rankmean_latest
                else "data/models/v288_top1style_schema/model.pkl"
                if v288_top1style_schema
                else "data/models/v292_v26090_v26105_v11_rankblend/model.pkl"
                if v292_v26090_v26105_v11_rankblend
                else "data/models/v289_v270w90_v288397avgw10_rankblend/model.pkl"
                if v289_v270w90_v288397avgw10_rankblend
                else "data/models/v290_v289w90_v11w10_runtime/model.pkl"
                if v290_v289w90_v11w10_runtime
                else "data/models/v270_v260w98_v263w01_v265w01_rankblend/model.pkl"
                if v287_shape_adaptive_v11_v270
                else "data/models/v296_rankmlp_wide/model.pkl"
                if v297_shape_adaptive_v296_v11_v270
                else "data/models/v298_lambdamart_wide_ranker/model.pkl"
                if v298_lambdamart_wide_ranker
                else "data/models/v306_lambdamart_temporal_seed_ensemble/model.pkl"
                if v306_lambdamart_temporal_seed_ensemble
                else "data/models/v315_original_lambdamart_top10/model.pkl"
                if v315_original_lambdamart_top10
                else "data/models/v271_v11lock1_v268rest/model.pkl"
                if v271_v11lock1_v268rest
                else "data/models/v274_v11lock1_v273rest/model.pkl"
                if v274_v11lock1_v273rest
                else "data/models/v276_livesized6080100_v273/model.pkl"
                if v276_livesized6080100_v273
                else "data/models/v277_livesized6080100_temporal_probe/model.pkl"
                if v277_livesized6080100_temporal
                else "data/models/v280_livesized6080100_temporal_consistency/model.pkl"
                if v280_livesized6080100_temporal_consistency
                else "data/models/v294_hg2_rebuild/model.pkl"
                if v294_hg2_rebuild
                else "data/models/v200_stackseq_last3/model.pkl"
                if v200_stackseq_last3
                else "data/models/v201_stackseq_wide8/model.pkl"
                if v201_stackseq_wide8
                else "data/models/v183_v11lock1_v181rest/model.pkl"
                if v183_v11lock1_v181rest
                else "data/models/v181_actioncap8_livehand89_maxks075_noidentity_fullheads/model.pkl"
                if v181_actioncap8_livehand89_maxks075_fullheads
                else "data/models/v179_actioncap8_livehand89_ks060_noidentity/model.pkl"
                if v179_actioncap8_livehand89_ks060
                else "data/models/v175_actioncap8_multilive_ks075_noidentity_livesized/model.pkl"
                if v175_actioncap8_ks075
                else "data/models/v173_actioncap8_noidentity_livesized/model.pkl"
                if v173_actioncap8
                else "data/models/v140_multiseed_livesized/model.pkl"
                if v140_multi
                else "data/models/v136_livesized_20260707/model.pkl"
                if v136_live
                else "data/models/v133_v113_split_ngram/model.pkl"
                if v133_split
                else "data/models/v132_behav_ngram/model.pkl"
                if v132_ngram
                else "data/models/v131_behav_mix/model.pkl"
                if v131_behav
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
                                "data/models/v113_daily_seg35_mean_avg/model.pkl"
                                if daily and strategy == "seg35_mean_avg"
                                else "data/models/v113_daily/model.pkl"
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
        self._last_score_extra: dict[str, Any] = {}
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
        elif family == "v294_hg2_rebuild":
            files.extend(
                [
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "infer.py",
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "hg_model.py",
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "hg_features.py",
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "features_v2.py",
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "hg2_features_base.py",
                    REPO_ROOT / self.variant_cfg["model_file"],
                    REPO_ROOT / "data" / "models" / "v294_hg2_rebuild" / "meta.json",
                ]
            )
        elif family == "v297_shape_adaptive_v296_v11_v270":
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
                    REPO_ROOT / "poker44" / "score" / "v112_super_inference.py",
                    REPO_ROOT / "poker44" / "score" / "ngram_ranker.py",
                    REPO_ROOT / "poker44" / "score" / "robust_schema" / "__init__.py",
                    REPO_ROOT / "poker44" / "score" / "robust_schema" / "features.py",
                    REPO_ROOT / "poker44" / "score" / "sequence_schema.py",
                    REPO_ROOT / "poker44" / "score" / "temporal_consistency_features.py",
                    REPO_ROOT / "poker44" / "score" / "action_anomaly_features.py",
                    REPO_ROOT / "poker44" / "score" / "statistical_v25.py",
                    REPO_ROOT / "poker44" / "score" / "features_v13_safe.py",
                    REPO_ROOT / "poker44" / "score" / "extended_features.py",
                    REPO_ROOT / "poker44" / "score" / "enterprise_features.py",
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "hg_features.py",
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "features_v2.py",
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "hg2_features_base.py",
                    REPO_ROOT / self.variant_cfg["model_file"],
                    REPO_ROOT / "data" / "models" / "v296_rankmlp_wide" / "report.json",
                    REPO_ROOT / "data" / "models" / "v270_v260w98_v263w01_v265w01_rankblend" / "model.pkl",
                    REPO_ROOT / "data" / "models" / "v270_v260w98_v263w01_v265w01_rankblend" / "report.json",
                ]
            )
        elif family in {
            "v298_lambdamart_wide_ranker",
            "v306_lambdamart_temporal_seed_ensemble",
        }:
            files.extend(
                [
                    REPO_ROOT / "poker44" / "score" / "lambdamart_wide_inference.py",
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "hg_features.py",
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "features_v2.py",
                    REPO_ROOT / "poker44" / "score" / "hg2_runtime" / "hg2_features_base.py",
                    REPO_ROOT / self.variant_cfg["model_file"],
                    REPO_ROOT
                    / "data"
                    / "models"
                    / family
                    / "report.json",
                ]
            )
        elif family == "v315_original_lambdamart_top10":
            files.extend(
                [
                    REPO_ROOT / "poker44" / "score" / "original_behavior_features.py",
                    REPO_ROOT / "poker44" / "score" / "original_lambdamart_inference.py",
                    REPO_ROOT / self.variant_cfg["model_file"],
                    REPO_ROOT / "data" / "models" / family / "report.json",
                ]
            )
        elif family in {
            "v112_super",
            "v113_daily",
            "v115_short",
            "v118_live",
            "v118_stable75",
            "v118_seg35",
            "v125_topk",
            "v131_behav",
            "v132_ngram",
            "v133_split",
            "v136_live",
            "v140_multi",
            "v142_rankblend",
            "v173_actioncap8",
            "v175_actioncap8_ks075",
            "v179_actioncap8_livehand89_ks060",
            "v181_actioncap8_livehand89_maxks075_fullheads",
            "v183_v11lock1_v181rest",
            "v184_v11lock2_v181rest",
            "v193_v11lock1_v145rank_rest",
            "v190_contract60_80_100_ks050_livesized",
            "v209_served_rankcap_ks055",
            "v216_served_rankcap_ks060",
            "v217_v11lock2_v216rest",
            "v218_withinbatch_behav",
            "v219_rebuilt_superv2",
            "v220_v11lock2_v219rest",
            "v221_v11lock2_nightlyrest",
            "v223_withinbatch_behav_refresh",
            "v228_shaped_v223",
            "v234_behav_mix_v11",
            "v237_v11lock1_v234rest",
            "v241_v11lock1_v118rest",
            "v245_v11lock1_v244rest",
            "v248_batchrank_schema",
            "v249_batchrank_behavmix_v11",
            "v252_clean_top1recipe_schema",
            "v253_oldwindow_schema",
            "v255_oldwindow_top1schema",
            "v257_trainonly_top1schema",
            "v260_fit80_sanitized_top1schema",
            "v262_v260w95_v11w5_rankblend",
            "v264_v260w80_v263w20_rankblend",
            "v270_v260w98_v263w01_v265w01_rankblend",
            "v288_top1style_schema",
            "v292_v26090_v26105_v11_rankblend",
            "v289_v270w90_v288397avgw10_rankblend",
            "v290_v289w90_v11w10_runtime",
            "v287_shape_adaptive_v11_v270",
            "v297_shape_adaptive_v296_v11_v270",
            "v271_v11lock1_v268rest",
            "v274_v11lock1_v273rest",
            "v276_livesized6080100_v273",
            "v277_livesized6080100_temporal",
            "v280_livesized6080100_temporal_consistency",
            "v294_hg2_rebuild",
            "v200_stackseq_last3",
            "v201_stackseq_wide8",
        }:
            files.extend(
                [
                    REPO_ROOT / "poker44" / "score" / "v112_super_inference.py",
                    REPO_ROOT / "poker44" / "score" / "ngram_ranker.py",
                    REPO_ROOT / "poker44" / "score" / "robust_schema" / "__init__.py",
                    REPO_ROOT / "poker44" / "score" / "robust_schema" / "features.py",
                    REPO_ROOT / "poker44" / "score" / "sequence_schema.py",
                    REPO_ROOT / "poker44" / "score" / "temporal_consistency_features.py",
                    REPO_ROOT / "poker44" / "score" / "action_anomaly_features.py",
                    REPO_ROOT / "poker44" / "score" / "statistical_v25.py",
                    REPO_ROOT / "poker44" / "score" / "features_pot_geometry.py",
                    REPO_ROOT / "poker44" / "score" / "features_v13_safe.py",
                    REPO_ROOT / "poker44" / "score" / "extended_features.py",
                    REPO_ROOT / "poker44" / "score" / "enterprise_features.py",
                    REPO_ROOT / self.variant_cfg["model_file"],
                ]
            )
            if family == "v133_split":
                files.append(
                    REPO_ROOT / "data" / "models" / "v133_v113_split_ngram" / "report.json"
                )
            if family == "v136_live":
                files.append(
                    REPO_ROOT / "data" / "models" / "v136_livesized_20260707" / "report.json"
                )
            if family == "v140_multi":
                files.append(
                    REPO_ROOT / "data" / "models" / "v140_multiseed_livesized" / "report.json"
                )
            if family == "v142_rankblend":
                files.append(
                    REPO_ROOT / "data" / "models" / "v142_rankblend" / "report.json"
                )
            if family == "v173_actioncap8":
                files.append(
                    REPO_ROOT / "data" / "models" / "v173_actioncap8_noidentity_livesized" / "report.json"
                )
            if family == "v175_actioncap8_ks075":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v175_actioncap8_multilive_ks075_noidentity_livesized"
                    / "report.json"
                )
            if family == "v179_actioncap8_livehand89_ks060":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v179_actioncap8_livehand89_ks060_noidentity"
                    / "report.json"
                )
            if family == "v181_actioncap8_livehand89_maxks075_fullheads":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v181_actioncap8_livehand89_maxks075_noidentity_fullheads"
                    / "report.json"
                )
            if family == "v183_v11lock1_v181rest":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v183_v11lock1_v181rest"
                    / "report.json"
                )
            if family == "v184_v11lock2_v181rest":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v184_v11lock2_v181rest"
                    / "report.json"
                )
            if family == "v193_v11lock1_v145rank_rest":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v193_v11lock1_v145rank_rest"
                    / "report.json"
                )
            if family == "v190_contract60_80_100_ks050_livesized":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v190_contract60_80_100_ks050_livesized"
                    / "report.json"
                )
            if family == "v209_served_rankcap_ks055":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v209_served_rankcap_ks055"
                    / "report.json"
                )
            if family == "v216_served_rankcap_ks060":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v216_served_rankcap_ks060"
                    / "report.json"
                )
            if family == "v217_v11lock2_v216rest":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v217_v11lock2_v216rest"
                    / "report.json"
                )
            if family == "v218_withinbatch_behav":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v218_withinbatch_behav"
                    / "report.json"
                )
            if family == "v219_rebuilt_superv2":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "rebuilt_superv2_inference.py",
                        REPO_ROOT / "poker44_ml" / "__init__.py",
                        REPO_ROOT / "poker44_ml" / "features.py",
                        REPO_ROOT / "poker44_ml" / "inference.py",
                        REPO_ROOT / "poker44_ml" / "stacked.py",
                        REPO_ROOT / "poker44_ml" / "calibration.py",
                        REPO_ROOT / "poker44_ml" / "chunk_score_metrics.py",
                        REPO_ROOT / "data" / "models" / "v219_rebuilt_superv2" / "report.json",
                    ]
                )
            if family == "v220_v11lock2_v219rest":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT / "poker44" / "score" / "rebuilt_superv2_inference.py",
                        REPO_ROOT / "poker44_ml" / "__init__.py",
                        REPO_ROOT / "poker44_ml" / "features.py",
                        REPO_ROOT / "poker44_ml" / "inference.py",
                        REPO_ROOT / "poker44_ml" / "stacked.py",
                        REPO_ROOT / "poker44_ml" / "calibration.py",
                        REPO_ROOT / "poker44_ml" / "chunk_score_metrics.py",
                        REPO_ROOT / "data" / "models" / "v219_rebuilt_superv2" / "report.json",
                    ]
                )
            if family == "v221_v11lock2_nightlyrest":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT / "data" / "models" / "v221_v11lock2_nightlyrest" / "report.json",
                    ]
                )
            if family == "v223_withinbatch_behav_refresh":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v223_withinbatch_behav_refresh"
                    / "report.json"
                )
            if family == "v228_shaped_v223":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v228_shaped_v223"
                    / "report.json"
                )
            if family == "v234_behav_mix_v11":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT
                        / "data"
                        / "models"
                        / "v234_behav_mix_v11_shaped"
                        / "report.json",
                    ]
                )
            if family == "v237_v11lock1_v234rest":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT
                        / "data"
                        / "models"
                        / "v237_v11lock1_v234rest"
                        / "report.json",
                    ]
                )
            if family == "v241_v11lock1_v118rest":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT
                        / "data"
                        / "models"
                        / "v241_v11lock1_v118rest"
                        / "report.json",
                    ]
                )
            if family == "v245_v11lock1_v244rest":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT
                        / "data"
                        / "models"
                        / "v245_v11lock1_v244rest"
                        / "report.json",
                    ]
                )
            if family == "v248_batchrank_schema":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v248_batchrank_schema"
                    / "report.json"
                )
            if family == "v249_batchrank_behavmix_v11":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v249_batchrank_behavmix_v11"
                    / "report.json"
                )
            if family == "v252_clean_top1recipe_schema":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v252_clean_top1recipe_schema"
                    / "report.json"
                )
            if family == "v253_oldwindow_schema":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v253_oldwindow_schema"
                    / "report.json"
                )
            if family == "v255_oldwindow_top1schema":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v255_oldwindow_top1schema"
                    / "report.json"
                )
            if family == "v257_trainonly_top1schema":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v257_trainonly_top1schema"
                    / "report.json"
                )
            if family == "v260_fit80_sanitized_top1schema":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v260_fit80_sanitized_top1schema"
                    / "report.json"
                )
            if family == "v262_v260w95_v11w5_rankblend":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v262_v260w95_v11w5_rankblend"
                    / "report.json"
                )
            if family == "v264_v260w80_v263w20_rankblend":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v264_v260w80_v263w20_rankblend"
                    / "report.json"
                )
            if family == "v270_v260w98_v263w01_v265w01_rankblend":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v270_v260w98_v263w01_v265w01_rankblend"
                        / "report.json"
                )
            if family == "v291_v263_rankmean_latest":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v291_v263_rankmean_latest"
                    / "report.json"
                )
            if family == "v288_top1style_schema":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v288_top1style_schema"
                    / "report.json"
                )
            if family == "v292_v26090_v26105_v11_rankblend":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT
                        / "data"
                        / "models"
                        / "v292_v26090_v26105_v11_rankblend"
                        / "report.json",
                    ]
                )
            if family == "v289_v270w90_v288397avgw10_rankblend":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v289_v270w90_v288397avgw10_rankblend"
                    / "report.json"
                )
            if family == "v290_v289w90_v11w10_runtime":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT
                        / "data"
                        / "models"
                        / "v290_v289w90_v11w10_runtime"
                        / "report.json",
                    ]
                )
            if family == "v287_shape_adaptive_v11_v270":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT
                        / "data"
                        / "models"
                        / "v270_v260w98_v263w01_v265w01_rankblend"
                        / "report.json",
                    ]
                )
            if family == "v271_v11lock1_v268rest":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT
                        / "data"
                        / "models"
                        / "v271_v11lock1_v268rest"
                        / "report.json",
                    ]
                )
            if family == "v274_v11lock1_v273rest":
                files.extend(
                    [
                        REPO_ROOT / "poker44" / "score" / "ensemble_v11.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v5.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v6.py",
                        REPO_ROOT / "poker44" / "score" / "statistical_v9.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8.py",
                        REPO_ROOT / "poker44" / "score" / "sequence_v8_markov.py",
                        REPO_ROOT / "poker44" / "score" / "features_response_curves.py",
                        REPO_ROOT
                        / "data"
                        / "models"
                        / "v274_v11lock1_v273rest"
                        / "report.json",
                    ]
                )
            if family == "v276_livesized6080100_v273":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v276_livesized6080100_v273"
                    / "report.json"
                )
            if family == "v277_livesized6080100_temporal":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v277_livesized6080100_temporal_probe"
                    / "report.json"
                )
            if family == "v280_livesized6080100_temporal_consistency":
                files.append(
                    REPO_ROOT
                    / "data"
                    / "models"
                    / "v280_livesized6080100_temporal_consistency"
                    / "report.json"
                )
            if family in {"v200_stackseq_last3", "v201_stackseq_wide8"}:
                files.extend(
                    [
                        REPO_ROOT / "poker44_ml" / "__init__.py",
                        REPO_ROOT / "poker44_ml" / "features.py",
                        REPO_ROOT / "poker44_ml" / "inference.py",
                        REPO_ROOT / "poker44_ml" / "stacked.py",
                        REPO_ROOT / "poker44_ml" / "sequence_model.py",
                        REPO_ROOT / "poker44_ml" / "calibration.py",
                        REPO_ROOT
                        / "data"
                        / "models"
                        / ("v201_stackseq_wide8" if family == "v201_stackseq_wide8" else "v200_stackseq_last3")
                        / "report.json",
                    ]
                )
        return [path for path in files if path.exists()]

    def _build_manifest(self) -> dict[str, Any]:
        family = self.variant_cfg["family"]
        if family in {
            "v112_super",
            "v113_daily",
            "v115_short",
            "v118_live",
            "v118_stable75",
            "v118_seg35",
            "v125_topk",
            "v131_behav",
            "v132_ngram",
            "v133_split",
            "v136_live",
            "v140_multi",
            "v142_rankblend",
            "v173_actioncap8",
            "v175_actioncap8_ks075",
            "v179_actioncap8_livehand89_ks060",
            "v181_actioncap8_livehand89_maxks075_fullheads",
            "v183_v11lock1_v181rest",
            "v184_v11lock2_v181rest",
            "v193_v11lock1_v145rank_rest",
            "v190_contract60_80_100_ks050_livesized",
            "v209_served_rankcap_ks055",
            "v216_served_rankcap_ks060",
            "v217_v11lock2_v216rest",
            "v218_withinbatch_behav",
            "v219_rebuilt_superv2",
            "v220_v11lock2_v219rest",
            "v223_withinbatch_behav_refresh",
            "v260_fit80_sanitized_top1schema",
            "v262_v260w95_v11w5_rankblend",
            "v264_v260w80_v263w20_rankblend",
            "v270_v260w98_v263w01_v265w01_rankblend",
            "v288_top1style_schema",
            "v292_v26090_v26105_v11_rankblend",
            "v289_v270w90_v288397avgw10_rankblend",
            "v290_v289w90_v11w10_runtime",
            "v287_shape_adaptive_v11_v270",
            "v298_lambdamart_wide_ranker",
            "v306_lambdamart_temporal_seed_ensemble",
            "v315_original_lambdamart_top10",
            "v271_v11lock1_v268rest",
            "v274_v11lock1_v273rest",
            "v276_livesized6080100_v273",
            "v277_livesized6080100_temporal",
            "v280_livesized6080100_temporal_consistency",
            "v200_stackseq_last3",
            "v201_stackseq_wide8",
        }:
            if family == "v142_rankblend":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-07 using miner-visible hand/action payload "
                    "fields only. It is a self-contained rank-space blend of two independently "
                    "trained live-sized public-benchmark rankers; no validator-private labels, "
                    "wallets, hotkeys, IP addresses, or deployment logs were used for training."
                )
            elif family == "v173_actioncap8":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. Training merged public chunks into live-sized units and "
                    "applied the same max-8 actions-per-hand cap used by the served bundle. "
                    "No validator-private labels, wallets, hotkeys, IP addresses, or "
                    "deployment logs were used for training."
                )
            elif family == "v175_actioncap8_ks075":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. Training merged public chunks into live-sized units, "
                    "applied max-8 actions-per-hand train/serve parity, and used unlabeled "
                    "miner-received forward-audit payloads only to remove high train-live "
                    "drift features by KS stability filtering. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, or deployment logs were used for training."
                )
            elif family == "v179_actioncap8_livehand89_ks060":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. Training merged public chunks into 80-100 hand live-shaped "
                    "units with target 89 hands, applied max-8 actions-per-hand train/serve "
                    "parity, and used unlabeled miner-received forward-audit payloads only "
                    "to remove train-live drift features with KS<=0.60 stability filtering. "
                    "No validator-private labels, wallets, hotkeys, IP addresses, or "
                    "deployment logs were used for training."
                )
            elif family == "v181_actioncap8_livehand89_maxks075_fullheads":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. Training merged public chunks into 80-100 hand live-shaped "
                    "units with target 89 hands, applied max-8 actions-per-hand train/serve "
                    "parity, used six independently trained rank heads, and used unlabeled "
                    "miner-received forward-audit payloads only to remove train-live drift "
                    "features by worst-live-payload KS<=0.75 stability filtering. No "
                    "validator-private labels, wallets, hotkeys, IP addresses, or deployment "
                    "logs were used for training."
                )
            elif family in {"v183_v11lock1_v181rest", "v184_v11lock2_v181rest"}:
                lock_text = "top-1" if family == "v183_v11lock1_v181rest" else "top-2"
                training_statement = (
                    "Model combines the deterministic public v11 behavioural scorer with "
                    "a v181 segment ranker trained only on public Poker44 benchmark "
                    "releaseVersion v1.13 through sourceDate 2026-07-08. The served "
                    f"ranker locks v11 {lock_text} picks and orders the remaining chunks "
                    "with the v181 public-benchmark ranker. Forward-audit payloads were "
                    "used only for unlabeled train/serve shape and drift checks; no "
                    "validator-private labels, wallets, hotkeys, IP addresses, or "
                    "deployment logs were used for training."
                )
            elif family == "v193_v11lock1_v145rank_rest":
                training_statement = (
                    "Model combines the deterministic public v11 behavioural scorer with "
                    "a v145 rank-mean rest ranker trained on public Poker44 benchmark "
                    "releaseVersion v1.13 through sourceDate 2026-07-08 plus the released "
                    "subnet human-hand corpus as public negative examples. The served "
                    "ranker locks the v11 top-1 pick and orders the remaining chunks with "
                    "the v145 public-benchmark/human-corpus ranker. Forward-audit payloads "
                    "were used only for unlabeled train/serve shape and drift checks; no "
                    "validator-private labels, wallets, hotkeys, IP addresses, or deployment "
                    "logs were used for training."
                )
            elif family == "v190_contract60_80_100_ks050_livesized":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. Training uses 60/80/100 hand live-shaped units and "
                    "unlabeled miner-received forward-audit payloads only to remove "
                    "train-live drift features with worst-live-payload KS<=0.50 stability "
                    "filtering. No validator-private labels, wallets, hotkeys, IP "
                    "addresses, or deployment logs were used for training."
                )
            elif family == "v209_served_rankcap_ks055":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. Training uses 60/80/100 hand live-shaped units, selects "
                    "the candidate by served rank-cap official reward, and uses unlabeled "
                    "miner-received forward-audit payloads only to remove train-live drift "
                    "features with worst-live-payload KS<=0.55 stability filtering. No "
                    "validator-private labels, wallets, hotkeys, IP addresses, or deployment "
                    "logs were used for training."
                )
            elif family == "v216_served_rankcap_ks060":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. Training uses 60/80/100 hand live-shaped units, selects "
                    "the candidate by served rank-cap official reward, and uses unlabeled "
                    "miner-received forward-audit payloads only to remove train-live drift "
                    "features with worst-live-payload KS<=0.60 stability filtering. No "
                    "validator-private labels, wallets, hotkeys, IP addresses, or deployment "
                    "logs were used for training."
                )
            elif family == "v217_v11lock2_v216rest":
                training_statement = (
                    "Model combines the deterministic public v11 behavioural scorer with "
                    "the v216 served-rankcap KS<=0.60 public-benchmark ranker. The served "
                    "ranker locks the v11 top-2 picks and orders the remaining chunks with "
                    "the v216 ranker trained on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. Forward-audit payloads were used only for unlabeled "
                    "train/serve shape and drift checks; no validator-private labels, "
                    "wallets, hotkeys, IP addresses, or deployment logs were used for training."
                )
            elif family == "v218_withinbatch_behav":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. Training builds live-sized 70-110 hand same-date/same-label "
                    "units across multiple merge seeds, uses within-batch behavioural "
                    "features, selects by served rank-cap official reward on batch100 LODO, "
                    "and uses unlabeled miner-received forward-audit payloads only for "
                    "train-live shape and drift checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, or deployment logs were used for training."
                )
            elif family == "v219_rebuilt_superv2":
                training_statement = (
                    "Model is a local rebuild of the public super-v2 stacked architecture, "
                    "trained only on Poker44 public benchmark releaseVersion v1.13 through "
                    "sourceDate 2026-07-08 using miner-visible hand/action payload fields. "
                    "No validator-private labels, wallets, hotkeys, IP addresses, deployment "
                    "logs, or private player data were used for training."
                )
            elif family == "v220_v11lock2_v219rest":
                training_statement = (
                    "Model combines the deterministic public v11 behavioural scorer with "
                    "a local rebuild of the public super-v2 stacked architecture. The served "
                    "ranker locks the v11 top-2 picks and orders the remaining chunks with "
                    "the rebuilt super-v2 public-benchmark ranker trained only on Poker44 "
                    "public benchmark releaseVersion v1.13 through sourceDate 2026-07-08 "
                    "using miner-visible hand/action payload fields. No validator-private "
                    "labels, wallets, hotkeys, IP addresses, deployment logs, or private "
                    "player data were used for training."
                )
            elif family == "v221_v11lock2_nightlyrest":
                training_statement = (
                    "Model combines the deterministic public v11 behavioural scorer with "
                    "a supervised super-sequence stack. The served ranker locks the v11 "
                    "top-2 picks and orders the remaining chunks with the super-sequence "
                    "stack trained only on Poker44 public benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-09 using miner-visible hand/action payload "
                    "fields. Unlabeled miner-received forward-audit payloads were used only "
                    "for train/serve shape and topology checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player data "
                    "were used for training."
                )
            elif family == "v223_withinbatch_behav_refresh":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-09 using miner-visible hand/action payload "
                    "fields only. Training builds live-sized 70-110 hand same-date/same-label "
                    "units across multiple merge seeds, uses within-batch behavioural "
                    "features, selects by served rank-cap official reward on batch100 LODO, "
                    "and uses unlabeled miner-received forward-audit payloads only for "
                    "train-live shape and drift checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player data "
                    "were used for training."
                )
            elif family == "v228_shaped_v223":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-09 using miner-visible hand/action payload "
                    "fields only. It reuses the within-batch behavioural ranker family and "
                    "adds a monotone score-shaping head selected on public benchmark replay "
                    "to preserve ordering while exposing more chunks above the validator's "
                    "fixed 0.5 threshold. Unlabeled miner-received forward-audit payloads "
                    "were used only for score-shape and train-live topology checks. No "
                    "validator-private labels, wallets, hotkeys, IP addresses, deployment "
                    "logs, or private player data were used for training."
                )
            elif family == "v234_behav_mix_v11":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-09 using miner-visible hand/action payload "
                    "fields only. It augments the within-batch behavioural ranker with "
                    "deterministic v11 behavioural telemetry features, then serves a "
                    "rank-ladder over the rank_mean head through rank-cap top5. Unlabeled miner-received forward-audit "
                    "payloads were used only for train-live topology and shape checks. No "
                    "validator-private labels, wallets, hotkeys, IP addresses, deployment "
                    "logs, or private player data were used for training."
                )
            elif family == "v237_v11lock1_v234rest":
                training_statement = (
                    "Model combines deterministic public v11 top-1 locking with a v234 "
                    "rest-ranking child trained only on public Poker44 benchmark releaseVersion "
                    "v1.13 through sourceDate 2026-07-09 using miner-visible hand/action payload "
                    "fields only. Unlabeled miner-received forward-audit payloads were used only "
                    "for train-live topology and shape checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player data were "
                    "used for training."
                )
            elif family == "v241_v11lock1_v118rest":
                training_statement = (
                    "Model combines deterministic public v11 top-1 locking with a v118 "
                    "rest-ranking child trained only on public Poker44 benchmark releaseVersion "
                    "v1.13 through sourceDate 2026-07-09 using miner-visible hand/action payload "
                    "fields only. Unlabeled miner-received forward-audit payloads were used only "
                    "for train-live topology and shape checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player data were "
                    "used for training."
                )
            elif family == "v245_v11lock1_v244rest":
                training_statement = (
                    "Model combines deterministic public v11 top-1 locking with a v244 "
                    "clean-restart schema rest-ranking child trained only on public Poker44 "
                    "benchmark releaseVersion v1.13 through sourceDate 2026-07-09 using "
                    "miner-visible hand/action payload fields only. Unlabeled miner-received "
                    "forward-audit payloads were used only for train-live topology and shape "
                    "checks. No validator-private labels, wallets, hotkeys, IP addresses, "
                    "deployment logs, or private player data were used for training."
                )
            elif family == "v248_batchrank_schema":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-09 using miner-visible hand/action payload "
                    "fields only. It transforms robust schema features into within-batch "
                    "feature ranks over 100-chunk public benchmark pseudo-batches during "
                    "training and applies the same batch-rank-only transform at serve time. "
                    "Unlabeled miner-received forward-audit payloads were used only for "
                    "shape and topology checks. No validator-private labels, wallets, "
                    "hotkeys, IP addresses, deployment logs, or private player data were "
                    "used for training."
                )
            elif family == "v249_batchrank_behavmix_v11":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-09 using miner-visible hand/action payload "
                    "fields only. It uses behavioural/v11 telemetry schema features transformed "
                    "into within-batch feature ranks over 100-chunk public benchmark "
                    "pseudo-batches during training and applies the same batch-rank-only "
                    "transform at serve time. Unlabeled miner-received forward-audit payloads "
                    "were used only for shape and topology checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player data "
                    "were used for training."
                )
            elif family == "v252_clean_top1recipe_schema":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-09 using miner-visible hand/action payload "
                    "fields only. It uses robust schema features filtered to stable absolute "
                    "features and serves a rank-space strategy over 100-chunk validator "
                    "payloads. Unlabeled miner-received forward-audit payloads were used only "
                    "for topology, shape, and train/serve checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player data "
                    "were used for training."
                )
            elif family == "v253_oldwindow_schema":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "sourceDates 2026-06-03 through 2026-07-02 using miner-visible "
                    "hand/action payload fields only. It uses a deliberately old-window "
                    "robust schema tree ensemble to reduce benchmark-overfit to the latest "
                    "sanitized release and serves a rank-space strategy over validator "
                    "payloads. Unlabeled miner-received forward-audit payloads were used only "
                    "for topology, shape, and train/serve checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player data "
                    "were used for training."
                )
            elif family == "v255_oldwindow_top1schema":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "sourceDates 2026-06-03 through 2026-07-02 using miner-visible "
                    "hand/action payload fields only. It uses the old-window robust schema "
                    "with repeat-compression features removed to match the 293-feature "
                    "top1 public model shape, top1-style tree hyperparameters, and a "
                    "rank-space strategy over validator payloads. Unlabeled miner-received "
                    "forward-audit payloads were used only for topology, shape, and "
                    "train/serve checks. No validator-private labels, wallets, hotkeys, "
                    "IP addresses, deployment logs, or private player data were used for training."
                )
            elif family == "v257_trainonly_top1schema":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "sourceDates 2026-06-03 through 2026-06-30 using miner-visible "
                    "hand/action payload fields only, with 2026-07-01 through 2026-07-02 "
                    "held out for reporting. It uses the 293-feature robust schema with "
                    "repeat-compression features removed, top1-style tree hyperparameters, "
                    "and a monotone rank-ladder serve strategy over validator payloads. "
                    "Unlabeled miner-received forward-audit payloads were used only for "
                    "topology, shape, and train/serve checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player "
                    "data were used for training."
                )
            elif family == "v260_fit80_sanitized_top1schema":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "sourceDates 2026-06-03 through 2026-06-30 using miner-visible "
                    "hand/action payload fields only, with 2026-07-01 through 2026-07-02 "
                    "held out for reporting. Validator miner-visible sanitization was "
                    "applied before feature extraction, repeat-compression features were "
                    "removed from the robust schema, and a stratified fit split was used "
                    "to train the tree heads. Scores are served through a monotone "
                    "rank-ladder strategy over validator payloads. Unlabeled miner-received "
                    "forward-audit payloads were used only for topology, shape, and "
                    "train/serve checks. No validator-private labels, wallets, hotkeys, "
                    "IP addresses, deployment logs, or private player data were used for training."
                )
            elif family == "v262_v260w95_v11w5_rankblend":
                training_statement = (
                    "Rank-space blend of two disclosed AceGuard components: a fit-split "
                    "293-feature schema tree ranker trained only on public Poker44 benchmark "
                    "sourceDates 2026-06-03 through 2026-06-30 with miner-visible sanitization "
                    "and 2026-07-01 through 2026-07-02 held out for reporting, plus the "
                    "independent deterministic v11 behavioural scorer. Unlabeled miner-received "
                    "forward-audit payloads were used only for topology, shape, and train/serve "
                    "checks. No validator-private labels, wallets, hotkeys, IP addresses, "
                    "deployment logs, or private player data were used for training."
                )
            elif family == "v264_v260w80_v263w20_rankblend":
                training_statement = (
                    "Rank-space blend of two disclosed AceGuard schema components: v260 ET, "
                    "a fit-split 293-feature schema ranker trained only on public Poker44 "
                    "benchmark sourceDates through 2026-07-02 with miner-visible "
                    "sanitization, and v263 rank_mean, the same schema family refreshed "
                    "against public benchmark sourceDates through 2026-07-10 with "
                    "2026-07-09 and 2026-07-10 held out for reporting. Unlabeled "
                    "miner-received forward-audit payloads were used only for topology, "
                    "shape, and train/serve checks. No validator-private labels, wallets, "
                    "hotkeys, IP addresses, deployment logs, or private player data were "
                    "used for training."
                )
            elif family == "v270_v260w98_v263w01_v265w01_rankblend":
                training_statement = (
                    "Rank-space blend of disclosed AceGuard schema components selected by "
                    "live-topology replay against current public leaders: 0.98 v260 ET, "
                    "0.01 v263 rank_mean, and 0.01 v265 schema-anomaly rank_mean. Child "
                    "models were trained only on public Poker44 benchmark data with "
                    "miner-visible sanitization. Unlabeled miner-received forward-audit "
                    "payloads were used only for topology, shape, and train/serve checks. "
                    "No validator-private labels, wallets, hotkeys, IP addresses, "
                    "deployment logs, or private player data were used for training."
                )
            elif family == "v291_v263_rankmean_latest":
                training_statement = (
                    "Pure v263 rank_mean AceGuard schema scorer trained only on public Poker44 "
                    "benchmark data through 2026-07-10 with miner-visible sanitization. "
                    "The candidate is selected from the latest labeled benchmark check where "
                    "it outperformed the public UID99 reference under the official reward. "
                    "Unlabeled miner-received forward-audit payloads were used only for "
                    "topology, shape, and train/serve checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player data "
                    "were used for training."
                )
            elif family == "v288_top1style_schema":
                training_statement = (
                    "Fresh 397-feature robust-schema AceGuard ranker trained only on public "
                    "Poker44 benchmark releaseVersion v1.13 sourceDates 2026-05-26 through "
                    "2026-07-10 with miner-visible sanitization. The candidate uses the same "
                    "serve-time schema scorer as other disclosed AceGuard schema models and is "
                    "served through a rank-ladder strategy over validator payloads. Unlabeled "
                    "miner-received forward-audit payloads were used only for topology, shape, "
                    "and train/serve checks. No validator-private labels, wallets, hotkeys, IP "
                    "addresses, deployment logs, or private player data were used for training."
                )
            elif family == "v292_v26090_v26105_v11_rankblend":
                training_statement = (
                    "Rank-space blend of disclosed AceGuard components selected by live-payload "
                    "topology replay against current public leaders: 0.90 v260 ET, 0.05 "
                    "independent-seed v261 ET, and 0.05 deterministic v11 behavioural scorer. "
                    "Tree child models were trained only on public Poker44 benchmark data with "
                    "miner-visible sanitization; v11 is a deterministic scorer over miner-visible "
                    "hand-history fields. Unlabeled miner-received forward-audit payloads were "
                    "used only for topology, shape, and train/serve checks. No validator-private "
                    "labels, wallets, hotkeys, IP addresses, deployment logs, or private player "
                    "data were used for training."
                )
            elif family == "v289_v270w90_v288397avgw10_rankblend":
                training_statement = (
                    "Rank-space blend of disclosed AceGuard components selected by "
                    "live-payload topology replay: 0.90 v270 UID99-like schema rank blend "
                    "and 0.10 v288 397-feature top1-style robust-schema branch. Child "
                    "models were trained only on public Poker44 benchmark data with "
                    "miner-visible sanitization. Unlabeled miner-received forward-audit "
                    "payloads were used only for topology, shape, and train/serve checks. "
                    "No validator-private labels, wallets, hotkeys, IP addresses, "
                    "deployment logs, or private player data were used for training."
                )
            elif family == "v290_v289w90_v11w10_runtime":
                training_statement = (
                    "Runtime rank-space blend of disclosed AceGuard components selected by "
                    "held-out stress replay: 0.90 v289 UID99-like schema branch and 0.10 "
                    "deterministic v11 behavioural branch. The v289 child model was trained "
                    "only on public Poker44 benchmark data with miner-visible sanitization; "
                    "v11 is a deterministic scorer over miner-visible hand-history fields. "
                    "Unlabeled miner-received forward-audit payloads were used only for "
                    "topology, shape, and train/serve checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player "
                    "data were used for training or routing."
                )
            elif family == "v287_shape_adaptive_v11_v270":
                training_statement = (
                    "Shape-adaptive scorer combining two disclosed AceGuard components: "
                    "the deterministic public v11 behavioural top-2 scorer and the v270 "
                    "rank-space schema blend. The v270 child models were trained only on "
                    "public Poker44 benchmark data with miner-visible sanitization. The "
                    "serve-time router uses only unlabeled current batch shape statistics "
                    "(preflop and fold action shares) observed in the miner-visible payload "
                    "to choose v11/top2 for lower-preflop/lower-fold batches and v270/top20 "
                    "for high-preflop/high-fold batches. Unlabeled miner-received "
                    "forward-audit payloads were used only for topology, shape, and "
                    "train/serve checks. No validator-private labels, wallets, hotkeys, "
                    "IP addresses, deployment logs, or private player data were used for "
                    "training or routing."
                )
            elif family == "v271_v11lock1_v268rest":
                training_statement = (
                    "v11 top-1 locked behavioural anchor plus v268, a fresh robust-schema "
                    "ranker trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-10 using miner-visible hand/action payload "
                    "fields. Unlabeled miner-received forward-audit payloads were used only "
                    "for topology, shape, and train/serve checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player data "
                    "were used for training."
                )
            elif family == "v274_v11lock1_v273rest":
                training_statement = (
                    "v11 top-1 locked behavioural anchor plus v273, a live-sized "
                    "60/80/100 hand contract ranker trained only on public Poker44 "
                    "benchmark releaseVersion v1.13 through sourceDate 2026-07-10 "
                    "using miner-visible hand/action payload fields. Unlabeled "
                    "miner-received forward-audit payloads were used only for topology, "
                    "shape, and train/serve checks. No validator-private labels, wallets, "
                    "hotkeys, IP addresses, deployment logs, or private player data were "
                    "used for training."
                )
            elif family == "v276_livesized6080100_v273":
                training_statement = (
                    "Live-sized 60/80/100 hand-contract super_seq ranker trained only "
                    "on public Poker44 benchmark releaseVersion v1.13 through sourceDate "
                    "2026-07-10 using miner-visible hand/action payload fields. Unlabeled "
                    "miner-received forward-audit payloads were used only for topology, "
                    "shape, and train/serve checks. No validator-private labels, wallets, "
                    "hotkeys, IP addresses, deployment logs, or private player data were "
                    "used for training."
                )
            elif family == "v277_livesized6080100_temporal":
                training_statement = (
                    "Live-sized 60/80/100 hand-contract super_seq ranker trained only "
                    "on public Poker44 benchmark releaseVersion v1.13 through sourceDate "
                    "2026-07-10 using miner-visible hand/action payload fields, including "
                    "temporal lag, trend, action-bigram, and street-share schema features. "
                    "Unlabeled miner-received forward-audit payloads were used only for "
                    "topology, shape, and train/serve checks. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player "
                    "data were used for training."
                )
            elif family == "v280_livesized6080100_temporal_consistency":
                training_statement = (
                    "Live-sized 60/80/100 hand-contract super_seq_temporal ranker "
                    "trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-10 using miner-visible hand/action "
                    "payload fields, including schema, hashed sequence, temporal "
                    "consistency, quartile drift, action-bigram, street-share, and "
                    "bet/pot clustering features. Unlabeled miner-received "
                    "forward-audit payloads were used only for topology, shape, and "
                    "train/serve checks. No validator-private labels, wallets, hotkeys, "
                    "IP addresses, deployment logs, or private player data were used "
                    "for training."
                )
            elif family == "v294_hg2_rebuild":
                training_statement = (
                    "HG2 weighted-rank blend trained only on public Poker44 benchmark "
                    "releaseVersion v1.13 through sourceDate 2026-07-10 using "
                    "miner-visible sanitized payload fields. Training used "
                    "prepare_hand_for_miner parity plus same-date/same-label public "
                    "benchmark live-size pooled augmentation, a stacked tree ensemble, "
                    "a monotone constrained tree member, and a PCA-MLP member. "
                    "Unlabeled miner-received forward-audit payloads were used only "
                    "for topology, shape, and train/serve checks. No validator-private "
                    "labels, wallets, hotkeys, IP addresses, deployment logs, or "
                    "private player data were used for training."
                )
            elif family == "v298_lambdamart_wide_ranker":
                training_statement = (
                    "LightGBM LambdaMART wide-feature ranker trained only on public "
                    "Poker44 benchmark releaseVersion v1.13 through sourceDate "
                    "2026-07-10 using miner-visible sanitized hand/action payload "
                    "fields. Features are HG2 wide behavioral aggregates computed "
                    "with train/serve parity; the served head is a monotone top-k "
                    "threshold-safe remap preserving rank order. No competitor "
                    "weights, forward labels, wallets, hotkeys, IP addresses, "
                    "deployment logs, or private player data were used for training."
                )
            elif family == "v306_lambdamart_temporal_seed_ensemble":
                training_statement = (
                    "Two independently seeded LightGBM LambdaMART rankers trained "
                    "only on public Poker44 benchmark releases through sourceDate "
                    "2026-06-30 using miner-visible sanitized fields and 611 "
                    "deterministic wide features. Rank-space weights were selected "
                    "on July 6-7 and evaluated once on untouched July 8-10 batches. "
                    "No validator labels, forward-audit labels, competitor weights, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private "
                    "player data were used for training."
                )
            elif family == "v315_original_lambdamart_top10":
                training_statement = (
                    "AceGuard LightGBM LambdaMART ranker trained only on miner-visible "
                    "public Poker44 benchmark data through sourceDate 2026-06-30. The "
                    "64 selected features are an independent implementation of within-hand "
                    "action transitions, hero responses, pot geometry, position, stack "
                    "context, and order-invariant chunk distributions; identities, cards, "
                    "outcomes, dates, and merge-boundary temporal features are excluded. "
                    "Unlabeled miner payloads received no later than 2026-07-05T23:59:59Z "
                    "were used only for train/live KS filtering. Hyperparameters and the "
                    "top-10 threshold head were selected on July 6-7 and evaluated once on "
                    "untouched July 8-10 data under several bot-prevalence levels. No "
                    "competitor implementation or weights, validator labels, forward labels, "
                    "wallets, hotkeys, IP addresses, or private data were used."
                )
            elif family == "v200_stackseq_last3":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. It is a stacked ensemble of tree learners and a CPU "
                    "chunk-sequence learner trained on the latest public releases with "
                    "2026-07-08 held out for reporting. No validator-private labels, "
                    "wallets, hotkeys, IP addresses, deployment logs, or private player "
                    "data were used for training."
                )
            elif family == "v201_stackseq_wide8":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-08 using miner-visible hand/action payload "
                    "fields only. It is a wider stacked ensemble of tree learners and a CPU "
                    "chunk-sequence learner with wider sequence embeddings, 64-hand context, "
                    "10-action hand caps, and 2026-07-08 held out for reporting. No "
                    "validator-private labels, wallets, hotkeys, IP addresses, deployment "
                    "logs, or private player data were used for training."
                )
            elif family == "v140_multi":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-07 using miner-visible hand/action payload "
                    "fields only; same-date/same-label chunks were merged with multiple "
                    "random seeds into live-sized units for cross-date ranking validation. "
                    "No validator-private labels, wallets, hotkeys, IP addresses, or "
                    "deployment logs were used for training."
                )
            elif family == "v136_live":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "through sourceDate 2026-07-07 using miner-visible hand/action payload "
                    "fields only; live-sized same-date/same-label synthetic chunks were used "
                    "for cross-date ranking validation and no validator-private labels were used."
                )
            elif family == "v133_split":
                training_statement = (
                    "Model trained only on public Poker44 benchmark releaseVersion v1.13 "
                    "using split=train chunk groups; split=validation was held out for "
                    "reporting; features use miner-visible hand/action payload fields only, "
                    "with no chunk IDs, dates, hashes, wallets, or validator-private labels."
                )
            else:
                training_statement = (
                    "Model trained on public Poker44 benchmark releases using "
                    "miner-visible payload views only."
                )
            framework = (
                "python+scikit-learn+torch"
                if family in {"v200_stackseq_last3", "v201_stackseq_wide8"}
                else "python+scikit-learn"
                if family == "v294_hg2_rebuild"
                else "python+scikit-learn"
            )
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
                "model_version": os.getenv(
                    "POKER44_MODEL_VERSION",
                    "2026.07.10-v315"
                    if family == "v315_original_lambdamart_top10"
                    else "2026.07.10-v306"
                    if family == "v306_lambdamart_temporal_seed_ensemble"
                    else "2026.06.17",
                ),
                "framework": framework,
                "license": "MIT",
                "repo_url": os.getenv("POKER44_MODEL_REPO_URL", ""),
                "repo_commit": _repo_commit(),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": training_statement,
                "training_data_sources": (
                    [
                        "https://api.poker44.net/api/v1/benchmark",
                        "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-08",
                    ]
                    if family in {
                        "v173_actioncap8",
                        "v175_actioncap8_ks075",
                        "v179_actioncap8_livehand89_ks060",
                        "v181_actioncap8_livehand89_maxks075_fullheads",
                        "v183_v11lock1_v181rest",
                        "v184_v11lock2_v181rest",
                        "v193_v11lock1_v145rank_rest",
                        "v190_contract60_80_100_ks050_livesized",
                        "v209_served_rankcap_ks055",
                        "v216_served_rankcap_ks060",
                        "v217_v11lock2_v216rest",
                        "v218_withinbatch_behav",
                        "v219_rebuilt_superv2",
                        "v220_v11lock2_v219rest",
                        "v200_stackseq_last3",
                    }
                    else
                    [
                        "https://api.poker44.net/api/v1/benchmark",
                        "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-09",
                    ]
                    if family in {"v221_v11lock2_nightlyrest", "v223_withinbatch_behav_refresh"}
                    or family == "v228_shaped_v223"
                    or family == "v234_behav_mix_v11"
                    or family == "v237_v11lock1_v234rest"
                    or family == "v241_v11lock1_v118rest"
                    or family == "v245_v11lock1_v244rest"
                    or family == "v248_batchrank_schema"
                    or family == "v249_batchrank_behavmix_v11"
                    or family == "v252_clean_top1recipe_schema"
                    or family == "v253_oldwindow_schema"
                    or family == "v255_oldwindow_top1schema"
                    or family == "v257_trainonly_top1schema"
                    or family == "v260_fit80_sanitized_top1schema"
                    or family == "v262_v260w95_v11w5_rankblend"
                    or family == "v264_v260w80_v263w20_rankblend"
                    or family == "v270_v260w98_v263w01_v265w01_rankblend"
                    or family == "v291_v263_rankmean_latest"
                    or family == "v288_top1style_schema"
                    or family == "v292_v26090_v26105_v11_rankblend"
                    or family == "v289_v270w90_v288397avgw10_rankblend"
                    or family == "v290_v289w90_v11w10_runtime"
                    or family == "v287_shape_adaptive_v11_v270"
                    or family == "v271_v11lock1_v268rest"
                    or family == "v274_v11lock1_v273rest"
                    or family == "v276_livesized6080100_v273"
                    or family == "v277_livesized6080100_temporal"
                    or family == "v280_livesized6080100_temporal_consistency"
                    else
                    [
                        "https://api.poker44.net/api/v1/benchmark",
                        "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-07",
                    ]
                    if family in {"v136_live", "v140_multi", "v142_rankblend"}
                    else
                    [
                        "https://api.poker44.net/api/v1/benchmark",
                        "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-06",
                    ]
                    if family == "v133_split"
                    else []
                ),
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
        if family == "v294_hg2_rebuild":
            manifest["training_data_sources"] = [
                "https://api.poker44.net/api/v1/benchmark",
                "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-10",
            ]
        if family == "v297_shape_adaptive_v296_v11_v270":
            manifest["training_data_sources"] = [
                "https://api.poker44.net/api/v1/benchmark",
                "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-10",
            ]
            manifest["training_refresh"] = "v296_rankmlp_wide_v297_shape_router_candidate_2026-07-10"
        if family == "v298_lambdamart_wide_ranker":
            manifest["training_data_sources"] = [
                "https://api.poker44.net/api/v1/benchmark",
                "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-10",
            ]
            manifest["training_refresh"] = "v298_lambdamart_wide_ranker_candidate_2026-07-10"
        if family == "v306_lambdamart_temporal_seed_ensemble":
            manifest["training_data_sources"] = [
                "https://api.poker44.net/api/v1/benchmark",
                "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-10",
            ]
            manifest["training_refresh"] = "v306_temporal_seed_ensemble_candidate_2026-07-10"
        if family == "v315_original_lambdamart_top10":
            manifest["training_data_sources"] = [
                "https://api.poker44.net/api/v1/benchmark",
                "https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-06-30",
            ]
            manifest["training_refresh"] = "v315_original_temporal_holdout_candidate_2026-07-10"
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
        if family == "v131_behav":
            manifest["training_refresh"] = "behavioural_mix_candidate_2026-07-06"
        if family == "v132_ngram":
            manifest["training_refresh"] = "behavioural_ngram_candidate_2026-07-06"
        if family == "v133_split":
            manifest["training_refresh"] = "v113_split_ngram_candidate_2026-07-06"
        if family == "v136_live":
            manifest["training_refresh"] = "v113_livesized_candidate_2026-07-07"
        if family == "v140_multi":
            manifest["training_refresh"] = "v113_multiseed_livesized_candidate_2026-07-07"
        if family == "v142_rankblend":
            manifest["training_refresh"] = "rankblend_livesized_candidate_2026-07-07"
        if family == "v173_actioncap8":
            manifest["training_refresh"] = "actioncap8_noidentity_livesized_candidate_2026-07-08"
        if family == "v175_actioncap8_ks075":
            manifest["training_refresh"] = "actioncap8_multilive_ks075_candidate_2026-07-08"
        if family == "v179_actioncap8_livehand89_ks060":
            manifest["training_refresh"] = "actioncap8_livehand89_ks060_candidate_2026-07-08"
        if family == "v181_actioncap8_livehand89_maxks075_fullheads":
            manifest["training_refresh"] = "actioncap8_livehand89_maxks075_fullheads_segment_candidate_2026-07-08"
        if family == "v183_v11lock1_v181rest":
            manifest["training_refresh"] = "v11_top1_lock_v181_rest_candidate_2026-07-08"
        if family == "v184_v11lock2_v181rest":
            manifest["training_refresh"] = "v11_top2_lock_v181_rest_candidate_2026-07-08"
        if family == "v193_v11lock1_v145rank_rest":
            manifest["training_refresh"] = "v11_top1_lock_v145_human_rank_rest_candidate_2026-07-08"
        if family == "v190_contract60_80_100_ks050_livesized":
            manifest["training_refresh"] = "contract60_80_100_ks050_livesized_candidate_2026-07-08"
        if family == "v209_served_rankcap_ks055":
            manifest["training_refresh"] = "served_rankcap_ks055_livesized_candidate_2026-07-08"
        if family == "v216_served_rankcap_ks060":
            manifest["training_refresh"] = "served_rankcap_ks060_livesized_candidate_2026-07-08"
        if family == "v217_v11lock2_v216rest":
            manifest["training_refresh"] = "v11_top2_lock_v216_rest_candidate_2026-07-08"
        if family == "v218_withinbatch_behav":
            manifest["training_refresh"] = "withinbatch_behav_batch100_candidate_2026-07-08"
        if family == "v219_rebuilt_superv2":
            manifest["training_refresh"] = "rebuilt_superv2_public_v113_candidate_2026-07-08"
        if family == "v220_v11lock2_v219rest":
            manifest["training_refresh"] = "v11_top2_lock_rebuilt_superv2_rest_candidate_2026-07-08"
        if family == "v221_v11lock2_nightlyrest":
            manifest["training_refresh"] = "v11_top2_lock_superseq_rest_candidate_2026-07-09"
        if family == "v223_withinbatch_behav_refresh":
            manifest["training_refresh"] = "withinbatch_behav_refresh_batch100_candidate_2026-07-09"
        if family == "v228_shaped_v223":
            manifest["training_refresh"] = "withinbatch_behav_shaped_cutoff_candidate_2026-07-09"
        if family == "v234_behav_mix_v11":
            manifest["training_refresh"] = "behav_mix_v11_live_sized_candidate_2026-07-09"
        if family == "v237_v11lock1_v234rest":
            manifest["training_refresh"] = "v11lock1_v234rest_candidate_2026-07-09"
        if family == "v241_v11lock1_v118rest":
            manifest["training_refresh"] = "v11lock1_v118rest_candidate_2026-07-09"
        if family == "v245_v11lock1_v244rest":
            manifest["training_refresh"] = "v11lock1_v244rest_candidate_2026-07-09"
        if family == "v248_batchrank_schema":
            manifest["training_refresh"] = "batchrank_schema_public_benchmark_candidate_2026-07-09"
        if family == "v249_batchrank_behavmix_v11":
            manifest["training_refresh"] = "batchrank_behavmix_v11_candidate_2026-07-09"
        if family == "v252_clean_top1recipe_schema":
            manifest["training_refresh"] = "clean_top1recipe_schema_candidate_2026-07-09"
        if family == "v253_oldwindow_schema":
            manifest["training_refresh"] = "oldwindow_schema_candidate_2026-07-09"
        if family == "v255_oldwindow_top1schema":
            manifest["training_refresh"] = "oldwindow_top1schema_candidate_2026-07-09"
        if family == "v257_trainonly_top1schema":
            manifest["training_refresh"] = "trainonly_top1schema_candidate_2026-07-09"
        if family == "v260_fit80_sanitized_top1schema":
            manifest["training_refresh"] = "fit80_sanitized_top1schema_candidate_2026-07-09"
        if family == "v262_v260w95_v11w5_rankblend":
            manifest["training_refresh"] = "v260w95_v11w5_rankblend_candidate_2026-07-09"
        if family == "v264_v260w80_v263w20_rankblend":
            manifest["training_refresh"] = "v260w80_v263w20_rankblend_candidate_2026-07-10"
        if family == "v270_v260w98_v263w01_v265w01_rankblend":
            manifest["training_refresh"] = "v260w98_v263w01_v265w01_rankblend_candidate_2026-07-10"
        if family == "v291_v263_rankmean_latest":
            manifest["training_refresh"] = "v263_rankmean_latest_candidate_2026-07-10"
        if family == "v288_top1style_schema":
            manifest["training_refresh"] = "v288_top1style_schema_candidate_2026-07-10"
        if family == "v292_v26090_v26105_v11_rankblend":
            manifest["training_refresh"] = "v260w90_v261w05_v11w05_rankblend_candidate_2026-07-10"
        if family == "v289_v270w90_v288397avgw10_rankblend":
            manifest["training_refresh"] = "v270w90_v288397avgw10_rankblend_candidate_2026-07-10"
        if family == "v290_v289w90_v11w10_runtime":
            manifest["training_refresh"] = "v289w90_v11w10_runtime_candidate_2026-07-10"
        if family == "v287_shape_adaptive_v11_v270":
            manifest["training_refresh"] = "shape_adaptive_v11_v270_candidate_2026-07-10"
        if family == "v271_v11lock1_v268rest":
            manifest["training_refresh"] = "v11lock1_v268rest_candidate_2026-07-10"
        if family == "v274_v11lock1_v273rest":
            manifest["training_refresh"] = "v11lock1_v273_livesized6080100_candidate_2026-07-10"
        if family == "v276_livesized6080100_v273":
            manifest["training_refresh"] = "v273_livesized6080100_candidate_2026-07-10"
        if family == "v277_livesized6080100_temporal":
            manifest["training_refresh"] = "v277_livesized6080100_temporal_candidate_2026-07-10"
        if family == "v280_livesized6080100_temporal_consistency":
            manifest["training_refresh"] = "v280_livesized6080100_temporal_consistency_candidate_2026-07-10"
        if family == "v294_hg2_rebuild":
            manifest["training_refresh"] = "hg2_rebuild_public_benchmark_candidate_2026-07-10"
        if family == "v200_stackseq_last3":
            manifest["training_refresh"] = "stackseq_last3_public_benchmark_candidate_2026-07-08"
        if family == "v201_stackseq_wide8":
            manifest["training_refresh"] = "stackseq_wide8_public_benchmark_candidate_2026-07-08"
        return manifest

    def _score_stackseq_model(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.rank_cap_remap import rank_cap_remap
        from poker44_ml.inference import Poker44Model

        model_file = os.getenv(
            "POKER44_V201_MODEL_PATH" if self.variant_cfg["family"] == "v201_stackseq_wide8" else "POKER44_V200_MODEL_PATH",
            str(REPO_ROOT / self.variant_cfg["model_file"]),
        )
        model_path = Path(model_file)
        if not model_path.exists():
            bt.logging.error(f"{self.variant_cfg['family']} model missing: {model_path}")
            return [0.49 for _ in chunks]

        cache = getattr(self, "_stackseq_model_cache", {})
        mtime = model_path.stat().st_mtime
        cached = cache.get(str(model_path))
        if cached is None or cached[0] != mtime:
            cache[str(model_path)] = (mtime, Poker44Model(model_path))
            self._stackseq_model_cache = cache
        model = cache[str(model_path)][1]

        strategy = str(self.variant_cfg.get("strategy", "final")).strip().lower()
        if strategy in {"raw", "precal"}:
            raw_scores = model.debug_score_components(chunks).get("raw_scores", [])
        elif strategy in {"cal", "calibrated"}:
            raw_scores = model.debug_score_components(chunks).get("calibrated_scores", [])
        else:
            raw_scores = model.predict_chunk_scores(chunks)
        self._last_raw_scores = [float(v) for v in raw_scores]
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        scores = rank_cap_remap(raw_scores, top_n)
        try:
            bt.logging.info(
                f"{self.variant_cfg['family']} top_n={top_n} "
                f"raw_std={float(np.std(raw_scores)):.4f} "
                f"positives={sum(1 for v in scores if v >= 0.5)}/{len(scores)}"
            )
        except Exception:
            pass
        return [round(float(v), 6) for v in scores]

    def _score_rebuilt_superv2_model(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.rank_cap_remap import rank_cap_remap
        from poker44.score.rebuilt_superv2_inference import score_from_file

        model_file = os.getenv(
            "POKER44_V219_MODEL_PATH",
            str(REPO_ROOT / self.variant_cfg["model_file"]),
        )
        model_path = Path(model_file)
        if not model_path.exists():
            bt.logging.error(f"{self.variant_cfg['family']} model missing: {model_path}")
            return [0.49 for _ in chunks]

        raw_scores = score_from_file(chunks, model_path)
        self._last_raw_scores = [float(v) for v in raw_scores]
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        scores = rank_cap_remap(raw_scores, top_n)
        try:
            bt.logging.info(
                f"{self.variant_cfg['family']} top_n={top_n} "
                f"raw_std={float(np.std(raw_scores)):.4f} "
                f"positives={sum(1 for v in scores if v >= 0.5)}/{len(scores)}"
            )
        except Exception:
            pass
        return [round(float(v), 6) for v in scores]

    def _score_v11lock2_v219rest_model(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.ensemble_v11 import score_chunks_v11
        from poker44.score.rank_cap_remap import rank_cap_remap
        from poker44.score.rebuilt_superv2_inference import score_from_file

        model_file = os.getenv(
            "POKER44_V219_MODEL_PATH",
            str(REPO_ROOT / self.variant_cfg["model_file"]),
        )
        model_path = Path(model_file)
        if not model_path.exists():
            bt.logging.error(f"{self.variant_cfg['family']} model missing: {model_path}")
            return [0.49 for _ in chunks]

        v11_scores, _telemetry, _types = score_chunks_v11(chunks)
        raw_scores = np.asarray(score_from_file(chunks, model_path), dtype=float)
        if len(raw_scores) != len(chunks):
            bt.logging.error(
                f"{self.variant_cfg['family']} returned {len(raw_scores)} scores for {len(chunks)} chunks"
            )
            return [0.49 for _ in chunks]

        if len(raw_scores):
            locked = np.argsort(-np.asarray(v11_scores, dtype=float))[: min(2, len(raw_scores))]
            floor = float(np.nanmax(raw_scores)) if np.isfinite(raw_scores).any() else 0.0
            for rank, idx in enumerate(locked):
                raw_scores[int(idx)] = floor + float(len(locked) - rank)
            shaped = np.zeros_like(raw_scores, dtype=float)
            for rank, idx in enumerate(np.argsort(-raw_scores, kind="mergesort")):
                shaped[int(idx)] = float(len(raw_scores) - rank)
            raw_scores = shaped

        self._last_raw_scores = [float(v) for v in raw_scores]
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        scores = rank_cap_remap(raw_scores, top_n)
        try:
            bt.logging.info(
                f"{self.variant_cfg['family']} top_n={top_n} "
                f"raw_std={float(np.std(raw_scores)):.4f} "
                f"positives={sum(1 for v in scores if v >= 0.5)}/{len(scores)}"
            )
        except Exception:
            pass
        return [round(float(v), 6) for v in scores]

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

    def _batch_action_shape(self, chunks: list[list[dict[str, Any]]]) -> dict[str, float]:
        actions_total = 0
        preflop_count = 0
        fold_count = 0
        for chunk in chunks:
            for hand in chunk:
                actions = hand.get("actions") if isinstance(hand, dict) else None
                if not isinstance(actions, list):
                    continue
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    actions_total += 1
                    street = str(action.get("street", "")).lower()
                    action_type = str(action.get("action_type", action.get("type", ""))).lower()
                    if street == "preflop":
                        preflop_count += 1
                    if action_type == "fold":
                        fold_count += 1
        denom = max(actions_total, 1)
        return {
            "actions_total": float(actions_total),
            "preflop_share": float(preflop_count / denom),
            "fold_share": float(fold_count / denom),
        }

    def _rank01(self, values: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty(arr.size, dtype=float)
        ranks[order] = (np.arange(arr.size, dtype=float) + 1.0) / max(arr.size, 1)
        return ranks

    def _score_v290_runtime_model(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.ensemble_v11 import score_chunks_v11
        from poker44.score.rank_cap_remap import rank_cap_remap
        from poker44.score.v112_super_inference import score_from_file

        model_file = os.getenv(
            "POKER44_V290_MODEL_PATH",
            os.getenv("POKER44_V289_MODEL_PATH", str(REPO_ROOT / self.variant_cfg["model_file"])),
        )
        model_path = Path(model_file)
        if not model_path.exists():
            bt.logging.error(f"{self.variant_cfg['family']} model missing: {model_path}")
            return [0.49 for _ in chunks]

        v289_scores = np.asarray(score_from_file(chunks, model_path, strategy="ladder_rank_mean"), dtype=float)
        v11_scores, _telemetry, _types = score_chunks_v11(chunks)
        v11_scores = np.asarray(v11_scores, dtype=float)
        blended = 0.90 * self._rank01(v289_scores) + 0.10 * self._rank01(v11_scores)
        self._last_raw_scores = [float(v) for v in blended]
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        scores = rank_cap_remap(blended, top_n)
        self._last_score_extra = {
            "branch": "v290_runtime",
            "v289_weight": 0.90,
            "v11_weight": 0.10,
            "top_n": int(top_n),
            "v289_std": round(float(np.std(v289_scores)), 8),
            "v11_std": round(float(np.std(v11_scores)), 8),
        }
        try:
            bt.logging.info(
                f"{self.variant_cfg['family']} runtime top_n={top_n} "
                f"v289_std={float(np.std(v289_scores)):.4f} "
                f"v11_std={float(np.std(v11_scores)):.4f} "
                f"positives={sum(1 for v in scores if v >= 0.5)}/{len(scores)}"
            )
        except Exception:
            pass
        return [round(float(v), 6) for v in scores]

    def _score_v287_shape_adaptive_model(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.calibration import rank_based_calibrate
        from poker44.score.ensemble_v11 import score_chunks_v11
        from poker44.score.rank_cap_remap import rank_cap_remap
        from poker44.score.v112_super_inference import score_from_file

        shape = self._batch_action_shape(chunks)
        use_schema_branch = shape["preflop_share"] >= 0.735
        if use_schema_branch:
            model_file = os.getenv(
                "POKER44_V287_MODEL_PATH",
                os.getenv("POKER44_V270_MODEL_PATH", str(REPO_ROOT / self.variant_cfg["model_file"])),
            )
            model_path = Path(model_file)
            if not model_path.exists():
                bt.logging.error(f"{self.variant_cfg['family']} model missing: {model_path}")
                return [0.49 for _ in chunks]
            raw_scores = score_from_file(chunks, model_path, strategy="ladder_rank_mean")
            self._last_raw_scores = [float(v) for v in raw_scores]
            top_n = _env_int("POKER44_V287_SCHEMA_TOP_N", 20)
            scores = rank_cap_remap(raw_scores, top_n)
            branch = "v270_ladder_rank_mean"
        else:
            raw_scores, _telemetry, _types = score_chunks_v11(chunks)
            self._last_raw_scores = [float(v) for v in raw_scores]
            top_n = _env_int("POKER44_V287_V11_TOP_N", 2)
            bot_ratio = min(max(top_n / max(len(raw_scores), 1), 0.0), 0.1)
            scores = rank_based_calibrate(raw_scores, bot_ratio=bot_ratio)
            branch = "v11"

        try:
            bt.logging.info(
                f"{self.variant_cfg['family']} branch={branch} top_n={top_n} "
                f"preflop_share={shape['preflop_share']:.4f} "
                f"fold_share={shape['fold_share']:.4f} "
                f"raw_std={float(np.std(raw_scores)):.4f} "
                f"positives={sum(1 for v in scores if v >= 0.5)}/{len(scores)}"
            )
        except Exception:
            pass
        self._last_score_extra = {
            "branch": branch,
            "preflop_share": round(float(shape["preflop_share"]), 8),
            "fold_share": round(float(shape["fold_share"]), 8),
            "actions_total": int(shape["actions_total"]),
            "top_n": int(top_n),
        }
        return [round(float(v), 6) for v in scores]

    def _score_v296_rankmlp_branch(
        self,
        chunks: list[list[dict[str, Any]]],
        model_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        import pickle
        import sys
        from scipy.stats import rankdata

        runtime_dir = REPO_ROOT / "poker44" / "score" / "hg2_runtime"
        runtime_s = str(runtime_dir)
        if runtime_s not in sys.path:
            sys.path.insert(0, runtime_s)
        module = sys.modules.get("hg_features")
        loaded_from = str(getattr(module, "__file__", "")) if module is not None else ""
        if module is not None and loaded_from and runtime_s not in loaded_from:
            sys.modules.pop("hg_features", None)
        from hg_features import wide_view

        runtime = getattr(self, "_v296_runtime", None)
        runtime_path = getattr(self, "_v296_runtime_path", None)
        if runtime is None or runtime_path != str(model_path):
            with model_path.open("rb") as handle:
                runtime = pickle.load(handle)
            self._v296_runtime = runtime
            self._v296_runtime_path = str(model_path)

        keys = list(runtime["keys"])
        rows = [wide_view(chunk or []) for chunk in chunks]
        x = np.asarray([[float(row.get(key, 0.0)) for key in keys] for row in rows], dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        ranked = np.zeros_like(x, dtype=float)
        denom = max(x.shape[0], 1)
        for col in range(x.shape[1]):
            ranked[:, col] = rankdata(x[:, col], method="average") / denom

        proba = runtime["model"].predict_proba(ranked)
        raw = np.asarray(proba[:, 1] if proba.ndim == 2 else proba, dtype=float)
        threshold = min(max(float(runtime.get("deploy_threshold", 0.5)), 1e-6), 1.0 - 1e-6)
        scores = np.where(
            raw >= threshold,
            0.5 + 0.5 * (raw - threshold) / (1.0 - threshold),
            0.5 * raw / threshold,
        )
        scores = np.clip(scores, 0.0, 1.0)
        max_frac = float(
            os.getenv(
                "POKER44_V297_MAX_POS_FRAC",
                str(runtime.get("max_positive_fraction", 0.15)),
            )
        )
        if scores.size and max_frac < 1.0:
            k = max(1, int(np.floor(scores.size * max_frac)))
            positive = np.flatnonzero(scores >= 0.5)
            if positive.size > k:
                order = positive[np.argsort(-scores[positive], kind="stable")]
                squeeze = order[k:]
                below = scores[scores < 0.5]
                lo = min(float(below.max()) if below.size else 0.45, 0.499)
                span = 0.5 - lo
                m = len(squeeze)
                for rank, idx in enumerate(squeeze):
                    scores[idx] = lo + span * (m - rank) / (m + 1.0)
        return raw, np.clip(scores, 0.0, 1.0)

    def _score_v297_shape_adaptive_model(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        from poker44.score.calibration import rank_based_calibrate
        from poker44.score.ensemble_v11 import score_chunks_v11
        from poker44.score.rank_cap_remap import rank_cap_remap
        from poker44.score.v112_super_inference import score_from_file

        shape = self._batch_action_shape(chunks)
        preflop_share = float(shape["preflop_share"])
        if preflop_share >= 0.735:
            branch = "v270_ladder_rank_mean"
            model_file = os.getenv(
                "POKER44_V297_V270_MODEL_PATH",
                os.getenv(
                    "POKER44_V270_MODEL_PATH",
                    str(REPO_ROOT / "data" / "models" / "v270_v260w98_v263w01_v265w01_rankblend" / "model.pkl"),
                ),
            )
            model_path = Path(model_file)
            if not model_path.exists():
                bt.logging.error(f"{self.variant_cfg['family']} v270 model missing: {model_path}")
                return [0.49 for _ in chunks]
            raw_scores = np.asarray(score_from_file(chunks, model_path, strategy="ladder_rank_mean"), dtype=float)
            top_n = _env_int("POKER44_V297_SCHEMA_TOP_N", 20)
            scores = np.asarray(rank_cap_remap(raw_scores, top_n), dtype=float)
        elif preflop_share <= 0.705 or 0.713 <= preflop_share <= 0.723:
            branch = "v296_rankmlp_wide"
            model_file = os.getenv(
                "POKER44_V297_MODEL_PATH",
                os.getenv("POKER44_V296_MODEL_PATH", str(REPO_ROOT / self.variant_cfg["model_file"])),
            )
            model_path = Path(model_file)
            if not model_path.exists():
                bt.logging.error(f"{self.variant_cfg['family']} v296 model missing: {model_path}")
                return [0.49 for _ in chunks]
            raw_scores, scores = self._score_v296_rankmlp_branch(chunks, model_path)
            top_n = int(sum(1 for value in scores if float(value) >= 0.5))
        else:
            branch = "v11"
            raw_scores, _telemetry, _types = score_chunks_v11(chunks)
            raw_scores = np.asarray(raw_scores, dtype=float)
            top_n = _env_int("POKER44_V297_V11_TOP_N", 2)
            bot_ratio = min(max(top_n / max(len(raw_scores), 1), 0.0), 0.1)
            scores = np.asarray(rank_based_calibrate(raw_scores, bot_ratio=bot_ratio), dtype=float)

        self._last_raw_scores = [float(v) for v in raw_scores]
        self._last_score_extra = {
            "branch": branch,
            "preflop_share": round(float(preflop_share), 8),
            "fold_share": round(float(shape["fold_share"]), 8),
            "actions_total": int(shape["actions_total"]),
            "top_n": int(top_n),
        }
        try:
            bt.logging.info(
                f"{self.variant_cfg['family']} branch={branch} top_n={top_n} "
                f"preflop_share={shape['preflop_share']:.4f} "
                f"fold_share={shape['fold_share']:.4f} "
                f"raw_std={float(np.std(raw_scores)):.4f} "
                f"positives={sum(1 for v in scores if float(v) >= 0.5)}/{len(scores)}"
            )
        except Exception:
            pass
        return [round(float(v), 6) for v in scores]

    def _score_v298_lambdamart_wide_model(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        family = self.variant_cfg["family"]
        is_v306 = family == "v306_lambdamart_temporal_seed_ensemble"
        is_v315 = family == "v315_original_lambdamart_top10"
        if is_v315:
            from poker44.score.original_lambdamart_inference import load_bundle, score_chunks
        else:
            from poker44.score.lambdamart_wide_inference import load_bundle, score_chunks

        env_prefix = "POKER44_V315" if is_v315 else "POKER44_V306" if is_v306 else "POKER44_V298"
        model_file = os.getenv(
            f"{env_prefix}_MODEL_PATH",
            str(REPO_ROOT / self.variant_cfg["model_file"]),
        )
        model_path = Path(model_file)
        if not model_path.exists():
            bt.logging.error(f"{self.variant_cfg['family']} model missing: {model_path}")
            return [0.49 for _ in chunks]

        runtime = getattr(self, "_v298_runtime", None)
        runtime_path = getattr(self, "_v298_runtime_path", None)
        if runtime is None or runtime_path != str(model_path):
            runtime = load_bundle(model_path)
            self._v298_runtime = runtime
            self._v298_runtime_path = str(model_path)

        raw_scores = np.asarray(score_chunks(chunks, runtime), dtype=float)
        top_n = _env_int(
            f"{env_prefix}_TOP_N",
            int(self.variant_cfg.get("default_top_n", runtime.get("top_n", 2))),
        )
        scores = np.asarray(_banded_rank_remap(raw_scores, top_n), dtype=float)

        self._last_raw_scores = [float(v) for v in raw_scores]
        self._last_score_extra = {
            "branch": self.variant_cfg["family"],
            "top_n": int(top_n),
            "raw_std": round(float(np.std(raw_scores)), 8) if raw_scores.size else 0.0,
            "positive_count": int(np.sum(scores >= 0.5)),
        }
        try:
            bt.logging.info(
                f"{self.variant_cfg['family']} top_n={top_n} "
                f"raw_std={float(np.std(raw_scores)):.4f} "
                f"positives={sum(1 for v in scores if float(v) >= 0.5)}/{len(scores)}"
            )
        except Exception:
            pass
        return [round(float(v), 6) for v in scores]

    def _score_v294_hg2_model(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        import importlib
        import sys

        runtime_dir = REPO_ROOT / "poker44" / "score" / "hg2_runtime"
        model_path = Path(
            os.getenv(
                "POKER44_V294_MODEL_PATH",
                str(REPO_ROOT / self.variant_cfg["model_file"]),
            )
        )
        if not model_path.exists():
            bt.logging.error(f"{self.variant_cfg['family']} model missing: {model_path}")
            return [0.49 for _ in chunks]
        if not (model_path.parent / "meta.json").exists():
            bt.logging.error(f"{self.variant_cfg['family']} meta missing: {model_path.parent / 'meta.json'}")
            return [0.49 for _ in chunks]

        runtime_s = str(runtime_dir)
        if runtime_s not in sys.path:
            sys.path.insert(0, runtime_s)
        for module_name in ("infer", "hg_features", "features_v2", "hg2_features_base", "hg_model"):
            module = sys.modules.get(module_name)
            loaded_from = str(getattr(module, "__file__", "")) if module is not None else ""
            if module is not None and loaded_from and runtime_s not in loaded_from:
                sys.modules.pop(module_name, None)

        top_n = _env_int("POKER44_V294_TOP_N", self.variant_cfg["default_top_n"])
        max_frac_default = min(max(float(top_n) / max(len(chunks), 1), 0.01), 0.30)
        max_pos_frac = float(os.getenv("POKER44_V294_MAX_POS_FRAC", f"{max_frac_default:.8f}"))
        os.environ["POKER44_ARTIFACT"] = model_path.name
        os.environ["POKER44_MAX_POS_FRAC"] = f"{max_pos_frac:.8f}"

        infer = importlib.import_module("infer")
        setattr(infer, "_ARTIFACT", model_path.name)
        setattr(infer, "_MAX_POS_FRAC", max_pos_frac)
        runtime = getattr(self, "_v294_runtime", None)
        runtime_path = getattr(self, "_v294_runtime_path", None)
        if runtime is None or runtime_path != str(model_path):
            runtime = infer.ServingModel(art_dir=str(model_path.parent))
            self._v294_runtime = runtime
            self._v294_runtime_path = str(model_path)

        scores = [float(v) for v in runtime.score_chunks(chunks)]
        self._last_raw_scores = list(scores)
        positives = sum(1 for v in scores if v >= 0.5)
        self._last_score_extra = {
            "branch": "v294_hg2_rebuild",
            "top_n": int(top_n),
            "max_pos_frac": round(float(max_pos_frac), 8),
            "artifact": model_path.name,
        }
        try:
            bt.logging.info(
                f"{self.variant_cfg['family']} top_n={top_n} "
                f"max_pos_frac={max_pos_frac:.4f} "
                f"raw_std={float(np.std(scores)):.4f} "
                f"positives={positives}/{len(scores)}"
            )
        except Exception:
            pass
        return [round(float(v), 6) for v in scores]

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
        elif self.variant_cfg["family"] == "v132_ngram":
            env_name = "POKER44_V132_MODEL_PATH"
        elif self.variant_cfg["family"] == "v133_split":
            env_name = "POKER44_V133_MODEL_PATH"
        elif self.variant_cfg["family"] == "v136_live":
            env_name = "POKER44_V136_MODEL_PATH"
        elif self.variant_cfg["family"] == "v140_multi":
            env_name = "POKER44_V140_MODEL_PATH"
        elif self.variant_cfg["family"] == "v142_rankblend":
            env_name = "POKER44_V142_MODEL_PATH"
        elif self.variant_cfg["family"] == "v173_actioncap8":
            env_name = "POKER44_V173_MODEL_PATH"
        elif self.variant_cfg["family"] == "v175_actioncap8_ks075":
            env_name = "POKER44_V175_MODEL_PATH"
        elif self.variant_cfg["family"] == "v179_actioncap8_livehand89_ks060":
            env_name = "POKER44_V179_MODEL_PATH"
        elif self.variant_cfg["family"] == "v181_actioncap8_livehand89_maxks075_fullheads":
            env_name = "POKER44_V181_MODEL_PATH"
        elif self.variant_cfg["family"] == "v183_v11lock1_v181rest":
            env_name = "POKER44_V183_MODEL_PATH"
        elif self.variant_cfg["family"] == "v184_v11lock2_v181rest":
            env_name = "POKER44_V184_MODEL_PATH"
        elif self.variant_cfg["family"] == "v193_v11lock1_v145rank_rest":
            env_name = "POKER44_V193_MODEL_PATH"
        elif self.variant_cfg["family"] == "v190_contract60_80_100_ks050_livesized":
            env_name = "POKER44_V190_MODEL_PATH"
        elif self.variant_cfg["family"] == "v209_served_rankcap_ks055":
            env_name = "POKER44_V209_MODEL_PATH"
        elif self.variant_cfg["family"] == "v216_served_rankcap_ks060":
            env_name = "POKER44_V216_MODEL_PATH"
        elif self.variant_cfg["family"] == "v217_v11lock2_v216rest":
            env_name = "POKER44_V217_MODEL_PATH"
        elif self.variant_cfg["family"] == "v218_withinbatch_behav":
            env_name = "POKER44_V218_MODEL_PATH"
        elif self.variant_cfg["family"] == "v223_withinbatch_behav_refresh":
            env_name = "POKER44_V223_MODEL_PATH"
        elif self.variant_cfg["family"] == "v228_shaped_v223":
            env_name = "POKER44_V228_MODEL_PATH"
        elif self.variant_cfg["family"] == "v234_behav_mix_v11":
            env_name = "POKER44_V234_MODEL_PATH"
        elif self.variant_cfg["family"] == "v237_v11lock1_v234rest":
            env_name = "POKER44_V237_MODEL_PATH"
        elif self.variant_cfg["family"] == "v241_v11lock1_v118rest":
            env_name = "POKER44_V241_MODEL_PATH"
        elif self.variant_cfg["family"] == "v245_v11lock1_v244rest":
            env_name = "POKER44_V245_MODEL_PATH"
        elif self.variant_cfg["family"] == "v248_batchrank_schema":
            env_name = "POKER44_V248_MODEL_PATH"
        elif self.variant_cfg["family"] == "v249_batchrank_behavmix_v11":
            env_name = "POKER44_V249_MODEL_PATH"
        elif self.variant_cfg["family"] == "v252_clean_top1recipe_schema":
            env_name = "POKER44_V252_MODEL_PATH"
        elif self.variant_cfg["family"] == "v253_oldwindow_schema":
            env_name = "POKER44_V253_MODEL_PATH"
        elif self.variant_cfg["family"] == "v255_oldwindow_top1schema":
            env_name = "POKER44_V255_MODEL_PATH"
        elif self.variant_cfg["family"] == "v257_trainonly_top1schema":
            env_name = "POKER44_V257_MODEL_PATH"
        elif self.variant_cfg["family"] == "v260_fit80_sanitized_top1schema":
            env_name = "POKER44_V260_MODEL_PATH"
        elif self.variant_cfg["family"] == "v262_v260w95_v11w5_rankblend":
            env_name = "POKER44_V262_MODEL_PATH"
        elif self.variant_cfg["family"] == "v264_v260w80_v263w20_rankblend":
            env_name = "POKER44_V264_MODEL_PATH"
        elif self.variant_cfg["family"] == "v270_v260w98_v263w01_v265w01_rankblend":
            env_name = "POKER44_V270_MODEL_PATH"
        elif self.variant_cfg["family"] == "v291_v263_rankmean_latest":
            env_name = "POKER44_V291_MODEL_PATH"
        elif self.variant_cfg["family"] == "v288_top1style_schema":
            env_name = "POKER44_V288_MODEL_PATH"
        elif self.variant_cfg["family"] == "v292_v26090_v26105_v11_rankblend":
            env_name = "POKER44_V292_MODEL_PATH"
        elif self.variant_cfg["family"] == "v289_v270w90_v288397avgw10_rankblend":
            env_name = "POKER44_V289_MODEL_PATH"
        elif self.variant_cfg["family"] == "v271_v11lock1_v268rest":
            env_name = "POKER44_V271_MODEL_PATH"
        elif self.variant_cfg["family"] == "v274_v11lock1_v273rest":
            env_name = "POKER44_V274_MODEL_PATH"
        elif self.variant_cfg["family"] == "v276_livesized6080100_v273":
            env_name = "POKER44_V276_MODEL_PATH"
        elif self.variant_cfg["family"] == "v277_livesized6080100_temporal":
            env_name = "POKER44_V277_MODEL_PATH"
        elif self.variant_cfg["family"] == "v280_livesized6080100_temporal_consistency":
            env_name = "POKER44_V280_MODEL_PATH"
        elif self.variant_cfg["family"] in {"v200_stackseq_last3", "v201_stackseq_wide8"}:
            return self._score_stackseq_model(chunks)
        else:
            env_name = "POKER44_V112_SUPER_MODEL_PATH"
        model_file = os.getenv(env_name, str(REPO_ROOT / self.variant_cfg["model_file"]))
        model_path = Path(model_file)
        if not model_path.exists():
            bt.logging.error(f"{self.variant_cfg['family']} model missing: {model_path}")
            return [0.49 for _ in chunks]

        strategy = str(self.variant_cfg.get("strategy", "rank_mean") or "rank_mean")
        if strategy.startswith("banded_"):
            base_strategy = strategy.removeprefix("banded_") or "rank_mean"
            raw_scores = score_from_file(chunks, model_path, strategy=base_strategy)
        else:
            raw_scores = score_from_file(chunks, model_path, strategy=strategy)
        self._last_raw_scores = [float(v) for v in raw_scores]
        top_n = _env_int("POKER44_MAX_N", self.variant_cfg["default_top_n"])
        if strategy.startswith("banded_"):
            scores = _banded_rank_remap(raw_scores, top_n)
        elif self.variant_cfg["family"] == "v228_shaped_v223":
            scores = [float(np.clip(v, 0.0, 1.0)) for v in raw_scores]
        else:
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
            self._last_score_extra = {}
            if family == "v5":
                scores = self._score_v5(chunks)
            elif family == "v10":
                scores = self._score_v10(chunks)
            elif family == "v8_markov":
                scores = self._score_v8_markov(chunks)
            elif family == "v11":
                scores = self._score_v11(chunks)
            elif family == "v290_v289w90_v11w10_runtime":
                scores = self._score_v290_runtime_model(chunks)
            elif family == "v287_shape_adaptive_v11_v270":
                scores = self._score_v287_shape_adaptive_model(chunks)
            elif family == "v297_shape_adaptive_v296_v11_v270":
                scores = self._score_v297_shape_adaptive_model(chunks)
            elif family in {
                "v298_lambdamart_wide_ranker",
                "v306_lambdamart_temporal_seed_ensemble",
                "v315_original_lambdamart_top10",
            }:
                scores = self._score_v298_lambdamart_wide_model(chunks)
            elif family == "v294_hg2_rebuild":
                scores = self._score_v294_hg2_model(chunks)
            elif family in {
                "v112_super",
                "v113_daily",
                "v115_short",
                "v118_live",
                "v118_stable75",
                "v118_seg35",
                "v125_topk",
                "v131_behav",
                "v132_ngram",
                "v133_split",
                "v136_live",
                "v140_multi",
                "v142_rankblend",
                "v173_actioncap8",
                "v175_actioncap8_ks075",
                "v179_actioncap8_livehand89_ks060",
                "v181_actioncap8_livehand89_maxks075_fullheads",
                "v183_v11lock1_v181rest",
                "v184_v11lock2_v181rest",
                "v193_v11lock1_v145rank_rest",
                "v190_contract60_80_100_ks050_livesized",
                "v209_served_rankcap_ks055",
                "v216_served_rankcap_ks060",
                "v217_v11lock2_v216rest",
                "v218_withinbatch_behav",
                "v219_rebuilt_superv2",
                "v220_v11lock2_v219rest",
                "v221_v11lock2_nightlyrest",
                "v223_withinbatch_behav_refresh",
                "v228_shaped_v223",
                "v234_behav_mix_v11",
                "v237_v11lock1_v234rest",
                "v241_v11lock1_v118rest",
                "v245_v11lock1_v244rest",
                "v248_batchrank_schema",
                "v249_batchrank_behavmix_v11",
                "v252_clean_top1recipe_schema",
                "v253_oldwindow_schema",
                "v255_oldwindow_top1schema",
                "v257_trainonly_top1schema",
                "v260_fit80_sanitized_top1schema",
                "v262_v260w95_v11w5_rankblend",
                "v264_v260w80_v263w20_rankblend",
            "v270_v260w98_v263w01_v265w01_rankblend",
            "v291_v263_rankmean_latest",
            "v288_top1style_schema",
            "v292_v26090_v26105_v11_rankblend",
                "v289_v270w90_v288397avgw10_rankblend",
                "v271_v11lock1_v268rest",
                "v274_v11lock1_v273rest",
                "v276_livesized6080100_v273",
                "v277_livesized6080100_temporal",
                "v280_livesized6080100_temporal_consistency",
                "v200_stackseq_last3",
                "v201_stackseq_wide8",
            }:
                if family == "v219_rebuilt_superv2":
                    scores = self._score_rebuilt_superv2_model(chunks)
                elif family == "v220_v11lock2_v219rest":
                    scores = self._score_v11lock2_v219rest_model(chunks)
                else:
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
            score_extra=getattr(self, "_last_score_extra", {}),
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
