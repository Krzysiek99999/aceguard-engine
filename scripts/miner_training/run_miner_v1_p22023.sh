#!/usr/bin/env bash
# UID 160 (port 22023) — STRATEGY A: voting ensemble cap 0.08
# A/B split from 2026-05-09: replaces previous neurons/miner.py CONTROL.
#   - Strategy A (voting ensemble): UID 163 (cap 0.10) + UID 160 (cap 0.08)
#   - Strategy B (codex v1_real_2026): UID 173 (cap 0.45) + UID 192 (cap 0.52)
# Voting ensemble loads 3 LightGBM tags: v1_top1_v3, v1_top1_v2, B_deeper.
# Uses agreement_2of3 strategy. Per memory: lokalny reward 0.445 vs baseline 0.419 (+5-8pp).
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_FLAG="--logging.debug"
[[ "${POKER44_TRACE:-0}" == "1" ]] && LOG_FLAG="--logging.trace"

VALIDATOR_ALLOWLIST=(
    5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u
    5EP9fmtknrTnDhQmLRY9ciFYoM7YZM8rPWvQ9J7yywEsn126
    5FxQcdsCXcNjWowQ63Y2oeMhN3JRQksejV3aHRr4XmtknM2k
    5HWe7T96SrY4vRvaLmSoriUJ2CGvhRc559U1vZ1pNPuyz2VA
    5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD
    5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp
    5G9hfkx9wGB1CLMT9WXkpHSAiYzjZb5o1Boyq4KAdDhjwrc5
    5HmkWGB5PVzKCNLB4QxWWHFVEHPAbKKxGyoXW7Evs38gs126
    5DqrUa2z6E9taJdY8FGiPCrtCswsEjHjPbVo5xcTw2GqvKZm
    5Hftk9jrMGSJtKBPWkkAkU53FUSr2BqHGPCThg7mbob3hEq1
)

export KMP_DUPLICATE_LIB_OK=TRUE
# 2026-05-18 21:00: UID 160 = CANARY for statistical detector v5.
# v2_benchmark_single saturated on live (RAW=0.651 std=0 for all 40 chunks) because
# benchmark API training distribution ≠ live validator chunks (no "other" action_type).
# v5 ranks chunks by behavioral anomaly (sizing/sequence/quantization) — no model training.
# Codex iter 19: "Statistical detectors most robust for unknown domains."
export POKER44_V1_VARIANT="v5_statistical"
export POKER44_MODEL_NAME="poker44-v5-statistical-uid160"
export POKER44_MODEL_VERSION="5.0"
export POKER44_MODEL_REPO_URL="https://github.com/Krzysiek99999/aceguard-engine"
export POKER44_MODEL_REPO_COMMIT="887d8726ab6b3cc4806e0bb433c9d2519b3e4653"
export POKER44_MODEL_OPEN_SOURCE="1"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="LightGBM single model trained on Poker44 public benchmark API (https://api.poker44.net/api/v1/benchmark) — 7 days train + 1 day val + 1 day holdout; codex iter 4/7/14 hyperparams."
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No external private user data; only validator-distributed evaluation chunks (40 hands each) collected from our own miner's dendrite calls."
export POKER44_DISABLE_LIVE_LOG=1

exec .venv/bin/python neurons/miner_v1.py \
    --netuid 126 \
    --wallet.name poker44_p22023 \
    --wallet.hotkey poker44_p22023_hot \
    --subtensor.network finney \
    --axon.port 22023 \
    --axon.external_ip 80.238.120.82 \
    --axon.external_port 22023 \
    "$LOG_FLAG" \
    --blacklist.allowed_validator_hotkeys "${VALIDATOR_ALLOWLIST[@]}"
