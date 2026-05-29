#!/usr/bin/env bash
# UID 131 (port 22016) — poker44-v14-rank-top1-uid131
# Wave 1 swap (2026-05-27): response_curves dead (per-seat aliasing) → v14 LightGBM + rank_cap top1.
# Live std=0.22, pearson ~-0.04 vs UID 160 (orthogonal).
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_FLAG="--logging.debug"

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
export POKER44_V1_VARIANT="v14_rank_top1"
export POKER44_MODEL_NAME="poker44-v14-rank-top1-uid131"
export POKER44_MODEL_VERSION="14.0"
export POKER44_MODEL_REPO_URL="https://github.com/Krzysiek99999/aceguard-engine"
# Compliance debt: commit 887d8726 nie zawiera v14_inference.py / v14_live_stable.txt (push pending).
export POKER44_MODEL_REPO_COMMIT="887d8726ab6b3cc4806e0bb433c9d2519b3e4653"
export POKER44_MODEL_OPEN_SOURCE="1"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="LightGBM v14 trained on public benchmark (2026-04-30..05-06 train, 05-07 val, 05-08 holdout) with KS-filter (live-distribution-stable 21 features) + rank_cap_remap top-1 binarization. Hold-out AP=0.79 FPR=0.04 reward=0.67. Live std=0.22."
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No external private user data; only validator-distributed evaluation chunks. Training data from poker44.net/api/v1/benchmark public release."
# v14_rank_top1: rank_cap_remap top_n=1 w dispatch, brak adaptive_n.

exec .venv/bin/python neurons/miner_v1.py \
    --netuid 126 \
    --wallet.name poker44_p22024 \
    --wallet.hotkey poker44_p22024_hot \
    --subtensor.network finney \
    --axon.port 22016 \
    --axon.external_ip 80.238.120.82 \
    --axon.external_port 22016 \
    "$LOG_FLAG" \
    --blacklist.allowed_validator_hotkeys "${VALIDATOR_ALLOWLIST[@]}"
