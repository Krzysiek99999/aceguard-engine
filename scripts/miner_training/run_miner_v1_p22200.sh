#!/usr/bin/env bash
# UID 89 (port 22200) — poker44-v19-rank-top1-uid89
# Wave v19-A (2026-05-28): v9_type_calibrated (jac 0.57 vs UID160) → v19 LambdaRank top1.
# Conservative slot: 1 bot/40 per request, orthogonal do v5/v14 (IoU<0.20 offline).
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
export POKER44_V1_VARIANT="v19_rank_top1"
export POKER44_MODEL_NAME="poker44-v19-rank-top1-uid89"
export POKER44_MODEL_VERSION="19.0"
export POKER44_MODEL_REPO_URL="https://github.com/Krzysiek99999/aceguard-engine"
# Compliance debt: commit 887d8726 nie zawiera v19_inference.py / v19_ranker.txt (push pending).
export POKER44_MODEL_REPO_COMMIT="887d8726ab6b3cc4806e0bb433c9d2519b3e4653"
export POKER44_MODEL_OPEN_SOURCE="1"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="LightGBM LambdaRank trained on Poker44 public benchmark API shadow-training-v1 release (2026-05-26, 90 chunks, 6 release groups, 50/50 balanced synthetic bots vs human anchors). Leave-one-release-out cross-validation OOF AP=0.84. Rank-cap remap top-1 binarization per 40-chunk window."
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No external private user data; only public benchmark API releases + validator-distributed evaluation chunks."
# v19_rank_top1: deterministic rank_cap_remap top_n=1 in dispatch, NO adaptive_n / cap envs.

exec .venv/bin/python neurons/miner_v1.py \
    --netuid 126 \
    --wallet.name poker44_p22021 \
    --wallet.hotkey poker44_hybrid_c \
    --subtensor.network finney \
    --axon.port 22200 \
    --axon.external_ip 80.238.120.82 \
    --axon.external_port 22200 \
    "$LOG_FLAG" \
    --blacklist.allowed_validator_hotkeys "${VALIDATOR_ALLOWLIST[@]}"
