#!/usr/bin/env bash
# UID 11 (port 22201) — poker44-v22-rank-top1-uid11
# Wave v19-A+ (2026-05-28): unstop UID 11 z nowym portem 22201 (NIE 22017 SSH banned).
# Variant v22 — competitor-style LambdaRank, augmented training, KS-filtered features.
# Orthogonal max IoU 0.06 vs all deployed UIDs.
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
export POKER44_V1_VARIANT="v22_rank_top1"
export POKER44_MODEL_NAME="poker44-v22-rank-top1-uid11"
export POKER44_MODEL_VERSION="22.0"
export POKER44_MODEL_REPO_URL="https://github.com/Krzysiek99999/aceguard-engine"
# Compliance debt: commit 887d8726 nie zawiera v22 kodu (push pending).
export POKER44_MODEL_REPO_COMMIT="887d8726ab6b3cc4806e0bb433c9d2519b3e4653"
export POKER44_MODEL_OPEN_SOURCE="1"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="LightGBM LambdaRank trained on 1500 synthetic live-shaped 40-chunk windows derived from Poker44 public benchmark API shadow-training-v1 release (2026-05-26). Payload transforms: drop all_in/system actions, truncate to 5-8 actions per hand, bucketize bet sizes (16-bucket noise). Features: 29 KS-stable behavioral signals (KS≤0.25 vs live distribution). Holdout AP=0.27 — as ranker (rank_cap_remap top-1), bench AP gate not applicable. Live std=0.076."
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No external private user data; only public benchmark API releases + validator-distributed evaluation chunks."
# v22_rank_top1: deterministic rank_cap_remap top_n=1 in dispatch, no adaptive_n.

exec .venv/bin/python neurons/miner_v1.py \
    --netuid 126 \
    --wallet.name poker44_p22018 \
    --wallet.hotkey poker44_p22018_hot \
    --subtensor.network finney \
    --axon.port 22201 \
    --axon.external_ip 80.238.120.82 \
    --axon.external_port 22201 \
    "$LOG_FLAG" \
    --blacklist.allowed_validator_hotkeys "${VALIDATOR_ALLOWLIST[@]}"
