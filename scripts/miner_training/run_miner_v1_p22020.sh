#!/usr/bin/env bash
# UID 226 (port 22020) — poker44-v5-invert-top3-uid226
# Wave 1 swap (2026-05-27): v9 redundant (jac 0.95 vs UID160) → v5 INVERTED + rank_cap top3.
# Polarity test — anti-correlated to UID 160 (pearson -0.843 offline sim).
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
export POKER44_V1_VARIANT="v5_statistical_invert_top3"
export POKER44_MODEL_NAME="poker44-v5-invert-top3-uid226"
export POKER44_MODEL_VERSION="5.1"
export POKER44_MODEL_REPO_URL="https://github.com/Krzysiek99999/aceguard-engine"
# Compliance debt: commit 887d8726 nie zawiera rank_cap_remap.py (push pending).
export POKER44_MODEL_REPO_COMMIT="887d8726ab6b3cc4806e0bb433c9d2519b3e4653"
export POKER44_MODEL_OPEN_SOURCE="1"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="v5_statistical detector with polarity INVERTED (final = 1 - raw) + deterministic rank_cap_remap top-3 binarization. Tests benchmark-vs-live polarity hypothesis from POLARITY_AUDIT_OUR_SCORERS.md. Anti-correlated to UID 160 (pearson -0.843 offline sim). No supervised training."
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No external private user data; only validator-distributed evaluation chunks (40 hands each)."
# v5_statistical_invert_top3: rank_cap_remap top_n=3 in dispatch (miner_v1.py), brak adaptive_n.

exec .venv/bin/python neurons/miner_v1.py \
    --netuid 126 \
    --wallet.name poker44_p22020 \
    --wallet.hotkey poker44_p22020_hot \
    --subtensor.network finney \
    --axon.port 22020 \
    --axon.external_ip 80.238.120.82 \
    --axon.external_port 22020 \
    "$LOG_FLAG" \
    --blacklist.allowed_validator_hotkeys "${VALIDATOR_ALLOWLIST[@]}"
