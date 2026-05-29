#!/usr/bin/env bash
# UID 97 (port 22019) — poker44-v5-adaptive-uid97
# Deployed 2026-05-23 jako część 5-UID portfolio (codex iter 8 diversification).
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
export POKER44_V1_VARIANT="v5_adaptive_n"
export POKER44_MAX_N="4"
export POKER44_MODEL_NAME="poker44-v5-adaptive-uid97"
export POKER44_MODEL_VERSION="5.0"
export POKER44_MODEL_REPO_URL="https://github.com/Krzysiek99999/aceguard-engine"
export POKER44_MODEL_REPO_COMMIT="887d8726ab6b3cc4806e0bb433c9d2519b3e4653"
export POKER44_MODEL_OPEN_SOURCE="1"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="Statistical detector v5 — behavioral anomaly score (sizing entropy, sequence repetition, quantization). No supervised training."
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No external private user data; only validator-distributed evaluation chunks (40 hands each)."

# === Adaptive N (codex strategy 2026-05-25) ===
export POKER44_ADAPTIVE_N="1"
export POKER44_ADAPTIVE_PROFILE="balanced"
export POKER44_ADAPTIVE_MIN_N="1"
export POKER44_ADAPTIVE_MAX_N="4"
unset POKER44_MAX_N

exec .venv/bin/python neurons/miner_v1.py \
    --netuid 126 \
    --wallet.name poker44_p22019 \
    --wallet.hotkey poker44_p22019_hot \
    --subtensor.network finney \
    --axon.port 22019 \
    --axon.external_ip 80.238.120.82 \
    --axon.external_port 22019 \
    "$LOG_FLAG" \
    --blacklist.allowed_validator_hotkeys "${VALIDATOR_ALLOWLIST[@]}"
