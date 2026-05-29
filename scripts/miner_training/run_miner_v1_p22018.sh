#!/usr/bin/env bash
# UID 4 (port 22018) — poker44-v10-mild-uid4
# Deployed 2026-05-23 — codex iter 27 expanded portfolio to 8 UIDs.
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
export POKER44_V1_VARIANT="v10_mild"
export POKER44_MAX_N="2"
export POKER44_MODEL_NAME="poker44-v10-mild-uid4"
export POKER44_MODEL_VERSION="11.0"
export POKER44_MODEL_REPO_URL="https://github.com/Krzysiek99999/aceguard-engine"
export POKER44_MODEL_REPO_COMMIT="887d8726ab6b3cc4806e0bb433c9d2519b3e4653"
export POKER44_MODEL_OPEN_SOURCE="1"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="Statistical detector — behavioral anomaly score. No supervised training. Codex-validated orthogonal arm."
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No external private user data; only validator-distributed evaluation chunks."

# === Adaptive N (codex strategy 2026-05-25) ===
export POKER44_ADAPTIVE_N="1"
export POKER44_ADAPTIVE_PROFILE="conservative"
export POKER44_ADAPTIVE_MIN_N="1"
export POKER44_ADAPTIVE_MAX_N="3"
unset POKER44_MAX_N

exec .venv/bin/python neurons/miner_v1.py \
    --netuid 126 \
    --wallet.name poker44_cold \
    --wallet.hotkey poker44_hot \
    --subtensor.network finney \
    --axon.port 22018 \
    --axon.external_ip 80.238.120.82 \
    --axon.external_port 22018 \
    "$LOG_FLAG" \
    --blacklist.allowed_validator_hotkeys "${VALIDATOR_ALLOWLIST[@]}"
