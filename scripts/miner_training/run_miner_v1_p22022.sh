#!/usr/bin/env bash
# UID 192 (port 22022) — v1_real_2026 LightGBM, cap 0.52 (HIGH bracket)
# Per codex strategy 2026-05-09: 3× v1_real_2026 z różnymi cap brackety.
# UID 192 = HIGH cap (0.52) — testuje strategy "high reward + ML penalty acceptance".
#
# Replaces previous F0.7 PROBE (neurons/miner.py reference clone) which had 0× top10
# in 2 dni testów (2026-05-07 → 2026-05-09). UID 160 (CONTROL) wystarczy jako reference baseline.
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
# 2026-05-21 22:15: UID 192 = v5_statistical CONSERVATIVE arm (max_n=2)
# vs UID 160 = v5_statistical AGGRESSIVE arm (max_n=3, won previous cycle).
# Codex iter 8 portfolio: not 2 clones — different calibrator regimes to break correlation.
# Conservative = 5% bot rate, lower FPR risk; if live bot_rate is ~5%, this may beat aggressive.
export POKER44_V1_VARIANT="v8_markov"
export POKER44_MAX_N="2"
export POKER44_MODEL_NAME="poker44-v8-sequence-uid192"
export POKER44_MODEL_VERSION="5.1"
export POKER44_MODEL_REPO_URL="https://github.com/Krzysiek99999/aceguard-engine"
export POKER44_MODEL_REPO_COMMIT="887d8726ab6b3cc4806e0bb433c9d2519b3e4653"
export POKER44_MODEL_OPEN_SOURCE="1"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="Statistical detector v5 — chunks ranked by behavioral anomaly score (sizing entropy, sequence repetition, quantization). No supervised training. Conservative arm: max_n=2 positives per 40-window."
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No external private user data; only validator-distributed evaluation chunks (40 hands each) collected from our own miner's dendrite calls."
export POKER44_DISABLE_LIVE_LOG=1

# === Adaptive N (codex strategy 2026-05-25) ===
export POKER44_ADAPTIVE_N="1"
export POKER44_ADAPTIVE_PROFILE="balanced"
export POKER44_ADAPTIVE_MIN_N="1"
export POKER44_ADAPTIVE_MAX_N="4"
unset POKER44_MAX_N

exec .venv/bin/python neurons/miner_v1.py \
    --netuid 126 \
    --wallet.name poker44_p22022 \
    --wallet.hotkey poker44_p22022_hot \
    --subtensor.network finney \
    --axon.port 22022 \
    --axon.external_ip 80.238.120.82 \
    --axon.external_port 22022 \
    "$LOG_FLAG" \
    --blacklist.allowed_validator_hotkeys "${VALIDATOR_ALLOWLIST[@]}"
