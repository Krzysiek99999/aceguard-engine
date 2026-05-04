# AceGuard Engine — Poker44 SN126 Bot Detection Miner

Bot detection miner for [Poker44 subnet 126](https://poker44.net) on Bittensor.

## Architecture

LightGBM-based bot detection with adaptive cap calibration:

- **Variant**: `v1_b_deeper_adaptive`
- **Detection model**: LightGBM `B_deeper` trained on 5000 labeled chunks (50% bot / 50% human)
- **Calibration**: `adaptive_safe_calibrate` — Otsu's method for natural threshold + hard safety cap
- **Per-batch operation**: 40 chunks per cycle, sliding window=20

## Reward formula (subnet contract)

```python
reward = (0.65 * AP + 0.35 * recall) * (1 - FPR)^2
if FPR >= 0.10: reward = 0  # human safety cliff
```

## Files

- `neurons/miner_v1.py` — main miner with 26 variants
- `neurons/miner.py` — Poker44 reference heuristic miner
- `poker44/score/calibration.py` — adaptive_safe_calibrate
- `poker44/score/features.py` — feature extraction (179 features)
- `poker44/models/` — neural / statistical detectors
- `poker44/utils/model_manifest.py` — manifest builder

## Run

```bash
export POKER44_V1_VARIANT=v1_b_deeper_adaptive
export POKER44_MAX_BOT_FRACTION=0.45
python neurons/miner_v1.py --netuid 126 --wallet.name <cold> --wallet.hotkey <hot> \
    --subtensor.network finney --axon.port <port>
```

## License

MIT
