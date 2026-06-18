# Model Inventory

## Mac models

### `v5_statistical`

Deterministic behavioral scorer. It uses chunk-level sizing, sequence, and quantization signals and then applies rank-based calibration in the miner.

Files:

- `poker44/score/statistical_v5.py`
- `poker44/score/calibration.py`
- `poker44/score/rank_cap_remap.py`

### `v10_mild`

Type-aware deterministic scorer using the `v9` family plus stage-2 mild calibration.

Files:

- `poker44/score/statistical_v9.py`
- `poker44/score/statistical_v6.py`
- `poker44/score/sequence_v8.py`
- `poker44/score/features_pot_geometry.py`
- `poker44/score/features_response_curves.py`
- `poker44/score/stage2_calibration.py`
- `poker44/score/calibration.py`

## Cherry models

### `v112_super`

Supervised schema model trained on miner-visible benchmark views. Public bundle uses neutral `schema__` feature keys.

Files:

- `poker44/score/v112_super_inference.py`
- `poker44/score/robust_schema/features.py`
- `poker44/score/statistical_v25.py`
- `data/models/v112_super/model.pkl`

Supported runtime strategies:

- `cat`
- `et`
- `lgb`
- `xgb`
- `rank_mean`
- `stack`

The deployment layer decides top1/top2/top3 through rank-cap calibration.
