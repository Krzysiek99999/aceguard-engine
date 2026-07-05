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
- `linear`
- `xgb`
- `avg`
- `rank_mean`
- `stack`

The deployment layer decides top1/top2/top3 through rank-cap calibration.

### `v113_daily`

Daily refreshed supervised schema model trained on current public Poker44 v1.12
benchmark releases using miner-visible payload views only.

Files:

- `poker44/score/v112_super_inference.py`
- `poker44/score/robust_schema/features.py`
- `poker44/score/statistical_v25.py`
- `data/models/v113_daily/model.pkl`

Supported runtime strategies are the same schema scorer heads:

- `cat`
- `et`
- `lgb`
- `linear`
- `xgb`
- `avg`
- `rank_mean`
- `stack`

### `v118_live`

Live-sized supervised schema and sequence model trained on public miner-visible
benchmark chunks merged into request-sized units. The current public bundle uses
the `super_seq` feature set, `abs_batch` feature mode, and date-holdout
validation under the current rank-first reward.

Files:

- `poker44/score/v112_super_inference.py`
- `poker44/score/robust_schema/features.py`
- `poker44/score/sequence_schema.py`
- `poker44/score/statistical_v25.py`
- `data/models/v118_livesized_chunks/model.pkl`

Supported runtime strategies:

- `et`
- `linear`
- `rf`
- `avg`
- `rank_mean`
- `stack`

The deployment layer applies rank-cap calibration after scoring. The rank order
is the primary signal for the current Poker44 reward.

### `v118_stable75`

Live-sized supervised schema and sequence model trained like `v118_live`, but
with a stricter benchmark-to-live stability filter. Feature channels with
high distribution drift against the latest unlabeled live audit payload are
removed before training and serving.

Files:

- `poker44/score/v112_super_inference.py`
- `poker44/score/robust_schema/features.py`
- `poker44/score/sequence_schema.py`
- `poker44/score/statistical_v25.py`
- `data/models/v118_stableall75/model.pkl`

Supported runtime strategies:

- `et`

The deployment layer applies rank-cap calibration after scoring.
