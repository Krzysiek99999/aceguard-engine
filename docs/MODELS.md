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

Daily refreshed supervised schema model trained on public Poker44 benchmark
releases using miner-visible payload views only.

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

### `v115_short`

Short-unit supervised schema + sequence model trained on public miner-visible
benchmark chunks. This family is a canary for the high-offline short-unit
signal and is intentionally deployed separately from the live-sized `v118`
family.

Files:

- `poker44/score/v112_super_inference.py`
- `poker44/score/robust_schema/features.py`
- `poker44/score/statistical_v25.py`
- `poker44/score/sequence_schema.py`
- `data/models/v115_short/model.pkl`

Supported runtime strategies:

- `avg`
- `rank_mean`
- `stack`
- base model heads included in the artifact

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

### `v136_live`

Current live-sized v1.13 supervised schema model trained on public Poker44
benchmark releases through source date 2026-07-07. The training contract merges
same-date, same-label public benchmark chunks into live-sized units and validates
ranking quality across source dates. No validator-private labels, wallets,
host identifiers, or deployment logs are used.

Files:

- `poker44/score/v112_super_inference.py`
- `poker44/score/robust_schema/features.py`
- `poker44/score/sequence_schema.py`
- `poker44/score/statistical_v25.py`
- `data/models/v136_livesized_20260707/model.pkl`
- `data/models/v136_livesized_20260707/report.json`

Supported runtime strategies:

- `cat`
- `xgb`
- `lgb`
- `et`
- `avg`
- `rank_mean`
- `stack`

The deployment layer applies rank-cap calibration after scoring. The canary
variant uses `cat` with `top2` on one Cherry slot.

### `v140_multi`

Multi-seed live-sized v1.13 behavioural n-gram ranker trained on public Poker44
benchmark releases through source date 2026-07-07. The training contract merges
same-date, same-label public benchmark chunks into live-sized units across
multiple random merge seeds, then validates the ranker across held-out source
dates and unseen merge seeds. No validator-private labels, wallets, hotkeys,
host identifiers, or deployment logs are used.

Files:

- `poker44/score/v112_super_inference.py`
- `poker44/score/ngram_ranker.py`
- `poker44/score/robust_schema/features.py`
- `poker44/score/sequence_schema.py`
- `poker44/score/features_pot_geometry.py`
- `poker44/score/features_v13_safe.py`
- `poker44/score/extended_features.py`
- `poker44/score/enterprise_features.py`
- `data/models/v140_multiseed_livesized/model.pkl`
- `data/models/v140_multiseed_livesized/report.json`

Supported runtime strategies:

- `rank_mean`
- `cat`
- `xgb`
- `lgb`
- `et`
- `avg`
- `stack`

The deployment layer applies rank-cap calibration after scoring. The canary
variant uses `rank_mean` with `top2` on one Cherry slot.

### `v142_rankblend`

Rank-space blend of two independently gated live-sized public-benchmark rankers:
`v140_multi` with `rank_mean` and a stricter drift-filtered v141 ET child. The
bundle is self-contained and embeds both child bundles; it does not reference
wallets, host identifiers, validator-private labels, or external model paths.

Files:

- `poker44/score/v112_super_inference.py`
- `poker44/score/ngram_ranker.py`
- `poker44/score/robust_schema/features.py`
- `poker44/score/sequence_schema.py`
- `poker44/score/features_pot_geometry.py`
- `poker44/score/features_v13_safe.py`
- `poker44/score/extended_features.py`
- `poker44/score/enterprise_features.py`
- `data/models/v142_rankblend/model.pkl`
- `data/models/v142_rankblend/report.json`

Supported runtime strategy:

- `rank_mean`

The deployment layer applies rank-cap calibration after scoring. This family is
prepared as a candidate and should only be deployed after a live reflection gate.

### `v334_v11_consensus_lock7_dual25_top8`

Original single-replacement hybrid for 100-chunk evaluation batches. Five
independently implemented behavioural scorers protect seven members selected
from the v11 top eight. A five-network permutation-invariant hand-set rank
ensemble orders the protected members and the rest of the batch, filling one
threshold position. The deployment head emits exactly eight scores at or above
`0.5` while preserving the complete continuous ranking.

Three hand-set components were trained only on miner-visible public benchmark
releases through source date 2026-06-30. Two refreshed components were trained
through source date 2026-07-07. Their disclosed rank weights are
`0.30/0.225/0.225/0.125/0.125`. The 75/25 generation blend and lock7/top8
operating point were selected post-hoc on July 8-11 robustness checks and
unlabeled public-model topology, so the model requires live canary validation
before any expansion.

Files:

- `poker44/score/v334_v11_consensus_lock_inference.py`
- `poker44/score/v323_v11_consensus_lock_inference.py`
- `poker44/score/ensemble_v11.py`
- `poker44/score/original_behavior_features.py`
- `poker44/score/original_set_model.py`
- `poker44/score/original_set_inference.py`
- `poker44/score/original_set_ensemble_inference.py`
- `data/models/v334_v11_consensus_lock7_dual25_top8/model.pkl`
- `data/models/v334_v11_consensus_lock7_dual25_top8/report.json`

No competitor implementation or weights, validator-private labels, identities,
cards, outcomes, wallets, hotkeys, IP addresses, or private data were used.

### `v373_original_hash_bag_top8`

Original canonical hand-bag ensemble trained on natural public Poker44
`split=train` SourceUnits through source date 2026-07-11. It does not merge
players, chunks, or source units and does not reapply the validator sanitizer.
The model combines LightGBM ranking, ExtraTrees motif/redundancy, and
HistGradientBoosting surfaces using within-batch percentile ranks.

The exact model bytes and top-eight head were frozen before the one-shot
2026-07-12 release. On 128 untouched natural units it achieved AP `0.986741`,
recall at 5% FPR `0.984375`, and ranking component `0.640672`, exceeding the
best reproducible frozen public baseline by `0.201102`. The result and all
rolling-fold diagnostics are recorded in the public report.

Files:

- `poker44/score/model_view_hand_hash.py`
- `poker44/score/original_schema_contract_features.py`
- `poker44/score/sequence_schema.py`
- `poker44/score/temporal_consistency_features.py`
- `poker44/score/original_tree_surface_features.py`
- `poker44/score/original_redundancy_features.py`
- `poker44/score/original_hash_bag_features.py`
- `poker44/score/original_hash_bag_inference.py`
- `scripts/miner_training/build_v362_original_tree_surface_oof.py`
- `data/models/v373_original_hash_bag_top8/model.pkl`
- `data/models/v373_original_hash_bag_top8/report.json`
- `data/models/v373_original_hash_bag_top8/oof_component_predictions.npz`

No competitor code or weights, validator-private labels, identities, cards,
outcomes, wallets, hotkeys, IP addresses, or private live data were used.
