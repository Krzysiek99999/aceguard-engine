# Segmented Inference Variant

`v113_daily_seg35_top2mean_avg` uses the same public v113 daily ensemble heads
as `v113_daily_avg`, but changes the serve-time unit contract.

The public benchmark units used for training are usually 30-40 hands per chunk.
Recent live competition requests can contain larger 80-100 hand chunks. Scoring
those larger chunks directly creates a train/serve mismatch for count-sensitive
features.

This variant splits each incoming chunk into 35-hand segments, scores all
segments in the request with the inner `avg` strategy, then assigns the original
chunk the mean of its two highest segment scores. The goal is to preserve the
training unit size while still returning one score per validator chunk.

No validator-private labels or private external data are used.
