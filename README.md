# AceGuard Poker44 Public Model Release

This directory is a clean release staging area for Poker44 miner model transparency.

It contains the model code and artifacts needed for the active Mac and Cherry model families, without wallets, hotkeys, IP addresses, raw validator data, logs, private notes, or deployment history.

## Included model families

- Mac control family: deterministic statistical scoring (`v5_statistical`).
- Mac secondary family: deterministic type-aware statistical scoring (`v10_mild`).
- Cherry canary family: supervised schema model (`v112_super`) with neutral public feature names.
- Daily challenger family: refreshed supervised schema model (`v113_daily`) trained on current v1.12 benchmark releases.
- Live-sized challenger family: supervised schema and sequence model (`v118_live`) trained on public miner-visible benchmark chunks merged to live-sized requests.

## Not included

- Wallet files, seed phrases, hotkey names, coldkey names.
- Host IPs, PM2 process names, SSH aliases, or production run scripts.
- Raw benchmark cache, raw validator chunks, forward-audit logs, dashboards, or private observations.
- Old experimental variants and research-only files.

## Publication rule

Only publish this clean release directory or an equivalent clean repo. Do not publish the root research workspace.
