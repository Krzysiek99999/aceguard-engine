"""Extended chunk features — FFT spectral + graph topology + temporal patterns.

Designed to add ORTHOGONAL signal to existing Travis schema + tomkaba ml outputs.

Features:
  FFT (12): top-K magnitudes + spectral entropy + dominant frequency on bet_size sequence
  GRAPH (10): player interaction graph stats (degree, clustering, betweenness, etc.)
  TEMPORAL (8): action timing rhythm + inter-event variance + autocorrelation
"""
from __future__ import annotations
from typing import Dict, List
import math
import numpy as np


# ============================================================
# FFT features
# ============================================================

def fft_features(values: List[float], n_top: int = 4) -> Dict[str, float]:
    """Extract spectral features from a 1D sequence.

    Returns dict with: fft_top{1..n}_mag, fft_dominant_freq, fft_spectral_entropy,
                       fft_spectral_centroid, fft_spectral_rolloff
    """
    out = {}
    n = len(values)
    if n < 4:
        for i in range(1, n_top + 1):
            out[f'fft_top{i}_mag'] = 0.0
        out['fft_dominant_freq'] = 0.0
        out['fft_spectral_entropy'] = 0.0
        out['fft_spectral_centroid'] = 0.0
        out['fft_spectral_rolloff'] = 0.0
        out['fft_total_energy'] = 0.0
        return out

    arr = np.asarray(values, dtype=np.float64)
    arr = arr - arr.mean()  # zero-mean
    if arr.std() < 1e-6:
        # Constant signal — no spectral info
        for i in range(1, n_top + 1):
            out[f'fft_top{i}_mag'] = 0.0
        out['fft_dominant_freq'] = 0.0
        out['fft_spectral_entropy'] = 0.0
        out['fft_spectral_centroid'] = 0.0
        out['fft_spectral_rolloff'] = 0.0
        out['fft_total_energy'] = 0.0
        return out

    fft = np.fft.rfft(arr)
    mag = np.abs(fft)
    mag_norm = mag / (mag.sum() + 1e-9)

    # Top-K magnitudes (sorted desc, padded with 0)
    sorted_mag = np.sort(mag)[::-1]
    for i in range(n_top):
        out[f'fft_top{i+1}_mag'] = float(sorted_mag[i]) if i < len(sorted_mag) else 0.0

    # Dominant frequency (index of max, normalized to [0,1])
    out['fft_dominant_freq'] = float(np.argmax(mag) / max(len(mag), 1))

    # Spectral entropy
    out['fft_spectral_entropy'] = float(-np.sum(mag_norm * np.log(mag_norm + 1e-9)) / np.log(max(len(mag_norm), 2)))

    # Spectral centroid (frequency-weighted mean)
    freqs = np.arange(len(mag))
    out['fft_spectral_centroid'] = float((freqs * mag).sum() / (mag.sum() + 1e-9))

    # Spectral rolloff (frequency below which 85% of energy is contained)
    cum = np.cumsum(mag) / (mag.sum() + 1e-9)
    rolloff_idx = np.argmax(cum >= 0.85)
    out['fft_spectral_rolloff'] = float(rolloff_idx / max(len(mag), 1))

    # Total energy (sum of magnitudes squared)
    out['fft_total_energy'] = float(np.sum(mag ** 2))

    return out


# ============================================================
# Graph features (player interaction)
# ============================================================

def graph_features(hands: List[dict]) -> Dict[str, float]:
    """Build directed graph: actor_seat → next_actor_seat. Compute topology stats."""
    out = {
        'graph_n_nodes': 0.0, 'graph_n_edges': 0.0, 'graph_density': 0.0,
        'graph_max_indegree': 0.0, 'graph_max_outdegree': 0.0, 'graph_self_loop_ratio': 0.0,
        'graph_avg_degree': 0.0, 'graph_degree_variance': 0.0,
        'graph_unique_seq_pair_count': 0.0, 'graph_top_edge_share': 0.0,
    }
    if not hands:
        return out

    # Collect all consecutive (actor_t, actor_t+1) edges
    edges = []
    seats = set()
    for hand in hands:
        if not isinstance(hand, dict):
            continue
        actions = hand.get('actions', [])
        for i in range(len(actions) - 1):
            a1 = actions[i].get('actor_seat', 0)
            a2 = actions[i+1].get('actor_seat', 0)
            edges.append((a1, a2))
            seats.add(a1); seats.add(a2)

    if not edges:
        return out

    n_nodes = len(seats)
    n_edges = len(edges)

    # Edge counts
    from collections import Counter
    edge_counts = Counter(edges)
    unique_edges = len(edge_counts)

    # In/out degree
    indegree = Counter()
    outdegree = Counter()
    for a1, a2 in edges:
        outdegree[a1] += 1
        indegree[a2] += 1

    max_in = max(indegree.values()) if indegree else 0
    max_out = max(outdegree.values()) if outdegree else 0
    self_loops = sum(1 for e in edges if e[0] == e[1])

    degrees = []
    for s in seats:
        degrees.append(indegree.get(s, 0) + outdegree.get(s, 0))

    out['graph_n_nodes'] = float(n_nodes)
    out['graph_n_edges'] = float(n_edges)
    out['graph_density'] = float(unique_edges / max(n_nodes * n_nodes, 1))
    out['graph_max_indegree'] = float(max_in / max(n_edges, 1))
    out['graph_max_outdegree'] = float(max_out / max(n_edges, 1))
    out['graph_self_loop_ratio'] = float(self_loops / max(n_edges, 1))
    out['graph_avg_degree'] = float(np.mean(degrees)) if degrees else 0.0
    out['graph_degree_variance'] = float(np.var(degrees)) if len(degrees) > 1 else 0.0
    out['graph_unique_seq_pair_count'] = float(unique_edges)
    # Top edge share — how concentrated is the most frequent action pair
    top_edge_n = edge_counts.most_common(1)[0][1] if edge_counts else 0
    out['graph_top_edge_share'] = float(top_edge_n / max(n_edges, 1))
    return out


# ============================================================
# Temporal/timing features
# ============================================================

def temporal_features(hands: List[dict]) -> Dict[str, float]:
    """Capture per-hand action-count rhythms + autocorrelation."""
    out = {
        'temporal_actions_per_hand_cv': 0.0,
        'temporal_actions_per_hand_lag1_autocorr': 0.0,
        'temporal_bet_size_lag1_autocorr': 0.0,
        'temporal_pot_growth_cv': 0.0,
        'temporal_hand_length_runs': 0.0,
        'temporal_n_long_runs_ge3': 0.0,
        'temporal_amount_jump_max': 0.0,
        'temporal_decision_density': 0.0,
    }
    if not hands:
        return out

    counts = [len(h.get('actions', [])) for h in hands if isinstance(h, dict)]
    if len(counts) < 2:
        return out
    arr = np.asarray(counts, dtype=np.float64)
    mean = arr.mean(); std = arr.std()

    out['temporal_actions_per_hand_cv'] = float(std / (mean + 1e-6))

    # Lag-1 autocorrelation
    if std > 1e-6 and len(arr) >= 2:
        out['temporal_actions_per_hand_lag1_autocorr'] = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])

    # Bet size sequence — pool all bet/raise amounts globally
    bet_sizes = []
    for h in hands:
        for a in h.get('actions', []):
            if a.get('action_type') in ('bet', 'raise', 'all_in'):
                bet_sizes.append(float(a.get('normalized_amount_bb', 0)))
    if len(bet_sizes) >= 2:
        b = np.asarray(bet_sizes)
        if b.std() > 1e-6:
            out['temporal_bet_size_lag1_autocorr'] = float(np.corrcoef(b[:-1], b[1:])[0, 1])
        out['temporal_amount_jump_max'] = float(np.max(np.abs(np.diff(b))) / max(b.mean(), 1e-6))

    # Pot growth per hand
    pot_growths = []
    for h in hands:
        actions = h.get('actions', [])
        if actions:
            pot_start = float(actions[0].get('pot_before_bb', 0))
            pot_end = float(actions[-1].get('pot_after_bb', 0))
            pot_growths.append(pot_end - pot_start)
    if len(pot_growths) >= 2:
        pg = np.asarray(pot_growths)
        m = pg.mean(); s = pg.std()
        if abs(m) > 1e-6:
            out['temporal_pot_growth_cv'] = float(s / (abs(m) + 1e-6))

    # Action count runs (consecutive same-count hands)
    runs = []
    if len(counts) >= 2:
        run = 1
        for i in range(1, len(counts)):
            if counts[i] == counts[i-1]:
                run += 1
            else:
                runs.append(run); run = 1
        runs.append(run)
        out['temporal_hand_length_runs'] = float(len(runs) / len(counts))
        out['temporal_n_long_runs_ge3'] = float(sum(1 for r in runs if r >= 3))

    # Decision density — total actions / total hands
    total_actions = sum(counts)
    out['temporal_decision_density'] = float(total_actions / max(len(counts), 1))

    return out


# ============================================================
# Combined extractor
# ============================================================

def compute_extended_features(chunk: List[dict]) -> Dict[str, float]:
    """All extended features for a single chunk."""
    hands = [h for h in chunk if isinstance(h, dict)]
    out = {}

    # FFT on bet sizes
    bet_sizes = []
    for h in hands:
        for a in h.get('actions', []):
            if a.get('action_type') in ('bet', 'raise', 'all_in', 'call'):
                bet_sizes.append(float(a.get('normalized_amount_bb', 0)))
    out.update({f'bet_{k}': v for k, v in fft_features(bet_sizes).items()})

    # FFT on action count per hand
    counts = [len(h.get('actions', [])) for h in hands]
    out.update({f'count_{k}': v for k, v in fft_features([float(c) for c in counts], n_top=3).items()})

    # FFT on pot growth per hand
    pot_g = []
    for h in hands:
        ax = h.get('actions', [])
        if ax:
            pot_g.append(float(ax[-1].get('pot_after_bb', 0)) - float(ax[0].get('pot_before_bb', 0)))
    out.update({f'pot_{k}': v for k, v in fft_features(pot_g, n_top=3).items()})

    # Graph topology
    out.update(graph_features(hands))

    # Temporal patterns
    out.update(temporal_features(hands))

    return out


if __name__ == "__main__":
    # Smoke test
    import json
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[2]
    for f in sorted((REPO_ROOT / "data" / "benchmark_real").glob("chunks_*.json")):
        d = json.load(open(f))
        chunk = d['data']['chunks'][0]['chunks'][0]
        feats = compute_extended_features(chunk)
        print(f"Extracted {len(feats)} features:")
        for k, v in list(feats.items())[:15]:
            print(f"  {k}: {v:.4f}")
        print(f"  ... ({len(feats)} total)")
        break
