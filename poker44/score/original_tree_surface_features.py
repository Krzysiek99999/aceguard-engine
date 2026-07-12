"""Independent wide, motif and temporal feature surface for Poker44 trees."""
from __future__ import annotations

from typing import Any

from poker44.score.original_schema_contract_features import chunk_features as contract_features
from poker44.score.sequence_schema import chunk_features as motif_features
from poker44.score.temporal_consistency_features import chunk_features as temporal_features


_SIZE_SHORTCUTS = {
    "contract__hand_count",
    "contract__chunk__hands",
    "contract__chunk__actions",
    "motif__seq_hand_count",
    "motif__seq_action_count",
    "motif__seq_actions_per_hand_max",
    "temporal__tc_hand_count",
}


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Return identity-free features without raw source-size shortcuts."""
    out: dict[str, float] = {}
    for prefix, extractor in (
        ("contract", contract_features),
        ("motif", motif_features),
        ("temporal", temporal_features),
    ):
        for key, value in extractor(chunk).items():
            name = f"{prefix}__{key}"
            if name in _SIZE_SHORTCUTS:
                continue
            out[name] = float(value)
    return out
