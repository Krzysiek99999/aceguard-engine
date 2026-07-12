"""Source-aware loading for the public Poker44 training benchmark.

Training code may use provenance to define honest splits and ranking groups,
but provenance must never enter the model feature matrix.  This module keeps
those two concerns separate.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poker44.score.model_view_hand_hash import (
    model_view_hand_hash,
    model_view_hand_payload,
)

@dataclass(frozen=True)
class SourceUnit:
    source_date: str
    release_version: str
    schema_version: str
    split: str
    group_id: str
    group_hash: str
    group_index: int
    item_index: int
    chunk: list[dict[str, Any]]
    label: int

    @property
    def source_key(self) -> str:
        return f"{self.source_date}|{self.group_id}|{self.item_index}"

    @property
    def ranking_group(self) -> str:
        return f"{self.source_date}|{self.group_id}"


def _label(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"bot", "1", "true"}:
            return 1
        if normalized in {"human", "0", "false"}:
            return 0
        raise ValueError(f"unsupported benchmark label: {value!r}")
    parsed = int(value)
    if parsed not in {0, 1}:
        raise ValueError(f"unsupported benchmark label: {value!r}")
    return parsed


def _root(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def canonical_model_hand_payload(hand: dict[str, Any]) -> dict[str, Any]:
    """Project one API hand onto fields available to deployable model features."""
    return model_view_hand_payload(hand)


def canonical_model_hand_hash(hand: dict[str, Any]) -> str:
    return model_view_hand_hash(hand)


def hand_overlap_audit(units: list[SourceUnit]) -> dict[str, int]:
    """Summarize exact model-view hand reuse across labels and partitions."""
    owners: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "labels": set(), "splits": set(), "dates": set()}
    )
    for unit in units:
        for hand in unit.chunk:
            digest = canonical_model_hand_hash(hand)
            row = owners[digest]
            row["count"] += 1
            row["labels"].add(int(unit.label))
            row["splits"].add(str(unit.split))
            row["dates"].add(str(unit.source_date))
    return {
        "hands": int(sum(row["count"] for row in owners.values())),
        "unique_model_view_hands": int(len(owners)),
        "duplicate_hashes": int(sum(row["count"] > 1 for row in owners.values())),
        "cross_label_hashes": int(sum(len(row["labels"]) > 1 for row in owners.values())),
        "cross_split_hashes": int(sum(len(row["splits"]) > 1 for row in owners.values())),
        "cross_date_hashes": int(sum(len(row["dates"]) > 1 for row in owners.values())),
        "max_multiplicity": int(max((row["count"] for row in owners.values()), default=0)),
    }


def load_source_units(
    benchmark_dir: Path,
    *,
    release_version: str = "",
    first_date: str = "",
    last_date: str = "",
    splits: set[str] | None = None,
    sanitize: bool = False,
) -> list[SourceUnit]:
    """Load canonical miner-visible API chunks with split/group provenance.

    The public benchmark contract already exposes the exact miner-visible
    model input. Reapplying ``prepare_hand_for_miner`` is non-idempotent and
    creates a train/serve mismatch, so the legacy flag is retained only as a
    fail-fast guard for stale callers.
    """

    if sanitize:
        raise ValueError(
            "benchmark API chunks are already miner-visible; refusing a second sanitization pass"
        )

    units: list[SourceUnit] = []
    seen: set[str] = set()
    for path in sorted(Path(benchmark_dir).glob("chunks_*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"benchmark payload is not an object: {path}")
        root = _root(payload)
        source_date = str(root.get("sourceDate") or path.stem.removeprefix("chunks_"))
        if first_date and source_date < first_date:
            continue
        if last_date and source_date > last_date:
            continue
        root_release = str(root.get("releaseVersion") or "")
        schema_version = str(root.get("schemaVersion") or "")
        groups = root.get("chunks") or []
        if not isinstance(groups, list):
            raise ValueError(f"benchmark chunks is not a list: {path}")
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict):
                raise ValueError(f"benchmark group is not an object: {path}:{group_index}")
            chunks = group.get("chunks") or []
            labels = group.get("groundTruth")
            if labels is None:
                labels = group.get("groundTruthLabels") or []
            if len(chunks) != len(labels):
                raise ValueError(
                    f"benchmark group has {len(chunks)} chunks and {len(labels)} labels: "
                    f"{path}:{group_index}"
                )
            split = str(group.get("split") or root.get("split") or "train").strip().lower()
            if splits is not None and split not in splits:
                continue
            group_id = str(group.get("chunkId") or f"{path.stem}:{group_index}")
            group_hash = str(group.get("chunkHash") or "")
            group_release = str(group.get("releaseVersion") or root_release)
            if release_version and group_release != release_version:
                continue

            for item_index, (raw_chunk, raw_label) in enumerate(zip(chunks, labels, strict=True)):
                if not isinstance(raw_chunk, list):
                    raise ValueError(f"benchmark chunk is not a list: {path}:{group_index}:{item_index}")
                chunk = [hand for hand in raw_chunk if isinstance(hand, dict)]
                if not chunk:
                    continue
                unit = SourceUnit(
                    source_date=source_date,
                    release_version=group_release,
                    schema_version=schema_version,
                    split=split,
                    group_id=group_id,
                    group_hash=group_hash,
                    group_index=int(group_index),
                    item_index=int(item_index),
                    chunk=chunk,
                    label=_label(raw_label),
                )
                if unit.source_key in seen:
                    raise ValueError(f"duplicate benchmark source unit: {unit.source_key}")
                seen.add(unit.source_key)
                units.append(unit)
    return units


def source_summary(units: list[SourceUnit]) -> dict[str, Any]:
    split_counts = Counter(unit.split for unit in units)
    return {
        "units": len(units),
        "bots": int(sum(unit.label for unit in units)),
        "humans": int(sum(1 - unit.label for unit in units)),
        "dates": sorted({unit.source_date for unit in units}),
        "groups": len({unit.ranking_group for unit in units}),
        "split_counts": dict(sorted(split_counts.items())),
        "release_versions": sorted({unit.release_version for unit in units}),
    }


def grouped_units(units: list[SourceUnit]) -> list[list[SourceUnit]]:
    by_group: dict[str, list[SourceUnit]] = {}
    for unit in units:
        by_group.setdefault(unit.ranking_group, []).append(unit)
    return [
        sorted(by_group[key], key=lambda unit: unit.item_index)
        for key in sorted(by_group)
    ]
