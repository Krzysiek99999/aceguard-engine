"""Canonical hashing for fields visible to a deployed Poker44 miner."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def model_view_hand_payload(hand: dict[str, Any]) -> dict[str, Any]:
    metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
    players = hand.get("players") if isinstance(hand.get("players"), list) else []
    streets = hand.get("streets") if isinstance(hand.get("streets"), list) else []
    actions = hand.get("actions") if isinstance(hand.get("actions"), list) else []
    return {
        "metadata": {
            key: metadata.get(key)
            for key in (
                "game_type",
                "limit_type",
                "max_seats",
                "hero_seat",
                "button_seat",
                "hand_ended_on_street",
                "sb",
                "bb",
                "ante",
            )
        },
        "players": [
            {"seat": row.get("seat"), "starting_stack": row.get("starting_stack")}
            for row in players
            if isinstance(row, dict)
        ],
        "streets": [
            {"street": row.get("street")}
            for row in streets
            if isinstance(row, dict)
        ],
        "actions": [
            {
                key: row.get(key)
                for key in (
                    "street",
                    "actor_seat",
                    "action_type",
                    "amount",
                    "raise_to",
                    "call_to",
                    "normalized_amount_bb",
                    "pot_before",
                    "pot_after",
                )
            }
            for row in actions
            if isinstance(row, dict)
        ],
    }


def model_view_hand_hash(hand: dict[str, Any]) -> str:
    payload = json.dumps(
        model_view_hand_payload(hand),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def canonical_hand_bag(
    chunk: list[dict[str, Any]], *, max_hands: int | None = None
) -> list[dict[str, Any]]:
    """Return a deterministic label-blind ordering of miner-visible hands."""
    ordered = sorted(
        (hand for hand in chunk if isinstance(hand, dict)),
        key=model_view_hand_hash,
    )
    return ordered if max_hands is None else ordered[: max(0, int(max_hands))]
