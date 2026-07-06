from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _size_bucket(amount: float, pot_before: float) -> str:
    if amount <= 0.0:
        return "z"
    if pot_before > 0.0:
        ratio = amount / pot_before
        if ratio < 0.25:
            return "rp0"
        if ratio < 0.50:
            return "rp1"
        if ratio < 0.90:
            return "rp2"
        if ratio < 1.50:
            return "rp3"
        return "rp4"
    if amount < 0.5:
        return "bb0"
    if amount < 1.0:
        return "bb1"
    if amount < 2.0:
        return "bb2"
    if amount < 4.0:
        return "bb3"
    if amount < 8.0:
        return "bb4"
    if amount < 16.0:
        return "bb5"
    return "bb6"


def _hand_tokens(hand: dict[str, Any]) -> list[str]:
    metadata = hand.get("metadata") or {}
    hero = _safe_int(metadata.get("hero_seat"), 0)
    button = _safe_int(metadata.get("button_seat"), 0)
    max_seats = max(1, _safe_int(metadata.get("max_seats"), 6))
    out: list[str] = []
    for action in hand.get("actions") or []:
        if not isinstance(action, dict):
            continue
        street = str(action.get("street") or "x").lower()[:1]
        action_type = str(action.get("action_type") or "x").lower()
        actor = _safe_int(action.get("actor_seat"), 0)
        role = "h" if hero and actor == hero else "o"
        try:
            rel = (actor - button) % max_seats if actor and button else 0
        except Exception:
            rel = 0
        amount = _safe_float(action.get("normalized_amount_bb"), _safe_float(action.get("amount"), 0.0))
        pot_before = _safe_float(action.get("pot_before"), 0.0)
        bucket = _size_bucket(max(0.0, amount), max(0.0, pot_before))
        out.append(f"{street}:{role}:{rel}:{action_type}:{bucket}")
    return out


def chunk_sentence(chunk: list[dict[str, Any]]) -> str:
    tokens: list[str] = []
    for hand in chunk or []:
        if not isinstance(hand, dict):
            continue
        hand_tokens = _hand_tokens(hand)
        tokens.extend(hand_tokens)
        tokens.append("EOH")
    return " ".join(tokens) or "EMPTY"


class ChunkNgramRanker:
    """Sparse action n-gram side learner for live-sized Poker44 chunks."""

    def __init__(
        self,
        *,
        ngram_max: int = 4,
        min_df: int = 2,
        c: float = 1.0,
        random_state: int = 126,
    ) -> None:
        self.ngram_max = int(ngram_max)
        self.min_df = int(min_df)
        self.c = float(c)
        self.random_state = int(random_state)
        self.vectorizer: TfidfVectorizer | None = None
        self.model: LogisticRegression | None = None

    def fit(
        self,
        chunks: list[list[dict[str, Any]]],
        y: np.ndarray | list[int],
        sample_weight: np.ndarray | list[float] | None = None,
    ) -> "ChunkNgramRanker":
        sentences = [chunk_sentence(chunk) for chunk in chunks]
        vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"[^\s]+",
            ngram_range=(1, self.ngram_max),
            min_df=self.min_df,
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(sentences)
        model = LogisticRegression(
            C=self.c,
            max_iter=3000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        model.fit(X, np.asarray(y, dtype=int), sample_weight=sample_weight)
        self.vectorizer = vectorizer
        self.model = model
        return self

    def predict_chunk_scores(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("ChunkNgramRanker is not fitted")
        X = self.vectorizer.transform([chunk_sentence(chunk) for chunk in chunks])
        return np.clip(self.model.predict_proba(X)[:, 1], 0.0, 1.0).astype(float).tolist()

    def predict_proba(self, chunks: list[list[dict[str, Any]]]) -> np.ndarray:
        scores = np.asarray(self.predict_chunk_scores(chunks), dtype=float)
        return np.column_stack([1.0 - scores, scores])
