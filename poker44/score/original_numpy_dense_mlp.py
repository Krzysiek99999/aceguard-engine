"""Pure NumPy inference for a trained dense Poker44 policy network."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


FAMILY = "original_numpy_dense_policy_mlp"
_BATCH_NORM_EPS = 1e-5


class OriginalNumpyDensePolicyMLP:
    """Torch-free inference wrapper for Linear/BatchNorm/ReLU dense networks."""

    def __init__(
        self,
        *,
        mean: np.ndarray,
        scale: np.ndarray,
        hidden_sizes: Sequence[int],
        dropout: float,
        state_dict: Mapping[str, np.ndarray],
        seed: int,
        batch_norm_eps: float = _BATCH_NORM_EPS,
    ) -> None:
        self.family = FAMILY
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.batch_norm_eps = float(batch_norm_eps)
        self._state = {
            str(name): np.asarray(value, dtype=np.float32)
            for name, value in state_dict.items()
            if not str(name).endswith("num_batches_tracked")
        }
        self._validate()

    def _required(self, name: str) -> np.ndarray:
        try:
            value = self._state[name]
        except KeyError as exc:
            raise ValueError(f"NumPy dense model is missing {name}") from exc
        if not np.isfinite(value).all():
            raise ValueError(f"NumPy dense model has non-finite {name}")
        return value

    def _validate(self) -> None:
        if self.mean.ndim != 1 or not len(self.mean):
            raise ValueError("NumPy dense model mean is invalid")
        if self.scale.shape != self.mean.shape or np.any(self.scale <= 0.0):
            raise ValueError("NumPy dense model scale is invalid")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("NumPy dense model normalization is non-finite")
        if not self.hidden_sizes or any(width <= 0 for width in self.hidden_sizes):
            raise ValueError("NumPy dense model hidden sizes are invalid")
        if not np.isfinite(self.batch_norm_eps) or self.batch_norm_eps <= 0.0:
            raise ValueError("NumPy dense model BatchNorm epsilon is invalid")

        left = len(self.mean)
        for index, right in enumerate(self.hidden_sizes):
            linear = index * 4
            norm = linear + 1
            expected_vector = (right,)
            if self._required(f"hidden.{linear}.weight").shape != (right, left):
                raise ValueError("NumPy dense model linear weight shape mismatch")
            if self._required(f"hidden.{linear}.bias").shape != expected_vector:
                raise ValueError("NumPy dense model linear bias shape mismatch")
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                if self._required(f"hidden.{norm}.{suffix}").shape != expected_vector:
                    raise ValueError("NumPy dense model BatchNorm shape mismatch")
            if np.any(self._required(f"hidden.{norm}.running_var") < 0.0):
                raise ValueError("NumPy dense model BatchNorm variance is invalid")
            left = right
        if self._required("output.weight").shape != (1, left):
            raise ValueError("NumPy dense model output weight shape mismatch")
        if self._required("output.bias").shape != (1,):
            raise ValueError("NumPy dense model output bias shape mismatch")

    def decision_function(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean):
            raise ValueError("NumPy dense model input shape mismatch")
        hidden = (matrix - self.mean) / self.scale
        for index, _ in enumerate(self.hidden_sizes):
            linear = index * 4
            norm = linear + 1
            hidden = (
                hidden @ self._required(f"hidden.{linear}.weight").T
                + self._required(f"hidden.{linear}.bias")
            )
            running_mean = self._required(f"hidden.{norm}.running_mean")
            running_var = self._required(f"hidden.{norm}.running_var")
            hidden = (hidden - running_mean) / np.sqrt(
                running_var + self.batch_norm_eps
            )
            hidden = (
                hidden * self._required(f"hidden.{norm}.weight")
                + self._required(f"hidden.{norm}.bias")
            )
            hidden = np.maximum(hidden, 0.0)
        logits = hidden @ self._required("output.weight").T
        logits = logits + self._required("output.bias")
        result = np.asarray(logits, dtype=float).reshape(-1)
        if not np.isfinite(result).all():
            raise ValueError("NumPy dense model produced non-finite scores")
        return result

    def predict_proba(self, values: Any) -> np.ndarray:
        logits = np.clip(self.decision_function(values), -40.0, 40.0)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])


def export_original_dense_to_numpy(model: Any) -> OriginalNumpyDensePolicyMLP:
    """Convert a trained dense wrapper without retaining a Torch dependency."""

    required = ("mean", "scale", "hidden_sizes", "dropout", "seed", "_state")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(f"dense export source is missing: {', '.join(missing)}")
    return OriginalNumpyDensePolicyMLP(
        mean=np.asarray(model.mean),
        scale=np.asarray(model.scale),
        hidden_sizes=tuple(model.hidden_sizes),
        dropout=float(model.dropout),
        state_dict=dict(model._state),
        seed=int(model.seed),
    )
