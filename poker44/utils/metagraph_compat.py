"""Compatibility fallback for singleton-composite Bittensor runtime values."""

from __future__ import annotations

from typing import Any, Optional

from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite
from bittensor.core.metagraph import Metagraph


_SINGLETON_FIELDS = (
    "hotkey",
    "coldkey",
    "netuid",
    "emission",
    "incentive",
    "consensus",
    "trust",
    "validator_trust",
    "dividends",
)


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def normalize_neuron_lite_runtime_value(value: dict[str, Any]) -> dict[str, Any]:
    """Translate the current singleton-composite RPC shape to the SDK shape."""
    normalized = dict(value)
    for field in _SINGLETON_FIELDS:
        if field in normalized:
            normalized[field] = _unwrap_singleton(normalized[field])

    stakes = []
    for item in normalized.get("stake", []):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("unexpected singleton-composite stake entry")
        account_id, amount = item
        stakes.append((_unwrap_singleton(account_id), _unwrap_singleton(amount)))
    normalized["stake"] = stakes
    return normalized


class RuntimeCompatibleMetagraph(Metagraph):
    """Use the SDK normally, falling back only for the known runtime mismatch."""

    _singleton_composite_runtime = False

    def sync(
        self,
        block: Optional[int] = None,
        lite: Optional[bool] = None,
        subtensor=None,
    ):
        if not self._singleton_composite_runtime:
            try:
                return super().sync(block=block, lite=lite, subtensor=subtensor)
            except ValueError as exc:
                message = str(exc)
                if "Invalid type for data" not in message or "Composite" not in message:
                    raise
                self._singleton_composite_runtime = True
        return self._sync_singleton_composite(block=block, subtensor=subtensor)

    def _sync_singleton_composite(self, *, block: Optional[int], subtensor):
        subtensor = self._initialize_subtensor(subtensor=subtensor)
        if block is None:
            block = subtensor.get_current_block()

        result = subtensor.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons_lite",
            params=[(int(self.netuid),)],
            block=block,
        )
        decoded = [normalize_neuron_lite_runtime_value(item) for item in (result or [])]
        self.neurons = NeuronInfoLite.list_from_dicts(decoded)
        self.lite = True
        self._set_metagraph_attributes(block)

        stake_values = [float(neuron.total_stake.tao) for neuron in self.neurons]
        stake = self._create_tensor(
            stake_values,
            dtype=self._dtype_registry["float32"],
        )
        zeros = self._create_tensor(
            [0.0 for _ in self.neurons],
            dtype=self._dtype_registry["float32"],
        )
        self.total_stake = stake
        self.stake = stake
        self.alpha_stake = stake
        self.tao_stake = zeros
        return self


def build_runtime_compatible_metagraph(subtensor, netuid: int) -> RuntimeCompatibleMetagraph:
    metagraph = RuntimeCompatibleMetagraph(
        netuid=int(netuid),
        network=subtensor.chain_endpoint,
        lite=True,
        sync=False,
        subtensor=subtensor,
    )
    metagraph.sync(subtensor=subtensor)
    return metagraph


def serve_axon_runtime_compatible(subtensor, netuid: int, axon):
    """Serve an axon, retrying only the known singleton-composite preflight."""
    response = subtensor.serve_axon(netuid=int(netuid), axon=axon)
    if response.success:
        return response

    message = str(response.message or "")
    if "Invalid type for data" not in message or "Composite" not in message:
        raise RuntimeError(f"serve_axon failed: {message}")

    # The first attempt failed before composing or signing an extrinsic: the
    # SDK could not decode its optional up-to-date check. Skip only that stale
    # read. Registration and endpoint ownership are still enforced on-chain by
    # the signed serve_axon call.
    sentinel = object()
    previous = subtensor.__dict__.get("get_neuron_for_pubkey_and_subnet", sentinel)
    subtensor.get_neuron_for_pubkey_and_subnet = (
        lambda *_args, **_kwargs: type("NullNeuron", (), {"is_null": True})()
    )
    try:
        response = subtensor.serve_axon(
            netuid=int(netuid),
            axon=axon,
            raise_error=True,
        )
    finally:
        if previous is sentinel:
            del subtensor.__dict__["get_neuron_for_pubkey_and_subnet"]
        else:
            subtensor.get_neuron_for_pubkey_and_subnet = previous

    if not response.success:
        raise RuntimeError(f"serve_axon compatibility retry failed: {response.message}")
    return response
