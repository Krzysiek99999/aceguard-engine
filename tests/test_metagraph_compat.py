from __future__ import annotations

import unittest
from unittest.mock import patch

from bittensor.core.metagraph import Metagraph

from poker44.utils.metagraph_compat import (
    RuntimeCompatibleMetagraph,
    normalize_neuron_lite_runtime_value,
)


class MetagraphCompatibilityTests(unittest.TestCase):
    def test_normalizes_only_known_singleton_composites(self) -> None:
        account = tuple(range(32))
        decoded = normalize_neuron_lite_runtime_value(
            {
                "hotkey": (account,),
                "coldkey": (account,),
                "netuid": (126,),
                "emission": (123,),
                "incentive": (456,),
                "consensus": (789,),
                "trust": (101,),
                "validator_trust": (112,),
                "dividends": (131,),
                "stake": (((account,), (999,)),),
                "uid": 40,
            }
        )
        self.assertEqual(decoded["hotkey"], account)
        self.assertEqual(decoded["netuid"], 126)
        self.assertEqual(decoded["emission"], 123)
        self.assertEqual(decoded["stake"], [(account, 999)])
        self.assertEqual(decoded["uid"], 40)

    def test_fallback_is_limited_to_known_composite_type_error(self) -> None:
        graph = RuntimeCompatibleMetagraph(netuid=126, sync=False)
        with patch.object(
            Metagraph,
            "sync",
            side_effect=ValueError("Invalid type for data: 126, type_def: Composite"),
        ):
            with patch.object(
                RuntimeCompatibleMetagraph,
                "_sync_singleton_composite",
                return_value=graph,
            ) as fallback:
                self.assertIs(graph.sync(), graph)
        fallback.assert_called_once()
        self.assertTrue(graph._singleton_composite_runtime)

    def test_unrelated_value_error_is_not_swallowed(self) -> None:
        graph = RuntimeCompatibleMetagraph(netuid=126, sync=False)
        with patch.object(Metagraph, "sync", side_effect=ValueError("different")):
            with self.assertRaisesRegex(ValueError, "different"):
                graph.sync()


if __name__ == "__main__":
    unittest.main()
