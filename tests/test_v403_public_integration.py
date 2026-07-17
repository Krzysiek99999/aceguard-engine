from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
VARIANT = "v403_v402_natural_multisalt60_sequence"
FAMILY = "original_v402_natural_multisalt_sequence_ensemble"
EXPECTED_PACKAGE_PATHS = {
        'data/models/v403_v402_natural_multisalt60_sequence/model.pkl',
        'data/models/v403_v402_natural_multisalt60_sequence/report.json',
        'data/models/v403_v402_natural_multisalt60_sequence/v394_model.pkl',
        'data/models/v403_v402_natural_multisalt60_sequence/v402_model.pkl',
        'poker44/__init__.py',
        'poker44/score/__init__.py',
        'poker44/score/balanced_hash_views.py',
        'poker44/score/chunk_sequence_model.py',
        'poker44/score/chunk_view_aggregation.py',
        'poker44/score/model_view_hand_hash.py',
        'poker44/score/natural_multisalt_sequence_inference.py',
        'poker44/score/natural_order_balanced_views.py',
        'poker44/score/original_hash_bag_features.py',
        'poker44/score/original_multiview_hash_bag_dense_inference.py',
        'poker44/score/original_multiview_hash_bag_inference.py',
        'poker44/score/original_numpy_dense_mlp.py',
        'poker44/score/original_policy_sequence_inference.py',
        'poker44/score/original_policy_sequence_model.py',
        'poker44/score/original_redundancy_features.py',
        'poker44/score/original_schema_contract_features.py',
        'poker44/score/original_tree_surface_features.py',
        'poker44/score/original_tree_surface_inference.py',
        'poker44/score/original_v402_natural_multisalt_sequence_inference.py',
        'poker44/score/scoring.py',
        'poker44/score/sequence_schema.py',
        'poker44/score/temporal_consistency_features.py',
}


def load_miner():
    path = REPO / "neurons/miner_v1.py"
    spec = importlib.util.spec_from_file_location("v403_isolated_public_miner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load isolated public miner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V403PublicIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_miner()

    def test_exact_variant_and_frozen_operating_head(self) -> None:
        cfg = self.module._variant_config(VARIANT)
        self.assertEqual(cfg["family"], FAMILY)
        self.assertEqual(cfg["default_top_n"], 8)
        self.assertEqual(
            cfg["model_file"],
            "data/models/v403_v402_natural_multisalt60_sequence/model.pkl",
        )

    def test_manifest_entry_pins_entire_frozen_package(self) -> None:
        entry = self.module.V403_PUBLIC_MODEL
        report = json.loads(
            (
                REPO
                / "data/models/v403_v402_natural_multisalt60_sequence/report.json"
            ).read_text()
        )
        self.assertEqual(entry["variant"], VARIANT)
        self.assertEqual(entry["family"], FAMILY)
        self.assertEqual(set(entry["package_paths"]), EXPECTED_PACKAGE_PATHS)
        self.assertEqual(entry["default_top_n"], 8)
        self.assertEqual(len(entry["model_sha256"]), 64)
        self.assertEqual(entry["frozen_before_source_date"], "2026-07-15")
        self.assertEqual(
            entry["training_data_statement"], report["training_data_statement"]
        )

    def test_runtime_manifest_covers_all_frozen_package_files(self) -> None:
        miner = object.__new__(self.module.Miner)
        miner.variant_cfg = self.module._variant_config(VARIANT)
        paths = {str(path.relative_to(REPO)) for path in miner._implementation_files()}
        self.assertTrue(EXPECTED_PACKAGE_PATHS.issubset(paths))
        self.assertIn("neurons/miner_v1.py", paths)
        self.assertIn("poker44/utils/model_manifest.py", paths)

    def test_forward_has_explicit_v403_dispatch(self) -> None:
        source = inspect.getsource(self.module.Miner.forward)
        tree = ast.parse(source.lstrip())
        self.assertIn("_score_v403_model", {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)})
        self.assertIn("V403_PUBLIC_MODEL", {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})

    def test_non_top8_environment_is_refused_before_model_io(self) -> None:
        miner = object.__new__(self.module.Miner)
        miner.variant_cfg = self.module._variant_config(VARIANT)
        with patch.dict(os.environ, {"POKER44_V403_TOP_N": "7"}, clear=False):
            with self.assertRaisesRegex(ValueError, "exact top8"):
                miner._score_v403_model([])


if __name__ == "__main__":
    unittest.main()
