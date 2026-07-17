from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def load_miner():
    path = REPO / "neurons/miner_v1.py"
    spec = importlib.util.spec_from_file_location("contract_public_miner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load isolated public miner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContractPublicIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_miner()

    def _specs(self):
        return (
            (self.module.V415_PUBLIC_MODEL, 20, "2026-07-16"),
            (self.module.V417_PUBLIC_MODEL, 15, "2026-07-17"),
        )

    def test_exact_variants_and_frozen_heads(self) -> None:
        for entry, top_n, source_date in self._specs():
            with self.subTest(variant=entry["variant"]):
                cfg = self.module._variant_config(entry["variant"])
                self.assertEqual(cfg["family"], entry["family"])
                self.assertEqual(cfg["default_top_n"], top_n)
                self.assertEqual(cfg["model_file"], entry["model_file"])
                self.assertEqual(entry["frozen_before_source_date"], source_date)

    def test_manifest_entries_pin_models_reports_and_runtime(self) -> None:
        required_runtime = {
            "poker44/score/v112_super_inference.py",
            "poker44/score/robust_schema/features.py",
            "poker44/score/sequence_schema.py",
            "poker44/score/temporal_consistency_features.py",
            "poker44/score/statistical_v25.py",
            "poker44/score/action_anomaly_features.py",
            "poker44/score/rank_cap_remap.py",
        }
        for entry, _top_n, _source_date in self._specs():
            with self.subTest(variant=entry["variant"]):
                paths = set(entry["package_paths"])
                self.assertIn(entry["model_file"], paths)
                self.assertIn(
                    str(Path(entry["model_file"]).with_name("report.json")),
                    paths,
                )
                self.assertTrue(required_runtime.issubset(paths))
                model_path = REPO / entry["model_file"]
                self.assertEqual(
                    hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    entry["model_sha256"],
                )
                report = json.loads(model_path.with_name("report.json").read_text())
                self.assertEqual(
                    entry["training_data_statement"],
                    report["training_data_statement"],
                )

    def test_runtime_manifest_covers_complete_packages(self) -> None:
        for entry, _top_n, _source_date in self._specs():
            with self.subTest(variant=entry["variant"]):
                miner = object.__new__(self.module.Miner)
                miner.variant_cfg = self.module._variant_config(entry["variant"])
                paths = {
                    str(path.relative_to(REPO))
                    for path in miner._implementation_files()
                }
                self.assertTrue(set(entry["package_paths"]).issubset(paths))
                self.assertIn("neurons/miner_v1.py", paths)
                self.assertIn("poker44/utils/model_manifest.py", paths)

    def test_forward_has_explicit_contract_dispatch(self) -> None:
        source = inspect.getsource(self.module.Miner.forward)
        tree = ast.parse(source.lstrip())
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertIn("_score_contract_public_model", attributes)
        self.assertIn("V415_PUBLIC_MODEL", names)
        self.assertIn("V417_PUBLIC_MODEL", names)

    def test_runtime_uses_exact_frozen_positive_counts(self) -> None:
        raw = np.linspace(0.0, 1.0, 25)
        chunks = [[] for _ in range(25)]
        for entry, top_n, _source_date in self._specs():
            with self.subTest(variant=entry["variant"]):
                miner = object.__new__(self.module.Miner)
                miner.variant_cfg = self.module._variant_config(entry["variant"])
                with patch(
                    "poker44.score.v112_super_inference.score_chunks",
                    return_value=raw,
                ):
                    served = miner._score_contract_public_model(chunks)
                self.assertEqual(len(served), len(chunks))
                self.assertEqual(sum(value >= 0.5 for value in served), top_n)
                self.assertTrue(np.isfinite(np.asarray(served, dtype=float)).all())


if __name__ == "__main__":
    unittest.main()
