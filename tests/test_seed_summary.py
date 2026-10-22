import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from liuwir.config import ROOT, read_config
from liuwir.io import canonical, digest
from liuwir.seed_summary import summarize_validation_seeds


class SeedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = read_config(ROOT / "configs/h2_confirmation.json")
        self.output = Path(self.temporary.name) / self.config["experiment_id"]
        self.output.mkdir()
        manifest = self.output / "manifest.json"
        manifest.write_text("synthetic fixed-image population; not research data\n")
        frozen = {"config": self.config, "manifest_sha256": digest(manifest)}
        (self.output / "frozen.json").write_text(json.dumps(frozen))
        for seed in self.config["seeds"]:
            for condition in self.config["conditions"]:
                directory = self.output / "runs" / f"seed_{seed}" / condition
                directory.mkdir(parents=True)
                # Different baseline levels expose unpaired variance calculations.
                effect = 0 if condition == "B0_input" else seed - 3
                (directory / "best.pt").write_bytes(b"synthetic checkpoint fixture")
                complete = {
                    "identity": {"condition": condition, "seed": seed, "frozen_identity": canonical(frozen)},
                    "best": {"epoch": 20, "psnr": 10 * seed + effect, "ssim": 0.8 + effect / 100},
                    "best_sha256": digest(directory / "best.pt"),
                }
                (directory / "complete.json").write_text(json.dumps(complete))

    def test_intervals_use_paired_seeds(self):
        summary = summarize_validation_seeds(self.config, self.output)
        self.assertEqual(summary["n_seed_pairs"], 5)
        self.assertEqual(len(summary["sources"]), 20)
        self.assertEqual(summary["degrees_of_freedom"], 4)
        psnr = summary["comparisons"][0]["metrics"]["psnr"]
        self.assertEqual(psnr["paired_seed_effects"], [-2, -1, 0, 1, 2])
        self.assertEqual(psnr["mean"], 0)
        self.assertAlmostEqual(psnr["sample_sd"], np.sqrt(2.5))
        # Analytic reference: t(.975, 4) * sqrt(2.5 / 5).
        np.testing.assert_allclose(psnr["ci95"], [-1.9632431615, 1.9632431615], atol=1e-9)

    def test_missing_run_does_not_reduce_n(self):
        path = self.output / "runs/seed_5/B3_combined/complete.json"
        path.rename(path.with_suffix(".missing"))
        with self.assertRaises(FileNotFoundError):
            summarize_validation_seeds(self.config, self.output)

    def test_cli_writes_summary_and_preserves_existing_output(self):
        destination = self.output / "summary.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/summarize_h2_confirmation.py"),
            "--config", str(ROOT / "configs/h2_confirmation.json"),
            "--output-root", self.temporary.name,
            "--destination", str(destination),
        ]
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        original = destination.read_bytes()
        self.assertEqual(json.loads(original)["n_seed_pairs"], 5)
        repeated = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("Summary exists", repeated.stderr)
        self.assertEqual(destination.read_bytes(), original)

    def test_changed_checkpoint_or_manifest_is_rejected(self):
        checkpoint = self.output / "runs/seed_1/B0_input/best.pt"
        checkpoint.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            summarize_validation_seeds(self.config, self.output)
        (self.output / "manifest.json").write_text("changed population")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            summarize_validation_seeds(self.config, self.output)

    def test_wrong_pair_or_nonfinite_metrics_are_rejected(self):
        path = self.output / "runs/seed_1/B0_input/complete.json"
        complete = json.loads(path.read_text())
        complete["identity"]["seed"] = 0
        path.write_text(json.dumps(complete))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            summarize_validation_seeds(self.config, self.output)
        complete["identity"]["seed"] = 1
        complete["best"]["psnr"] = float("nan")
        path.write_text(json.dumps(complete))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            summarize_validation_seeds(self.config, self.output)


if __name__ == "__main__":
    unittest.main()
