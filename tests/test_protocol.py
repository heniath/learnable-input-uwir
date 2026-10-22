import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from liuwir.analysis import paired_report
from liuwir.config import ROOT, read_config
from liuwir.data import read_manifest, transform_pair
from liuwir.experiment import assert_state_equal, pack, restore, validate_checkpoint_identity
from liuwir.io import save_checkpoint, write_csv
from liuwir.loops import EarlyStopping
from liuwir.metrics import metric_rows, summarize
from liuwir.scheduler import CosineAnnealingRestartLR


class ProtocolTests(unittest.TestCase):
    def test_frozen_split_and_overlap_rejection(self):
        manifest = read_manifest(ROOT / "data/uieb.json")
        self.assertEqual(manifest["indices"]["test"], list(range(800, 890)))
        altered = copy.deepcopy(manifest)
        altered["indices"]["validation"][0] = altered["indices"]["train"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_all_recipes_are_explicit(self):
        for path in (ROOT / "configs").glob("*.json"):
            config = read_config(path)
            self.assertEqual(config["training"]["scheduler_epochs"], 100)

    def test_paired_transform_and_replay(self):
        pixels = np.random.default_rng(7).integers(0, 256, (63, 93, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        random.seed(9)
        first = transform_pair(image, image, training=True, size=32)
        random.seed(9)
        second = transform_pair(image, image, training=True, size=32)
        torch.testing.assert_close(first[0], first[1], atol=0, rtol=0)
        torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)

    def test_per_image_metric_and_pair_guards(self):
        truth = torch.full((2, 3, 16, 16), 0.5)
        prediction = truth.clone()
        prediction[0] += 0.1
        prediction[1] += 0.2
        rows = metric_rows(prediction, truth, ["a", "b"])
        expected = (20 + -10 * np.log10(0.04)) / 2
        self.assertAlmostEqual(summarize(rows)["psnr"], expected, places=5)
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            write_csv(a, rows)
            write_csv(b, rows)
            self.assertEqual(paired_report(a, b)["delta_psnr"], 0)
            write_csv(b, rows[::-1])
            with self.assertRaises(ValueError):
                paired_report(a, b)

    def test_checkpoint_continuation_matches_uninterrupted_update(self):
        torch.manual_seed(11)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
        scheduler = CosineAnnealingRestartLR(optimizer, [100], [1.0], 0.000001)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        early = EarlyStopping(20)
        objects = (model, None, optimizer, scheduler, scaler, early)

        def step():
            optimizer.zero_grad()
            model(torch.rand(4, 3)).square().mean().backward()
            optimizer.step()
            scheduler.step()

        step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_checkpoint(path, pack(objects))
            step()
            uninterrupted = copy.deepcopy(pack(objects))
            restore(objects, torch.load(path, map_location="cpu", weights_only=False))
            step()
            assert_state_equal(uninterrupted, pack(objects))

    def test_checkpoint_labels_and_dimension_aliases(self):
        config = read_config(ROOT / "configs/h2_pilot.json")
        state = {"identity": {"condition": "B0_input", "seed": 0}}
        validate_checkpoint_identity(state, config, "B0_input", 0)
        with self.assertRaises(ValueError):
            validate_checkpoint_identity(state, config, "B3_combined", 0)
        with self.assertRaises(ValueError):
            validate_checkpoint_identity(state, config, "B0_input", 1)
        config = read_config(ROOT / "configs/global_dimension.json")
        validate_checkpoint_identity({"identity": {"condition": "d_6", "seed": 1}}, config, "d6", 1)
        validate_checkpoint_identity({"identity": {"condition": "A2_global", "seed": 0}}, config, "d3", 0)


if __name__ == "__main__":
    unittest.main()
