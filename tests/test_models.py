import unittest

import torch

from liuwir.models.conditioning import ConditionedUNet, GlobalFiLM, ModelSpec, SpatialModulation
from liuwir.models.baselines import EnhancementModel

torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def test_shared_initialization_and_absent_branches(self):
        both = ConditionedUNet("A3_both", 7)
        for condition, absent, active in (
            ("A1_spatial", "background_extractor", "transmission_extractor"),
            ("A2_global", "transmission_extractor", "background_extractor"),
        ):
            candidate = ConditionedUNet(condition, 7)
            self.assertFalse(hasattr(candidate.base, absent))
            self.assertTrue(hasattr(candidate.base, active))
            reference = both.state_dict()
            for name, value in candidate.state_dict().items():
                torch.testing.assert_close(value, reference[name], rtol=0, atol=0)
        rgb = ConditionedUNet("A0_rgb", 7)
        self.assertEqual(rgb.base.backbone.enc1.net[0].in_channels, 9)
        self.assertEqual(rgb.parameter_counts()["dormant_input_weights"], 3456)

    def test_a3_matches_original_bounded_function(self):
        torch.manual_seed(123)
        x = torch.rand(1, 3, 256, 256)
        original = EnhancementModel("unet", "bounded", 0).eval()
        candidate = ConditionedUNet("A3_both", 0).eval()
        with torch.inference_mode():
            torch.testing.assert_close(candidate(x), original(x), rtol=0, atol=0)

    def test_skip_modulation_preserves_encoder_stream(self):
        model = ConditionedUNet("B1_spatial_skip", 0).eval()
        x = torch.rand(1, 3, 256, 256)
        captured = []
        handle = model.base.backbone.enc3.register_forward_hook(
            lambda module, args, output: captured.append(output.detach().clone())
        )
        try:
            with torch.inference_mode():
                before = model(x)
                for adapter in model.spatial_modulation.values():
                    adapter.alpha.fill_(1)
                    adapter.proj.weight.fill_(0.1)
                    adapter.proj.bias.fill_(0.1)
                after = model(x)
            torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
            self.assertFalse(torch.equal(before, after))
        finally:
            handle.remove()

    def test_identity_and_gradient_unlock(self):
        torch.manual_seed(5)
        feature = torch.rand(2, 8, 8, 8)
        maps = torch.rand(2, 3, 16, 16)
        spatial = SpatialModulation(8)
        torch.testing.assert_close(spatial(feature, maps), feature, rtol=0, atol=0)
        spatial(feature, maps).sum().backward()
        self.assertGreater(spatial.alpha.grad.abs().max().item(), 0)
        self.assertEqual(spatial.proj.weight.grad.abs().max().item(), 0)
        film = GlobalFiLM(8)
        values = torch.rand(2, 3, 1, 1)
        torch.testing.assert_close(film(feature, values), feature, rtol=0, atol=0)
        film(feature, values).sum().backward()
        self.assertGreater(film.net[-1].weight.grad.abs().max().item(), 0)
        with self.assertRaises(ValueError):
            film(feature, values.expand(2, 3, 8, 8))

    def test_identity_modulation_matches_input_baseline(self):
        x = torch.rand(1, 3, 256, 256)
        baseline = ConditionedUNet("B0_input", 1).eval()
        with torch.inference_mode():
            expected = baseline(x)
            for condition in ("B1_spatial_skip", "B2_global_bottleneck", "B3_combined"):
                candidate = ConditionedUNet(condition, 1).eval()
                torch.testing.assert_close(candidate(x), expected, rtol=0, atol=0)

    def test_invalid_specs(self):
        for kwargs in (
            {"spatial_sites": ("e5",)},
            {"spatial_sites": ("e1", "e1")},
            {"spatial": False, "spatial_sites": ("e1",)},
        ):
            with self.assertRaises(ValueError):
                ModelSpec(**kwargs)


if __name__ == "__main__":
    unittest.main()
