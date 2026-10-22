"""Vary only the global latent width, preserving the historical d=3 function."""

import torch
from torch import nn

from .conditioning import ConditionedUNet


class GlobalLatentUNet(ConditionedUNet):
    def __init__(self, dimension: int, seed: int = 0):
        if type(dimension) is not int or dimension not in (1, 3, 6, 12):
            raise ValueError("Only the prespecified d=1/3/6/12 screen is enabled")
        super().__init__("A2_global", seed)
        self.dimension = dimension
        if dimension != 3:
            old_head = self.base.background_extractor.net[-2]
            old_input = self.base.backbone.enc1.net[0]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 50000)
                head = nn.Conv2d(old_head.in_channels, dimension, 1)
                first = nn.Conv2d(6 + dimension, old_input.out_channels, 3,
                                  padding=1, bias=old_input.bias is not None)
            overlap = min(3, dimension)
            with torch.no_grad():
                head.weight[:overlap].copy_(old_head.weight[:overlap])
                head.bias[:overlap].copy_(old_head.bias[:overlap])
                first.weight[:, :6].copy_(old_input.weight[:, :6])
                first.weight[:, 6:6 + overlap].copy_(old_input.weight[:, 6:6 + overlap])
                if first.bias is not None:
                    first.bias.copy_(old_input.bias)
            self.base.background_extractor.net[-2] = head
            self.base.backbone.enc1.net[0] = first

    def forward(self, rgb, return_maps=False, maps_override=None):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 256, 256):
            raise ValueError("Expected B x 3 x 256 x 256")
        spatial, values = self.extract_maps(rgb)
        if maps_override is not None:
            spatial, values = maps_override
        if spatial.shape != rgb.shape or values.shape != (rgb.shape[0], self.dimension, 1, 1):
            raise ValueError("Invalid spatial/global override shapes")
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite global vector")
        # Spatial slots are structurally dormant in this global-only screen.
        if torch.count_nonzero(spatial):
            raise ValueError("Global-only study requires zero spatial slots")
        inp = torch.cat((rgb, spatial, values.expand(-1, -1, rgb.shape[-2], rgb.shape[-1])), dim=1)
        b = self.base.backbone
        e1 = b.enc1(inp)
        e2 = b.enc2(e1)
        e3 = b.enc3(e2)
        e4 = b.enc4(e3)
        bn = b.bottleneck(e4)
        pred = b.head(b.dec1(b.dec2(b.dec3(b.dec4(bn, e4), e3), e2), e1))
        return (pred, (spatial, values)) if return_maps else pred
