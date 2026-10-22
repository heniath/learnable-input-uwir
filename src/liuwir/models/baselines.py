"""Paired original map heads and capacity controls on two pinned backbones."""

import torch
from .nafnet import NAFNet
from torch import nn
from .maps import (
    _BackgroundLightExtractor,
    _TransmissionExtractor,
)
from .unet import UNet5ch

from ..config import backbone_defaults as config

CONDITIONS = ("rgb", "bounded", "unbounded", "capacity")
BACKBONES = ("unet", "nafnet")


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


class CapacityBlock(nn.Module):
    def __init__(self, width, bottleneck):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, bottleneck, 1),
            nn.SiLU(),
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(bottleneck, width, 1),
        )
        nn.init.normal_(self.layers[-1].weight, std=0.001)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        return features + self.layers(features)


def capacity_parameters(width, bottleneck):
    return 2 * width * bottleneck + 9 * bottleneck**2 + 2 * bottleneck + width


class EnhancementModel(nn.Module):
    def __init__(self, backbone, condition, seed=0):
        super().__init__()
        if backbone not in BACKBONES or condition not in CONDITIONS:
            raise ValueError((backbone, condition))
        self.backbone_name, self.condition = backbone, condition
        self.capacity_width = None
        # Common parameters precede branch construction. Forking keeps loader
        # and training RNG independent of architecture-specific construction.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if backbone == "unet":
                self.backbone = UNet5ch(in_channels=3)
                width = 64
                old = self.backbone.enc1.net[0]
            else:
                c = config()["nafnet"]
                width = c["width"]
                self.backbone = NAFNet(
                    3,
                    width,
                    c["middle_blocks"],
                    c["encoder_blocks"],
                    c["decoder_blocks"],
                )
                old = self.backbone.intro
            torch.manual_seed(seed + 10000)
            transmission = _TransmissionExtractor(width=32, minimum=0.1)
            background = _BackgroundLightExtractor(width=32)
            self.added_map_parameters = (
                parameter_count(transmission)
                + parameter_count(background)
                + 6 * width * 9
            )
            if condition in ("bounded", "unbounded"):
                self.transmission_extractor = transmission
                self.background_extractor = background
                expanded = nn.Conv2d(9, width, 3, padding=1, bias=old.bias is not None)
                with torch.no_grad():
                    expanded.weight[:, :3].copy_(old.weight)
                    if old.bias is not None:
                        expanded.bias.copy_(old.bias)
                if backbone == "unet":
                    self.backbone.enc1.net[0] = expanded
                else:
                    self.backbone.intro = expanded
            torch.manual_seed(seed + 20000)
            self.capacity = nn.Identity()
            if condition == "capacity":
                self.capacity_width = min(
                    range(1, 257),
                    key=lambda n: (
                        abs(capacity_parameters(width, n) - self.added_map_parameters),
                        n,
                    ),
                )
                self.capacity = CapacityBlock(width, self.capacity_width)
                error = (
                    abs(parameter_count(self.capacity) - self.added_map_parameters)
                    / self.added_map_parameters
                )
                if error >= config()["capacity_added_parameter_tolerance"]:
                    raise ValueError(f"Capacity parameter mismatch: {error}")

    def extract_maps(self, rgb):
        if self.condition == "bounded":
            spatial = self.transmission_extractor(rgb)
            global_map = self.background_extractor(rgb)
        elif self.condition == "unbounded":
            spatial = self.transmission_extractor.net(rgb)
            # Remove only the final sigmoid; retain convs, BN and pooling.
            global_map = self.background_extractor.net[:-1](rgb)
        else:
            raise ValueError("This condition has no maps")
        return spatial, global_map.expand_as(rgb)

    def forward(self, rgb, return_maps=False, maps_override=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[-2:] != (256, 256):
            raise ValueError(
                f"Diagnostic protocol expects N×3×256×256, got {rgb.shape}"
            )
        maps = (
            self.extract_maps(rgb)
            if self.condition in ("bounded", "unbounded")
            else None
        )
        if maps_override is not None:
            if maps is None or any(m.shape != rgb.shape for m in maps_override):
                raise ValueError("Map override shape/condition mismatch")
            maps = maps_override
        inp = rgb if maps is None else torch.cat((rgb, *maps), dim=1)
        b = self.backbone
        if self.backbone_name == "unet":
            e1 = self.capacity(b.enc1(inp))
            e2, e3, e4 = b.enc2(e1), None, None
            e3 = b.enc3(e2)
            e4 = b.enc4(e3)
            bn = b.bottleneck(e4)
            pred = b.head(b.dec1(b.dec2(b.dec3(b.dec4(bn, e4), e3), e2), e1))
        else:
            features = self.capacity(b.intro(inp))
            skips = []
            for encoder, down in zip(b.encoders, b.downs):
                features = encoder(features)
                skips.append(features)
                features = down(features)
            features = b.middle_blks(features)
            for decoder, up, skip in zip(b.decoders, b.ups, reversed(skips)):
                features = decoder(up(features) + skip)
            # NAFNet's residual is anchored to RGB, never to the nine-channel input.
            pred = b.ending(features) + rgb
        return (pred, maps) if return_maps else pred


def convolution_macs(model, shape=(1, 3, 256, 256)):
    counts, handles = [], []

    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            counts.append(
                output.numel()
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
        elif isinstance(module, nn.Linear):
            counts.append(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    try:
        with torch.inference_mode():
            model(torch.zeros(shape, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
    return sum(counts)
