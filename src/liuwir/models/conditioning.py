"""Ablate map branches and condition fixed U-Net features without vendor edits."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .baselines import EnhancementModel


@dataclass(frozen=True)
class ModelSpec:
    spatial: bool = True
    global_branch: bool = True
    input_concat: bool = True
    spatial_sites: tuple[str, ...] = ()
    global_sites: tuple[str, ...] = ()
    constant_condition: bool = False

    def __post_init__(self):
        allowed = {"e1", "e2", "bottleneck"}
        for sites in (self.spatial_sites, self.global_sites):
            if len(set(sites)) != len(sites) or set(sites) - allowed:
                raise ValueError(f"Invalid or duplicate conditioning sites: {sites}")
        if self.spatial_sites and not self.spatial:
            raise ValueError("Spatial modulation needs the spatial branch")
        if self.global_sites and not self.global_branch:
            raise ValueError("Global modulation needs the global branch")


SPECS = {
    "A0_rgb": ModelSpec(spatial=False, global_branch=False),
    "A1_spatial": ModelSpec(global_branch=False),
    "A2_global": ModelSpec(spatial=False),
    "A3_both": ModelSpec(),
    "B0_input": ModelSpec(),
    "B1_spatial_skip": ModelSpec(spatial_sites=("e1", "e2")),
    "B2_global_bottleneck": ModelSpec(global_sites=("bottleneck",)),
    "B3_combined": ModelSpec(spatial_sites=("e1", "e2"), global_sites=("bottleneck",)),
    "P_spatial_bottleneck": ModelSpec(spatial_sites=("bottleneck",)),
    "P_global_skip": ModelSpec(global_sites=("e1", "e2")),
    "B1_constant": ModelSpec(spatial_sites=("e1", "e2"), constant_condition=True),
    "B2_constant": ModelSpec(global_sites=("bottleneck",), constant_condition=True),
    "B3_constant": ModelSpec(spatial_sites=("e1", "e2"), global_sites=("bottleneck",), constant_condition=True),
    "B1_no_concat": ModelSpec(global_branch=False, input_concat=False, spatial_sites=("e1", "e2")),
    "B2_no_concat": ModelSpec(spatial=False, input_concat=False, global_sites=("bottleneck",)),
    "B3_no_concat": ModelSpec(input_concat=False, spatial_sites=("e1", "e2"), global_sites=("bottleneck",)),
}


class SpatialModulation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Keep this projection random. Zeroing both projection and alpha kills gradients.
        self.proj = nn.Conv2d(3, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, feature, maps):
        maps = F.interpolate(maps, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return feature * (1 + self.alpha * torch.tanh(self.proj(maps)))


class GlobalFiLM(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature, values):
        if values.ndim == 4 and values.shape[-2:] == (1, 1):
            values = values[:, :, 0, 0]
        if values.shape != (feature.shape[0], 3):
            raise ValueError("GlobalFiLM expects B×3 or B×3×1×1, before broadcast")
        gamma, beta = self.net(values).chunk(2, dim=1)
        return feature * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ConditionedUNet(nn.Module):
    """E1/E2 gates act on decoder skip copies, never on the encoder stream.

    The 9-channel first convolution is retained for all conditions. Disabled
    slots are zeros; disabled heads are removed and not evaluated. This counts
    dormant first-convolution columns explicitly rather than calling them active.
    """

    def __init__(self, condition: str, seed: int = 0):
        super().__init__()
        self.spec = SPECS[condition]
        self.condition = condition
        self.base = EnhancementModel("unet", "bounded", seed)
        if not self.spec.spatial:
            del self.base.transmission_extractor
        if not self.spec.global_branch:
            del self.base.background_extractor
        widths = {"e1": 64, "e2": 128, "bottleneck": 1024}
        # Isolate architecture-specific construction from training and shared heads.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 30000)
            self.spatial_modulation = nn.ModuleDict({site: SpatialModulation(widths[site]) for site in self.spec.spatial_sites})
            torch.manual_seed(seed + 40000)
            self.global_modulation = nn.ModuleDict({site: GlobalFiLM(widths[site]) for site in self.spec.global_sites})

    def extract_maps(self, rgb):
        spatial = self.base.transmission_extractor(rgb) if self.spec.spatial else torch.zeros_like(rgb)
        global_values = self.base.background_extractor(rgb) if self.spec.global_branch else rgb.new_zeros((rgb.shape[0], 3, 1, 1))
        return spatial, global_values

    def modulate(self, site, feature, spatial, global_values):
        if self.spec.constant_condition:
            # Same adapter shapes/parameter counts, but no image-conditioned input.
            spatial = torch.ones_like(spatial)
            global_values = torch.ones_like(global_values)
        if site in self.spatial_modulation:
            feature = self.spatial_modulation[site](feature, spatial)
        if site in self.global_modulation:
            feature = self.global_modulation[site](feature, global_values)
        return feature

    def forward(self, rgb, return_maps=False, maps_override=None):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 256, 256):
            raise ValueError(f"Expected B×3×256×256, got {tuple(rgb.shape)}")
        spatial, global_values = self.extract_maps(rgb)
        if maps_override is not None:
            spatial, global_values = maps_override
            if spatial.shape != rgb.shape or global_values.shape != (rgb.shape[0], 3, 1, 1):
                raise ValueError("Override expects S=B×3×256×256 and G=B×3×1×1")
        if self.spec.input_concat:
            inp = torch.cat((rgb, spatial, global_values.expand_as(rgb)), dim=1)
        else:
            inp = torch.cat((rgb, torch.zeros_like(rgb), torch.zeros_like(rgb)), dim=1)
        b = self.base.backbone
        e1 = b.enc1(inp)
        e2 = b.enc2(e1)
        e3 = b.enc3(e2)
        e4 = b.enc4(e3)
        bn = self.modulate("bottleneck", b.bottleneck(e4), spatial, global_values)
        s1 = self.modulate("e1", e1, spatial, global_values)
        s2 = self.modulate("e2", e2, spatial, global_values)
        pred = b.head(b.dec1(b.dec2(b.dec3(b.dec4(bn, e4), e3), s2), s1))
        return (pred, (spatial, global_values)) if return_maps else pred

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        inactive_slots = 2 if not self.spec.input_concat else int(not self.spec.spatial) + int(not self.spec.global_branch)
        dormant = inactive_slots * 3 * 64 * 3 * 3
        return {"registered": total, "structurally_active": total - dormant, "dormant_input_weights": dormant,
                "adapters": sum(p.numel() for group in (self.spatial_modulation, self.global_modulation) for p in group.parameters())}
