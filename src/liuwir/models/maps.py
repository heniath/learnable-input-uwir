"""Original learned spatial/global heads; names retained for checkpoint compatibility."""
import torch
from torch import nn

class _TransmissionExtractor(nn.Module):
    """Estimate a dense transmission map for every RGB wavelength band."""

    def __init__(self, width: int = 32, minimum: float = 0.1):
        super().__init__()
        self.minimum = minimum
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 3, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        unit_transmission = torch.sigmoid(self.net(image))
        return self.minimum + (1.0 - self.minimum) * unit_transmission


class _BackgroundLightExtractor(nn.Module):
    """Estimate one global background-light value for each RGB wavelength band."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width * 2, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)
