"""Compact baseline; outputs raw class logits, not calibrated probabilities."""

import torch
from torch import nn
from torch.nn import functional as F


def block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(4, outputs),
        nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(4, outputs),
        nn.ReLU(inplace=True),
    )


class CompactUNet(nn.Module):
    def __init__(self, in_channels: int = 4, classes: int = 4, width: int = 8):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError('width must be a positive multiple of four')
        self.encoder1 = block(in_channels, width)
        self.encoder2 = block(width, width * 2)
        self.bridge = block(width * 2, width * 4)
        self.decoder2 = block(width * 6, width * 2)
        self.decoder1 = block(width * 3, width)
        self.head = nn.Conv2d(width, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.encoder1(x)
        b = self.encoder2(F.max_pool2d(a, 2))
        c = self.bridge(F.max_pool2d(b, 2))
        c = F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False)
        c = self.decoder2(torch.cat((c, b), dim=1))
        c = F.interpolate(c, size=a.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.decoder1(torch.cat((c, a), dim=1)))
