"""Fully-convolutional AlphaZero-style net for Hex.

Size-agnostic by construction:
  - trunk is conv + residual blocks (no board-size-dependent dimensions)
  - policy head is a 1x1 conv to one channel, flattened -> n*n logits
  - value head global-average-pools, so it accepts any board size

So a single set of weights applies to any n; we only train n=7 here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class HexNet(nn.Module):
    def __init__(self, channels=64, blocks=5, in_planes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        # policy: 1x1 conv -> 1 plane -> flatten
        self.p_conv = nn.Conv2d(channels, 1, 1)
        # value: 1x1 conv -> global avg pool -> mlp -> scalar
        self.v_conv = nn.Conv2d(channels, 16, 1)
        self.v_fc1 = nn.Linear(16, 32)
        self.v_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.tower(self.stem(x))
        b, _, n, m = x.shape
        p = self.p_conv(x).reshape(b, n * m)          # policy logits over cells
        v = self.v_conv(x).mean(dim=(2, 3))           # global average pool
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(1)      # value in (-1, 1)
        return p, v
