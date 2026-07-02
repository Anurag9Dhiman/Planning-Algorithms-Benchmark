"""
CNN encoder for LeWM.

Trained jointly with the dynamics / reward / value heads (no pretrained weights).
Input : (B, 3, H, W) float in [0, 1]
Output: (B, latent_dim) float
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Three conv blocks (stride-2 downsampling) followed by a linear projection.
    Produces a fixed-size latent regardless of input resolution.
    """

    def __init__(self, latent_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            # block 1:  (B, 3,   H,   W) -> (B, 32, H/2, W/2)
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            # block 2:  (B, 32,  H/2, W/2) -> (B, 64, H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # block 3:  (B, 64,  H/4, W/4) -> (B, 128, H/8, W/8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            # block 4:  (B, 128, H/8, W/8) -> (B, 256, H/16, W/16)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )

        # Global average pooling -> fixed (B, 256) regardless of spatial size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W), values in [0, 1]
        h = self.conv(x)          # (B, 256, H/16, W/16)
        h = self.pool(h)          # (B, 256, 1, 1)
        h = h.flatten(1)          # (B, 256)
        return self.norm(self.proj(h))   # (B, latent_dim)
