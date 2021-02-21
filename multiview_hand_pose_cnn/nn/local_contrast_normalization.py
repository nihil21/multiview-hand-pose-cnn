import torch
import torch.nn as nn
import kornia.filters as F
from typing import Optional


class LocalContrastNormalization(nn.Module):
    def __init__(self, kernel_size: Optional[int] = 9):
        super(LocalContrastNormalization, self).__init__()
        # Set kernel size and compute sigma
        self.kernel_size = kernel_size
        self.sigma = 0.3 * (kernel_size / 2 - 1) + 0.8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply subtractive normalization
        v = F.gaussian_blur2d(x, (self.kernel_size, self.kernel_size), (self.sigma, self.sigma))
        num = x - v
        # Apply divisive normalization
        sigma = F.gaussian_blur2d(torch.pow(num, 2), (self.kernel_size, self.kernel_size), (self.sigma, self.sigma))
        c = sigma.view(sigma.shape[0], sigma.shape[1], -1).mean(dim=2)
        den = torch.max(sigma, c.view(c.shape[0], c.shape[1], 1, 1).expand_as(sigma))
        # Normalize image
        x = num / den
        x = (x - x.min()) / (x.max() - x.min())
        return x
