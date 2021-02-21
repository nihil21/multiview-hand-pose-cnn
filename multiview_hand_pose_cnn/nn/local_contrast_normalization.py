import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LocalContrastNormalization(nn.Module):
    def __init__(self, kernel_size: Optional[int] = 9):
        super(LocalContrastNormalization, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert tensor to NumPy's array
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
        # Apply subtractive normalization
        v = cv2.GaussianBlur(x, (self.kernel_size, self.kernel_size), 0)
        num = x - v
        # Apply divisive normalization
        sigma = cv2.GaussianBlur(num**2, (self.kernel_size, self.kernel_size), 0)**0.5
        c = sigma.mean()
        den = np.maximum(c, sigma)
        # Normalize image
        x = num / den
        x = cv2.normalize(x, dst=None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        # Return normalized image
        return torch.from_numpy(x)
