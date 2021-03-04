import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List


class ToTensor:
    """Transform that turns a NumPy's array sample into a PyTorch's tensor."""

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]) -> (torch.Tensor, torch.Tensor):
        (projections, heatmaps) = sample
        projections = torch.tensor(projections, dtype=torch.float32)
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)
        return projections, heatmaps


class MSRADataset(Dataset):

    def __init__(self, root_folder: str, transforms: Optional[List] = None):
        super(MSRADataset, self).__init__()
        # Store the path to each sample
        self.samples = sorted(glob.glob(os.path.join(root_folder, '*.npz')))
        # Store transforms
        if transforms is not None:
            self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:
        filename = self.samples[item]
        # Load data from the .npz
        data = np.load(filename)
        sample = (data['x'], data['y'])
        # Apply additional transforms
        for transform in self.transforms:
            sample = transform(sample)
        return sample
