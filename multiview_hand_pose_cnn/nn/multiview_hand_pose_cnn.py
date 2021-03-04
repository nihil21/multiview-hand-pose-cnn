import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiViewHandPoseCNNBranch(nn.Module):
    def __init__(self, use_batch_norm: Optional[bool] = False):
        super(MultiViewHandPoseCNNBranch, self).__init__()
        # Two-staged CNN branches, one for each resolution
        self.fine_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=6),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ) if use_batch_norm else nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.middle_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ) if use_batch_norm else nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.coarse_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ) if use_batch_norm else nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Final linear layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=7776, out_features=6804),
            nn.ReLU(),
            nn.Linear(in_features=6804, out_features=6804)
        )

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Downsample to 24x24 and 48x48
        x_fine = x  # finer scale is 96x96, as the original
        x_middle = F.interpolate(x, size=48)
        x_coarse = F.interpolate(x, size=24)
        # Process each image separately
        x_fine = self.fine_branch(x_fine)
        x_middle = self.middle_branch(x_middle)
        x_coarse = self.coarse_branch(x_coarse)
        # Concatenate output feature maps
        x = torch.cat([x_fine, x_middle, x_coarse], 1)
        # Flatten tensor
        x = x.view(batch_size, -1)
        # Apply linear layers
        x = self.fc(x)
        # Final reshape into (batch_size, 21, 18, 18)
        return x.reshape(batch_size, 21, 18, 18)


class MultiViewHandPoseCNN(nn.Module):
    def __init__(self,
                 xy_branch: Optional[MultiViewHandPoseCNNBranch] = False,
                 yz_branch: Optional[MultiViewHandPoseCNNBranch] = False,
                 zx_branch: Optional[MultiViewHandPoseCNNBranch] = False):
        super(MultiViewHandPoseCNN, self).__init__()
        self.xy_branch = xy_branch if xy_branch is not None else MultiViewHandPoseCNNBranch()
        self.yz_branch = yz_branch if yz_branch is not None else MultiViewHandPoseCNNBranch()
        self.zx_branch = zx_branch if zx_branch is not None else MultiViewHandPoseCNNBranch()

    def forward(self, projections: torch.FloatTensor) -> (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):
        # Input tensor represents a batch of point clouds' projections
        # with shape [batch_size, 3, 96, 96]
        # Take the three projections
        xy_proj = projections[:, 0, :, :].unsqueeze(1)
        yz_proj = projections[:, 1, :, :].unsqueeze(1)
        zx_proj = projections[:, 2, :, :].unsqueeze(1)
        # Feed the projections to the three branches
        xy_heatmap = self.xy_branch(xy_proj)
        yz_heatmap = self.yz_branch(yz_proj)
        zx_heatmap = self.zx_branch(zx_proj)
        return xy_heatmap, yz_heatmap, zx_heatmap

    def inference(self, projections: torch.FloatTensor) -> (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):
        # Input tensor represents a single point clouds' projections
        # with shape [1, 3, 96, 96]
        with torch.no_grad():
            # Take the three projections
            xy_proj = projections[:, 0, :, :].unsqueeze(1)
            yz_proj = projections[:, 1, :, :].unsqueeze(1)
            zx_proj = projections[:, 2, :, :].unsqueeze(1)
            # Feed the projections to the three branches
            xy_heatmap = self.xy_branch(xy_proj)
            yz_heatmap = self.yz_branch(yz_proj)
            zx_heatmap = self.zx_branch(zx_proj)
        return xy_heatmap, yz_heatmap, zx_heatmap
