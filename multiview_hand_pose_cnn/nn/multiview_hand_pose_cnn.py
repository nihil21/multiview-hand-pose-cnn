import torch
import torch.nn as nn


class LocalContrastNormalization(nn.Module):
    def __init__(self):
        super(LocalContrastNormalization, self).__init__()
        raise NotImplementedError('TODO')

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        pass


class MultiViewHandPoseCNNBranch(nn.Module):
    def __init__(self):
        super(MultiViewHandPoseCNNBranch, self).__init__()
        self.lcn = LocalContrastNormalization()
        raise NotImplementedError('TODO')

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        pass


class MultiViewHandPoseCNN(nn.Module):
    def __init__(self):
        super(MultiViewHandPoseCNN, self).__init__()
        self.xy_branch = MultiViewHandPoseCNNBranch()
        self.xz_branch = MultiViewHandPoseCNNBranch()
        self.yz_branch = MultiViewHandPoseCNNBranch()
        raise NotImplementedError('TODO')

    def forward(self, point_clouds: torch.FloatTensor) -> torch.FloatTensor:
        # Input tensor represents a batch of point clouds with shape [batch_size, rows, cols, 3]
        # Separate channels
        x = point_clouds[:, :, :, 0].unsqueeze(dim=-1)
        y = point_clouds[:, :, :, 1].unsqueeze(dim=-1)
        z = point_clouds[:, :, :, 2].unsqueeze(dim=-1)
        # Project point cloud onto the three orthogonal planes
        xy_proj = torch.cat([x, y], dim=-1)
        xz_proj = torch.cat([x, z], dim=-1)
        yz_proj = torch.cat([y, z], dim=-1)
        # Feed the projections to the three branches
        xy_heatmap = self.xy_branch(xy_proj)
        xz_heatmap = self.xz_branch(xz_proj)
        yz_heatmap = self.yz_branch(yz_proj)
        return xy_heatmap
