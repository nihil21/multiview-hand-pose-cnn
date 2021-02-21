import torch
import torch.nn as nn
from multiview_hand_pose_cnn.nn.local_contrast_normalization import LocalContrastNormalization as LCN


class MultiViewHandPoseCNNBranch(nn.Module):
    def __init__(self):
        super(MultiViewHandPoseCNNBranch, self).__init__()
        self.lcn = LCN()
        raise NotImplementedError('TODO')

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        # Apply Local Contrast Normalization
        x = self.lcn(x)
        return x


class MultiViewHandPoseCNN(nn.Module):
    def __init__(self):
        super(MultiViewHandPoseCNN, self).__init__()
        self.xy_branch = MultiViewHandPoseCNNBranch()
        self.xz_branch = MultiViewHandPoseCNNBranch()
        self.yz_branch = MultiViewHandPoseCNNBranch()
        raise NotImplementedError('TODO')

    def forward(self, projections: torch.FloatTensor) -> torch.FloatTensor:
        # Input tensor represents a batch of point clouds' projections
        # with shape [batch_size, rows, cols, 3]
        # Take the three projections
        xy_proj = projections[:, :, :, 0].unsqueeze(dim=-1)
        yz_proj = projections[:, :, :, 1].unsqueeze(dim=-1)
        zx_proj = projections[:, :, :, 2].unsqueeze(dim=-1)
        # Feed the projections to the three branches
        xy_heatmap = self.xy_branch(xy_proj)
        yz_heatmap = self.yz_branch(yz_proj)
        zx_heatmap = self.xz_branch(zx_proj)
        return xy_heatmap
