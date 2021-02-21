import cv2
import torch
import numpy as np
import imutils
from typing import Dict, Union, Tuple, Optional
from multiview_hand_pose_cnn.dataset.data_utils import depth_to_point_cloud, align_point_cloud, project_to_obb_planes


class ToPointCloud:
    """Transform that, given camera's parameters (principal point and focal length),
    turns a depth image into a point cloud."""

    def __init__(self, camera_parameters: Dict[str, Union[Tuple[int, int], float]]):
        self.camera_parameters = camera_parameters

    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Tuple[float, float, float]]]) \
            -> (np.ndarray, Dict[str, Tuple[float, float, float]]):
        # Extract depth image and joints from sample
        (depth_image, joints) = sample
        point_cloud = depth_to_point_cloud(depth_image, self.camera_parameters)
        joints = {j: (coords[0], -coords[1], -coords[2]) for j, coords in joints.items()}
        # Re-build tuple and return it
        return point_cloud, joints


class AlignPointCloud:
    """Transform that aligns a point cloud to its principal components."""

    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Tuple[float, float, float]]]) \
            -> (np.ndarray, Dict[str, Tuple[float, float, float]], np.ndarray):
        # Extract point cloud and joints from sample
        (point_cloud, joints) = sample
        point_cloud, obb, joints = align_point_cloud(point_cloud, joints)
        # Re-build tuple and return it
        return point_cloud, joints, obb


class ProjectToOBBPlanes:
    """Transform that projects the point cloud onto the three OBB planes."""

    def __init__(self, size: Optional[int] = 96):
        self.size = size

    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Tuple[float, float, float]], np.ndarray]) \
            -> (np.ndarray, Dict[str, Tuple[float, float, float]]):
        # Extract point cloud, joints and obb from sample
        (point_cloud, joints, obb) = sample
        # Obtain the projections
        projections = project_to_obb_planes(point_cloud, obb[-1])

        # Refine with morph operations and median filter
        def refine(projection: np.ndarray):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            projection = cv2.morphologyEx(projection, cv2.MORPH_OPEN, kernel)
            projection = cv2.medianBlur(projection, 5)
            return projection

        projections = [refine(p) for p in projections]

        # Resize and pad each projection to the specified value
        def resize_and_pad(projection: np.ndarray):
            max_side = max(projection.shape)
            side = projection.shape.index(max_side)
            # Resize image s.t. the longer side is set to self.size
            projection = imutils.resize(projection, height=self.size) if side == 0 \
                else imutils.resize(projection, width=self.size)
            # Compute padding on the shorter side
            padding = self.size - projection.shape[1 - side]
            padding1 = padding // 2
            padding2 = padding1 if padding % 2 == 0 else padding1 + 1
            if side == 0:  # pad width
                projection = cv2.copyMakeBorder(projection, top=0, bottom=0, left=padding1, right=padding2,
                                                borderType=cv2.BORDER_CONSTANT, value=1.)
            else:  # pad height
                projection = cv2.copyMakeBorder(projection, top=padding1, bottom=padding2, left=0, right=0,
                                                borderType=cv2.BORDER_CONSTANT, value=1.)
            return projection

        projections = [resize_and_pad(p) for p in projections]

        # Re-build tuple and return it
        return projections, joints


class ToTensor(object):
    """Transform that turns a NumPy's array sample into a PyTorch's tensor."""

    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Tuple[float, float, float]]]) \
            -> (torch.Tensor, torch.Tensor):
        # Extract projections and joints from sample
        (projections, joints) = sample
        projections = torch.from_numpy(projections)
        joints = torch.tensor([c for c in joints.values()], dtype=torch.float64)
        # Re-build tuple and return it
        return projections, joints
