import cv2
import torch
import numpy as np
from typing import Dict, Union, Tuple
from multiview_hand_pose_cnn.dataset.data_utils import (depth_to_point_cloud, align_point_cloud,
                                                        project_to_obb_planes, generate_heatmaps)


class ToPointCloud:
    """Transform that, given camera's parameters (principal point and focal length),
    turns a depth image into a point cloud."""

    def __init__(self, camera_parameters: Dict[str, Union[Tuple[int, int], float]]):
        self.camera_parameters = camera_parameters

    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Tuple[float, float, float]]]) \
            -> (np.ndarray, Dict[str, Tuple[float, float, float]]):
        # Extract depth image and joints from sample
        (depth_image, joints) = sample
        point_cloud = depth_to_point_cloud(depth_image, self.camera_parameters)  # transform depth image in point cloud
        # Re-build tuple and return it
        return point_cloud, joints


class AlignPointCloud:
    """Transform that aligns a point cloud to its principal components."""

    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Tuple[float, float, float]]]) \
            -> (np.ndarray, Dict[str, Tuple[float, float, float]], np.ndarray):
        # Extract point cloud and joints from sample
        (point_cloud, joints) = sample
        point_cloud, obb, vertex_idx, joints = align_point_cloud(point_cloud, joints)
        # Re-build tuple and return it
        return point_cloud, joints, obb, vertex_idx


class ProjectToOBBPlanes:
    """Transform that projects the point cloud onto the three OBB planes."""

    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Tuple[float, float, float]], np.ndarray, int]) \
            -> (np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        # Extract point cloud, joints, obb and vertex index from sample
        (point_cloud, joints, obb, vertex_idx) = sample
        projections, boundaries = project_to_obb_planes(point_cloud, obb[vertex_idx])

        # Refine with morph operations and median filter
        def refine(projection: np.ndarray):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            projection = cv2.morphologyEx(projection, cv2.MORPH_OPEN, kernel)
            projection = cv2.medianBlur(projection, 5)
            return projection

        # Refine, resize to 96x96 and convert to NumPy's array
        projections = np.array(
            [cv2.resize(refine(p), dsize=(96, 96)) for p in projections]
        )

        # Project also joints coordinates and obtain heatmaps
        joints = {j: generate_heatmaps(np.array(c), boundaries) for j, c in joints.items()}

        # Re-build tuple and return it
        return projections, joints


class ToTensor(object):
    """Transform that turns a NumPy's array sample into a PyTorch's tensor."""

    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Tuple[float, float, float]]]) \
            -> (torch.Tensor, torch.Tensor):
        # Extract projections and joints from sample
        (projections, joints) = sample
        projections = torch.from_numpy(projections)
        joints = torch.tensor([c for c in joints.values()], dtype=torch.float32)
        # Re-build tuple and return it
        return projections, joints
