import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict, Union, Tuple, Optional, List, Callable, Any
from multiview_hand_pose_cnn.dataset.data_utils import (read_bin_to_depth, depth_to_point_cloud,
                                                        align_point_cloud, project_to_obb_planes)


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
        joints = {j: (coords[0], coords[1], -coords[2]) for j, coords in joints.items()}
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
    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Tuple[float, float, float]], np.ndarray]) \
            -> (np.ndarray, Dict[str, Tuple[float, float, float]]):
        # Extract point cloud, joints from sample
        (point_cloud, joints, obb) = sample
        projections = project_to_obb_planes(point_cloud, obb[-1])

        # Refine with morph operations and median filter
        def refine(projection: np.ndarray):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            projection = cv2.morphologyEx(projection, cv2.MORPH_OPEN, kernel)
            projection = cv2.medianBlur(projection, 5)
            return projection

        projections = np.array([refine(p) for p in projections])
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


class MSRADataset(Dataset):
    """Dataset subclass to load and pre-process the MSRA dataset; this dataset is structured as follows:
    Pi
    |- Gj
        |- joint.txt
        |- k_depth.bin
    where Pi is the i-th subject (9 in total), Gj is the j-th gesture (17 per subject), while k_depth.bin is the k-th
    binary file (500 per gesture); joint.txt is the file containing the 3 coordinates of each 21 hand joint, for every
    500 binary file. Such structure needs to be flattened in order to ease the indexing."""

    def __init__(self, root_folder: str, transforms: Optional[List[Callable[[Any], Any]]] = None):
        # Create a flat list that will contain the samples (bin file + labels)
        self.samples = []
        # List of 21 joints
        joint_list = ['wrist', 'index_mcp', 'index_pip', 'index_dip', 'index_tip', 'middle_mcp', 'middle_pip',
                      'middle_dip', 'middle_tip', 'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip', 'little_mcp',
                      'little_pip', 'little_dip', 'little_tip', 'thumb_mcp', 'thumb_pip', 'thumb_dip', 'thumb_tip']

        # Read the subjects folders
        subjects = glob.glob(os.path.join(root_folder, 'P*'))
        # For each subject, read the hand poses folders
        for subject in subjects:
            hand_poses = glob.glob(os.path.join(subject, '*', ''))
            # For each hand pose, save the path to each binary file
            for hand_pose in hand_poses:
                bin_files = glob.glob(os.path.join(hand_pose, '*.bin'))
                # List of dictionaries joint <-> 3D coordinates
                joints = []
                # Parse joint.txt file and save 3D coordinates of joints
                with open(os.path.join(hand_pose, 'joint.txt')) as f:
                    lines = f.readlines()
                    lines.pop(0)  # discard first line
                    for line in lines:
                        # Split line and convert to floats
                        coords_list = map(float, line.strip().split())
                        # Group the list by (x, y, z) coordinates
                        coords_list = zip(*(iter(coords_list),) * 3)
                        # Create dictionary and append it to list
                        joints.append(
                            {joint: coords for joint, coords in zip(joint_list, coords_list)}
                        )
                self.samples.extend(list(zip(bin_files, joints)))

        # Save transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item) -> Tuple[np.ndarray, Dict[str, Tuple[float, float, float]]]:
        (bin_file, joints) = self.samples[item]
        # Open and read the binary file to produce the actual depth image
        sample = (read_bin_to_depth(bin_file), joints)
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)
        return sample

    def add_transform(self, transform: Callable[[Any], Any]):
        self.transforms.append(transform)
