import os
import glob
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm.notebook import tqdm
from typing import Dict, Union, Tuple, Optional
from multiview_hand_pose_cnn.dataset.data_utils import (read_bin_to_depth, depth_to_point_cloud, align_point_cloud,
                                                        project_to_obb_planes, generate_heatmaps,
                                                        local_contrast_normalization)
from multiview_hand_pose_cnn.dataset.errors import InvalidDataError


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
            -> (np.ndarray, np.ndarray, np.ndarray):
        # Extract point cloud and joints from sample
        (point_cloud, joints) = sample
        point_cloud, obb, joints = align_point_cloud(point_cloud, joints)
        # Re-build tuple and return it
        return point_cloud, joints, obb


class ProjectToOBBPlanes:
    """Transform that projects the point cloud onto the three OBB planes."""

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray, np.ndarray]) \
            -> (np.ndarray, np.ndarray):
        # Extract point cloud, joints, obb and vertex index from sample
        (point_cloud, joints, obb) = sample
        projections, boundaries = project_to_obb_planes(point_cloud, vertex=obb[-1], res=96)

        # Refine with morph operations and median filter
        def refine(projection: np.ndarray):
            # Open and apply median filter
            projection = cv2.morphologyEx(projection,
                                          cv2.MORPH_OPEN,
                                          kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            projection = cv2.medianBlur(projection, ksize=5)

            return projection

        # Refine, normalize and convert list to NumPy's array
        projections = np.array(
            [refine(p.astype('float32')) for p in projections], dtype='float64'
        )

        # Project also joints coordinates and obtain heatmaps
        heatmaps = np.array(
            [generate_heatmaps(c, boundaries) for c in joints]
        )

        # Re-build tuple and return it
        return projections, heatmaps


class LocalContrastNormalization:
    """Transform that applies Local Contrast Normalization to each projection."""

    def __init__(self, kernel_size: Optional[int] = 9):
        # Set kernel size and compute sigma
        self.kernel_size = kernel_size
        self.sigma = 0.3 * (kernel_size / 2 - 1) + 0.8

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]) \
            -> (torch.Tensor, torch.Tensor):
        # Extract projections and joints heatmaps from sample
        (projections, heatmaps) = sample
        # Apply LCN to each projection
        projections = np.array(
            [local_contrast_normalization(x, self.kernel_size, self.sigma) for x in projections]
        )

        # Re-build tuple and return it
        return projections, heatmaps


class MSRAPreprocess:
    """Class to load and pre-process the MSRA dataset; this dataset is structured as follows:
    Pi
    |- Gj
        |- joint.txt
        |- k_depth.bin
    where Pi is the i-th subject (9 in total), Gj is the j-th gesture (17 per subject), while k_depth.bin is the k-th
    binary file (500 per gesture); joint.txt is the file containing the 3 coordinates of each 21 hand joint, for every
    500 binary file."""

    def __init__(self, root_folder: str):
        # Create a list that will contain the samples (bin file + labels), divided by subject
        self.subjects = []
        # List of 21 joints
        joint_list = ['wrist', 'index_mcp', 'index_pip', 'index_dip', 'index_tip', 'middle_mcp', 'middle_pip',
                      'middle_dip', 'middle_tip', 'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip', 'little_mcp',
                      'little_pip', 'little_dip', 'little_tip', 'thumb_mcp', 'thumb_pip', 'thumb_dip', 'thumb_tip']

        # Read the subjects folders
        subjects = sorted(glob.glob(os.path.join(root_folder, 'P*')))
        # For each subject, read the hand poses folders
        for subject in tqdm(subjects, leave=False):
            # Create a flat list that will contain the samples (bin file + labels) for the current subject
            samples = []

            hand_poses = sorted(glob.glob(os.path.join(subject, '*', '')))
            # For each hand pose, save the path to each binary file
            for hand_pose in tqdm(hand_poses, leave=False):
                bin_files = sorted(glob.glob(os.path.join(hand_pose, '*.bin')))
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
                samples.extend(list(zip(bin_files, joints)))
            self.subjects.append(samples)

        # Save additional preprocessing steps
        self.steps = []

    def __len__(self):
        lengths = map(len, self.subjects)
        return sum(lengths)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, Union[Dict[str, Tuple[float, float, float]], np.ndarray]]:
        # Find index of nested list
        lengths = list(map(len, self.subjects))
        cur_sub = 0
        cur_item = item
        while True:
            if cur_item >= lengths[cur_sub]:
                cur_item -= lengths[cur_sub]
                cur_sub += 1
            else:
                break
        (bin_file, joints) = self.subjects[cur_sub][cur_item]
        # Open and read the binary file to produce the actual depth image
        sample = (read_bin_to_depth(bin_file), joints)
        # Apply additional steps
        try:
            for step in self.steps:
                sample = step(sample)
        except InvalidDataError:
            print(f'InvalidDataError: removed sample ({cur_sub}, {cur_item})')
            self.subjects[cur_sub].pop(cur_item)  # get rid of invalid sample
            return self[item]  # call __getitem__ recursively
        return sample

    def add_step(self, step: Union[ToPointCloud, AlignPointCloud, ProjectToOBBPlanes, LocalContrastNormalization]):
        """Method to add preprocessing steps; the order must be the following:
            1. ToPointCloud
            2. AlignPointCloud
            3. ProjectToOBBPlanes
            4. LocalContrastNormalization.

            :param step: the preprocessing step."""

        if len(self.steps) == 0 and not isinstance(step, ToPointCloud):
            print('The first preprocessing step must be ToPointCloud.')
        elif len(self.steps) == 1 and not isinstance(step, AlignPointCloud):
            print('The second preprocessing step must be AlignPointCloud.')
        elif len(self.steps) == 2 and not isinstance(step, ProjectToOBBPlanes):
            print('The third preprocessing step must be ProjectToOBBPlanes.')
        elif len(self.steps) == 3 and not isinstance(step, LocalContrastNormalization):
            print('The forth preprocessing step must be LocalContrastNormalization.')
        elif len(self.steps) >= 4:
            print('Preprocessing steps are already complete.')
        else:
            self.steps.append(step)

    def write_to_disk(self, folder: str):
        """Method that writes the preprocessed dataset to disk in order to speed up training. The original complex
        file structure will be flattened.

            :param folder: path of the folder where the preprocessed data will be saved."""

        if len(self.steps) == 4:  # save only if all preprocessing steps have been added
            # Create directory for training and test samples
            train_dir = os.path.join(folder, 'Training')
            test_dir = os.path.join(folder, 'Testing')
            Path(train_dir).mkdir(parents=True, exist_ok=True)
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            # Reserve last subject for test
            train_test_sep = sum(map(len, self.subjects[:-1]))
            # Iterate over all samples
            for i in tqdm(range(len(self)), leave=False):
                # Take i-th sample
                sample = self[i]
                # Save in .npz
                if i < train_test_sep:  # train set
                    np.savez_compressed(os.path.join(train_dir, f'{i:06d}.npz'),
                                        x=sample[0].astype('float32'),
                                        y=sample[1].astype('float32'))
                else:  # test set
                    np.savez_compressed(os.path.join(test_dir, f'{i:06d}.npz'),
                                        x=sample[0].astype('float32'),
                                        y=sample[1].astype('float32'))
        else:
            print('You must add all the preprocessing step before saving data to disk.')
