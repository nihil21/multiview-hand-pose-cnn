import os
import glob
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional, List, Callable, Any
from multiview_hand_pose_cnn.dataset.data_utils import read_bin_to_depth


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
