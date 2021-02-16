import struct
import numpy as np
from typing import Union, Dict, Tuple


def read_bin_to_depth(filename: str) -> np.ndarray:
    """Function that reads depth information from a binary file and reconstructs a NumPy's array"""

    with open(filename, 'rb') as f:
        # Read first 24 bytes as 6 unsigned integers
        (img_width, img_height, left_offset, top_offset, right_offset, bottom_offset) = struct.unpack('<6I', f.read(24))
        img_shape = (img_height, img_width)
        roi_shape = (bottom_offset - top_offset, right_offset - left_offset)
        tot_val = roi_shape[0] * roi_shape[1]
        # Read the (4n) subsequent bytes as floats (n specified by the offsets) and convert the resulting
        # list into a NumPy's array
        depths_roi = struct.unpack(f'<{tot_val}f', f.read(tot_val * 4))
        depths_roi = np.array(depths_roi).reshape(roi_shape)
    # Create final array with the original shape
    depth_image = np.ones(img_shape) * 0
    # Change ROI according to values read from file
    depth_image[top_offset:bottom_offset, left_offset:right_offset] = depths_roi
    return depth_image


def depth_to_point_cloud(depth_image: np.ndarray,
                         camera_parameters: Dict[str, Union[Tuple[int, int], float]]) -> np.ndarray:
    """Function that, given:
        - a depth image (2D array in which each cell contains the depth in mm of a point);
        - camera's parameters (principal point and focal length);
    returns a point cloud, i.e. a 3D array containing, for each point, the 3 (x, y, z) coordinates."""

    rows, cols = depth_image.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    z = depth_image
    x = z * (c - camera_parameters['principal_point'][0]) / camera_parameters['focal_length']
    y = z * (r - camera_parameters['principal_point'][1]) / camera_parameters['focal_length']
    return np.dstack((x, y, z))
