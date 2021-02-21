import struct
from math import floor
import numpy as np
import open3d as o3d
from typing import Union, Dict, Tuple, Optional, Callable


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
    # The file contains only depth information about the ROI (where the hand is),
    # so an array with the original shape must be created
    depth_image = np.ones(img_shape) * 0
    # Change ROI according to values read from file
    depth_image[top_offset:bottom_offset, left_offset:right_offset] = depths_roi
    return depth_image


def depth_to_point_cloud(depth_image: np.ndarray,
                         camera_parameters: Dict[str, Union[Tuple[int, int], Tuple[float, float]]]) -> np.ndarray:
    """Function that, given:
        - a depth image (2D array in which each cell contains the depth in mm of a point);
        - camera's parameters (principal point and focal length);
    returns a point cloud, i.e. a 3D array containing, for each point, the 3 (x, y, z) coordinates."""

    rows, cols = depth_image.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    z = depth_image
    x = z * (c - camera_parameters['principal_point'][0]) / camera_parameters['focal_length'][0]
    y = z * (r - camera_parameters['principal_point'][1]) / camera_parameters['focal_length'][1]
    # Stack x, y and z
    point_cloud = np.dstack((x, y, z))
    # Remove points in (0, 0, 0)
    point_cloud = point_cloud[(point_cloud != 0).all(axis=2)]  # it will flatten the first two dimensions

    return point_cloud


def align_point_cloud(point_cloud: np.ndarray, joints: Optional[Dict[str, Tuple[float, float, float]]] = None) \
        -> (np.ndarray, np.ndarray, Dict[str, np.ndarray]):
    """Function that:
        - performs Principal Component Analysis (PCA) on a point cloud;
        - computes the Oriented Bounding Box (OBB) of the point cloud;
        - aligns the OBB and the point cloud with the Cartesian axes;
        - returns the aligned point cloud and the coordinates of the bounding box."""

    # Create PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # Computed OBB from point cloud
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(points=pcd.points)  # it automatically performs PCA
    # Get OBB center and rotation matrix
    center = obb.get_center()
    rot_mtx = np.linalg.inv(obb.R)
    pcd.rotate(R=rot_mtx, center=center).translate(-center)
    obb.rotate(R=rot_mtx, center=center).translate(-center)
    if joints is not None:
        coords = np.array([c for c in joints.values()])
        jcd = o3d.geometry.PointCloud()
        jcd.points = o3d.utility.Vector3dVector(coords)
        jcd.rotate(R=rot_mtx, center=center).translate(-center)
        coords = np.asarray(jcd.points)
        joints = {j: coords[i] for i, j in enumerate(joints.keys())}
    # Return new point cloud and coordinates of box points
    return np.asarray(pcd.points), np.asarray(obb.get_box_points()), joints


def project_to_obb_planes(point_cloud: np.ndarray, vertex: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """Function that, given an aligned point cloud and the coordinates of one OBB vertex, returns the three
    projections of the point cloud on the three xy, yz, zx planes sharing the vertex."""

    # Create list of index configurations for each projection xy, yz, zx
    proj_conf = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    # Define lambda that, given a scalar and an array, returns the index of the array element nearest to the value
    find_nearest: Callable[[float, np.ndarray], int] = lambda val, arr: (np.abs(val - arr)).argmin()

    proj_grids = []
    for (x_i, y_i, z_i) in proj_conf:
        # Set x_arr, y_arr, z_arr and plane according to current configuration
        x_arr = point_cloud[:, x_i]
        y_arr = point_cloud[:, y_i]
        z_arr = point_cloud[:, z_i]
        plane = vertex[z_i]
        # Compute distance from plane and normalize
        d_arr = np.abs(z_arr - plane)
        d_arr = (d_arr - d_arr.min()) / (d_arr.max() - d_arr.min())
        # Create linear spaces for x and y
        x_space = np.linspace(x_arr.min(), x_arr.max(), floor(x_arr.max() - x_arr.min()))
        y_space = np.linspace(y_arr.min(), y_arr.max(), floor(y_arr.max() - y_arr.min()))
        # Create projection pixel grid
        proj_grid = np.ones((len(x_space), len(y_space)), dtype=np.float32)
        # For each point in the cloud, find the best pixel coordinates
        for x, y, d in zip(x_arr, y_arr, d_arr):
            i = find_nearest(x, x_space)
            j = find_nearest(y, y_space)
            if d < proj_grid[i, j]:
                proj_grid[i, j] = d
        proj_grids.append(proj_grid)
    return tuple(proj_grids)
