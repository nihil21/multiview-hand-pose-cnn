from sys import float_info
import struct
import cv2
import numpy as np
from sklearn.decomposition import PCA
from typing import Union, Dict, Tuple, Optional, List, Callable


def read_bin_to_depth(filename: str) -> np.ndarray:
    """Function that reads depth information from a binary file and reconstructs a NumPy's array.
        :param filename: the path to the binary file.

        :returns the depth image corresponding to the binary file."""

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
    depth_image = np.zeros(img_shape)
    # Change ROI according to values read from file
    depth_image[top_offset:bottom_offset, left_offset:right_offset] = depths_roi
    return depth_image


def depth_to_point_cloud(depth_image: np.ndarray,
                         camera_parameters: Dict[str, Union[Tuple[int, int], Tuple[float, float]]]) -> np.ndarray:
    """Function that, given:
        - a depth image (2D array in which each cell contains the depth in mm of a point);
        - camera's parameters (principal point and focal length);
    returns a point cloud, i.e. a 3D array containing, for each point, the 3 (x, y, z) coordinates.

        :param depth_image: a 2D NumPy's array representing a depth_image;
        :param camera_parameters: a dictionary containing the principal point and the focal length of the camera.

        :returns the point cloud corresponding to the depth image."""

    rows, cols = depth_image.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    z = -depth_image
    x = -z * (c - camera_parameters['principal_point'][0]) / camera_parameters['focal_length'][0]
    y = z * (r - camera_parameters['principal_point'][1]) / camera_parameters['focal_length'][1]
    # Stack x, y and z
    point_cloud = np.dstack((x, y, z))
    # Remove points in (0, 0, 0)
    point_cloud = point_cloud[~np.all(point_cloud == 0, axis=2)]  # it will flatten the first two dimensions

    return point_cloud


def align_point_cloud(point_cloud: np.ndarray, joints: Optional[Dict[str, Tuple[float, float, float]]] = None) \
        -> (np.ndarray, np.ndarray, Optional[np.ndarray]):
    """Function that:
        - performs Principal Component Analysis (PCA) on a point cloud;
        - computes the Oriented Bounding Box (OBB) of the point cloud;
        - aligns the OBB and the point cloud with the Cartesian axes;
        - returns the aligned point cloud and the coordinates of the bounding box.

        :param point_cloud: NumPy's array representing a point cloud;
        :param joints: a dictionary containing, for each joint, a tuple of float representing the 3D coordinates
                       of the joint.

        :returns the new coordinates of the point cloud relative to the centre of the OBB;
        :returns the coordinates of the 8 OBB vertices w.r.t. the OBB reference system;
        :returns the index of the OBB vertex common to the three projection planes;
        :returns the joints dictionary with the updated coordinates."""

    # Perform PCA and obtain rotation matrix
    pca = PCA(n_components=3)
    pca.fit(point_cloud)
    rot_mtx = -pca.components_
    # Get center of point cloud
    center = point_cloud.mean(axis=0)
    # Change reference system
    point_cloud = (point_cloud - center).dot(rot_mtx)
    # Find OBB
    x_min, x_max = point_cloud[:, 0].min(), point_cloud[:, 0].max()
    y_min, y_max = point_cloud[:, 1].min(), point_cloud[:, 1].max()
    z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
    obb = np.array([[x_min, y_min, z_min],
                    [x_min, y_min, z_max],
                    [x_min, y_max, z_min],
                    [x_min, y_max, z_max],
                    [x_max, y_min, z_min],
                    [x_max, y_min, z_max],
                    [x_max, y_max, z_min],
                    [x_max, y_max, z_max]])

    # Roto-translate joints
    jc = None
    if joints is not None:
        jc = np.array([c for c in joints.values()])
        jc = (jc - center).dot(rot_mtx)

    # Return new point cloud and coordinates of box points
    return point_cloud, obb, jc


def project_to_obb_planes(point_cloud: np.ndarray, vertex: np.ndarray) -> (List[np.ndarray], List[np.ndarray]):
    """Function that, given an aligned point cloud and the coordinates of one OBB vertex, returns the three
    projections of the point cloud on the three xy, yz, zx planes sharing the vertex.

        :param point_cloud: NumPy's array representing a point cloud;
        :param vertex: 3D coordinates of the OBB vertex common to the three projection planes.

        :returns the three projections of the point cloud;
        :returns the projections' boundaries, useful to generate heatmaps."""

    # Create list of index configurations for each projection xy, yz, zx
    proj_conf = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    # Define lambda that, given a scalar and an array, returns the index of the array element nearest to the value.
    find_nearest: Callable[[float, np.ndarray], int] = lambda val, arr: (np.abs(val - arr)).argmin()

    proj_grids = []
    boundaries = []
    for (x_p, y_p, z_p) in proj_conf:
        # Set x_arr, y_arr, z_arr and plane according to current configuration
        x_arr = point_cloud[:, x_p]
        y_arr = point_cloud[:, y_p]
        z_arr = point_cloud[:, z_p]
        plane = vertex[z_p]
        # Compute distance from plane and normalize
        d_arr = np.abs(z_arr - plane)
        norm_factor = d_arr.max() - d_arr.min()
        d_arr = (d_arr - d_arr.min()) / norm_factor if norm_factor != 0 else d_arr - d_arr.min()

        # Create linear spaces for x and y
        x_space = np.arange(start=x_arr.min(), stop=x_arr.max() + 2, step=2)  # resolution of 2 mm
        y_space = np.arange(start=y_arr.min(), stop=y_arr.max() + 2, step=2)  # resolution of 2 mm
        # Create projection pixel grid
        proj_grid = np.ones((len(x_space), len(y_space)))
        # For each point in the cloud, find the best pixel coordinates
        for x, y, d in zip(x_arr, y_arr, d_arr):
            i = find_nearest(x, x_space)
            j = find_nearest(y, y_space)
            proj_grid[i, j] = min(d, proj_grid[i, j])
        # Normalize
        proj_grid = cv2.normalize(proj_grid, dst=None, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX)
        proj_grids.append(proj_grid)
        boundaries.append((x_arr.min(), x_arr.max(), y_arr.min(), y_arr.max()))
    return proj_grids, boundaries


def generate_heatmaps(joints_coords: np.ndarray, boundaries: List[np.ndarray]) -> List[np.ndarray]:
    """Function that, given an aligned point cloud of joint coordinates and three projection boundaries, returns the
    three (18x18) heatmaps of the joints coordinates obtained by projecting them onto the planes.
        :param joints_coords: the 3D coordinates of a joint as a NumPy's array with shape (1, 3);
        :param boundaries: the boundaries of the plane on which the joint's location will be projected.

        :returns the three (18x18) heatmaps."""

    # Create list of index configurations for each projection xy, yz, zx
    proj_conf = [(0, 1), (1, 2), (2, 0)]

    # Define lambda that convolves a Gaussian kernel onto a grid
    def gaussian(c: np.ndarray, r: np.ndarray, mu_x: float, mu_y: float, sigma: Optional[float] = 5):
        log_res = -((c - mu_x)**2 + (r - mu_y)**2) / (2 * sigma**2)
        # Clip to avoid underflow and overflow
        log_res = np.clip(log_res, a_min=np.log(float_info.min), a_max=np.log(float_info.max))
        return np.exp(log_res)

    heatmaps = []
    for (x_p, y_p), (x_min, x_max, y_min, y_max) in zip(proj_conf, boundaries):
        # Set x_arr, y_arr, z_arr and plane according to current configuration
        x_j = joints_coords[x_p]
        y_j = joints_coords[y_p]
        # Create linear spaces for x and y (resolution of 18)
        x_space = np.linspace(x_min, x_max, 18)
        y_space = np.linspace(y_min, y_max, 18)
        rows, cols = np.meshgrid(y_space, x_space)
        # Convolve Gaussian kernel
        heatmap = gaussian(cols, rows, x_j, y_j)
        heatmaps.append(heatmap)
    return heatmaps


def local_contrast_normalization(x: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    """Function that applies Local Contrast Normalization on a given image.
        :param x: input image as a NumPy's array;
        :param kernel: size of the LCN kernel;
        :param sigma: standard deviation of the LCN.

        :returns the normalized image."""

    # Apply subtractive normalization
    v = cv2.GaussianBlur(x, (kernel, kernel), sigma)
    num = x - v
    # Apply divisive normalization
    sigma = cv2.GaussianBlur(num**2, (kernel, kernel), sigma)
    c = sigma.mean()
    den = np.maximum(sigma, c)
    # Normalize image
    x = num / den
    return cv2.normalize(x, dst=None, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX)
