
import torch
import numpy as np

def get_delta_x_meters(P_source, P_dest):
    # Extract the translation components from the projection matrices
    t_source = P_source[0, 3] / P_source[0, 0]  # Translation component for camera 2
    t_dest = P_dest[0, 3] / P_dest[0, 0]  # Translation component for camera 3

    # Calculate the horizontal shift in pixels between camera 2 and camera 3
    delta_x = t_dest - t_source
    return delta_x

def get_delta_x_pixels(P_source, P_dest, depth):
    delta_x = (P_dest[0, 3] - P_source[0, 3])  / depth
    return delta_x

def pts2Dto3D(points2D, P, depths):
    return (depths * (points2D - P[:2, 2:3]).T - P[:2, 3:]).T / P[0, 0]

def pts2Dto3D_general(points, P):
    """
    Args:
        points (torch.Tensor): points in 2D images, [N, 3], \
            3 corresponds with x, y in the image and depth.
        view (np.ndarray): camera instrinsic, [3, 3]

    Returns:
        torch.Tensor: points in 3D space. [N, 3], \
            3 corresponds with x, y, z in 3D space.
    """
    assert P.shape[0] <= 4
    assert P.shape[1] <= 4
    assert points.shape[1] == 3

    points2D = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

    viewpad = torch.eye(4, dtype=points2D.dtype, device=points2D.device)
    viewpad[:P.shape[0], :P.shape[1]] = points2D.new_tensor(P)
    inv_viewpad = torch.inverse(viewpad).transpose(0, 1)

    # Do operation in homogenous coordinates.
    nbr_points = unnorm_points2D.shape[0]
    homo_points2D = torch.cat(
        [unnorm_points2D,
         points2D.new_ones((nbr_points, 1))], dim=1)
    points3D = torch.mm(homo_points2D, inv_viewpad)[:, :3]

    return points3D


def transform_alpha(source_alpha, P_source, P_dest, depth):
    """
    Transforms the alpha angle of an object from camera 2 to camera 3.

    Parameters:
    - source_alpha: The alpha angle in camera source (in radians)
    - source_cam: Projection matrix for camera source (4x4 numpy array)
    - dest_cam: Projection matrix for camera dest (4x4 numpy array)
    - depth: Depth of the object in the z-axis (in meters, relative to camera source)

    Returns:
    - alpha_dest: Transformed alpha angle in camera dest (in radians)
    """
    # Extract the translation components from the projection matrices
    t_source = P_source[0, 3] / P_source[0, 0]  # Translation component for camera 2
    t_dest = P_dest[0, 3] / P_dest[0, 0]  # Translation component for camera 3

    # Calculate the change in viewing angle (delta alpha) due to the shift
    delta_alpha = np.arctan((t_dest - t_source) / depth)

    # Adjust the alpha angle with respect to camera 3
    alpha_dest = source_alpha
    cond_in_range = np.vectorize(lambda x: (x > -np.pi) & (x < np.pi))
    alpha_dest[cond_in_range(alpha_dest)] = (alpha_dest + delta_alpha)[cond_in_range(alpha_dest)]
    return alpha_dest