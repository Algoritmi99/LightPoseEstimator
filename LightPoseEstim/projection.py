import cv2
import numpy as np
import trimesh

from LightPoseEstim.pose import Pose


def as_dist_coeffs(dist_coeffs: np.ndarray | None) -> np.ndarray:
    if dist_coeffs is None:
        return np.zeros((5, 1), dtype=np.float32)
    return np.asarray(dist_coeffs, dtype=np.float32).reshape(-1, 1)


def mesh_bbox_corners(mesh: trimesh.Trimesh) -> np.ndarray:
    xmin, ymin, zmin = mesh.bounds[0]
    xmax, ymax, zmax = mesh.bounds[1]
    return np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
    ], dtype=np.float32)


def project_points(
    points: np.ndarray,
    pose: Pose,
    camera_intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Project 3D points in the object frame to distorted image coordinates."""
    image_points, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float32),
        pose.get_rvec(),
        pose.get_tvec(),
        camera_intrinsics,
        as_dist_coeffs(dist_coeffs),
    )
    return image_points.reshape(-1, 2)


def project_mesh_vertices(
    mesh: trimesh.Trimesh,
    pose: Pose,
    camera_intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    return project_points(mesh.vertices, pose, camera_intrinsics, dist_coeffs)
