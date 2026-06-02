import logging

import cv2
import numpy as np
import trimesh

from .pose import Pose

logger = logging.getLogger(__name__)


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


def camera_origin_in_object_frame(pose: Pose) -> np.ndarray:
    rot_matrix, _ = cv2.Rodrigues(pose.get_rvec())
    return (-rot_matrix.T @ pose.get_tvec()).reshape(3)


def _face_normals_at_points(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    try:
        _, _, face_ids = mesh.nearest.on_surface(points)
        return mesh.face_normals[face_ids]
    except BaseException:
        dist = np.linalg.norm(
            mesh.vertices[None, :, :].astype(np.float64) - points[:, None, :],
            axis=2,
        )
        return mesh.vertex_normals[np.argmin(dist, axis=1)]


def _raycast_visible_bruteforce(
    mesh: trimesh.Trimesh,
    camera_position: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    from trimesh import intersections
    from trimesh import triangles as triangles_mod
    from trimesh.constants import tol as tr_tol

    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    cam = np.asarray(camera_position, dtype=np.float64).reshape(3)
    origins = np.repeat(cam.reshape(1, 3), n, axis=0)
    vec = points - origins
    target_dist = np.linalg.norm(vec, axis=1)
    dirs = vec / np.maximum(target_dist[:, None], 1e-12)
    tol = max(float(mesh.scale) * 1e-3, 1e-5)

    triangles = mesh.triangles
    face_normals = mesh.face_normals
    visible = np.zeros(n, dtype=bool)

    for i in range(n):
        line_origins = np.tile(origins[i], (len(triangles), 1))
        line_directions = np.tile(dirs[i], (len(triangles), 1))
        location, valid = intersections.planes_lines(
            triangles[:, 0, :],
            face_normals,
            line_origins,
            line_directions,
        )
        if not valid.any():
            continue

        bary = triangles_mod.points_to_barycentric(triangles[valid], location)
        hit = np.logical_and(
            (bary > -tr_tol.zero).all(axis=1),
            (bary < (1 + tr_tol.zero)).all(axis=1),
        )
        if not hit.any():
            continue

        hit_dist = np.linalg.norm(location[hit] - origins[i], axis=1)
        forward = hit_dist > 1e-6
        if not forward.any():
            continue

        nearest = hit_dist[forward].min()
        visible[i] = abs(nearest - target_dist[i]) <= tol

    return visible


def raycast_visible(
    mesh: trimesh.Trimesh,
    camera_position: np.ndarray,
    points: np.ndarray,
    allow_fallback: bool = False,
) -> np.ndarray | None:
    """Return whether each point is the first surface hit along rays from the camera."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    origins = np.repeat(np.asarray(camera_position, dtype=np.float64).reshape(1, 3), n, axis=0)
    vec = points - origins
    dist = np.linalg.norm(vec, axis=1)
    dirs = vec / np.maximum(dist[:, None], 1e-12)

    try:
        loc, idx_ray, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=dirs,
            multiple_hits=False,
        )
        visible = np.zeros(n, dtype=bool)
        if idx_ray.shape[0] == 0:
            return visible

        hit_dist = np.linalg.norm(loc - origins[idx_ray], axis=1)
        tol = max(float(mesh.scale) * 1e-3, 1e-5)
        visible[idx_ray] = np.abs(hit_dist - dist[idx_ray]) <= tol
        return visible
    except BaseException:
        try:
            return _raycast_visible_bruteforce(mesh, camera_position, points)
        except BaseException:
            if allow_fallback:
                return np.ones(n, dtype=bool)
            return None


def keypoint_visibility(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    pose: Pose,
    depth_eps: float = 1e-6,
    facing_eps: float = 0.0,
) -> np.ndarray:
    """
    Per-keypoint visibility from the pose camera: in front, facing, unoccluded.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    visible = np.zeros(n, dtype=bool)

    rot_matrix, _ = cv2.Rodrigues(pose.get_rvec())
    in_front = (
        (rot_matrix @ points.T) + pose.get_tvec().reshape(3, 1)
    ).T[:, 2] > depth_eps
    if not np.any(in_front):
        return visible

    cam_origin = camera_origin_in_object_frame(pose)
    normals = _face_normals_at_points(mesh, points)
    facing = np.einsum("ij,ij->i", normals, cam_origin - points) > facing_eps
    candidates = in_front & facing
    if not np.any(candidates):
        return visible

    cand_idx = np.flatnonzero(candidates)
    unoccluded = raycast_visible(mesh, cam_origin, points[candidates], allow_fallback=False)
    if unoccluded is None:
        logger.warning("Mesh ray tracing unavailable; using depth and facing only.")
        visible[candidates] = True
    else:
        visible[cand_idx] = unoccluded

    return visible
