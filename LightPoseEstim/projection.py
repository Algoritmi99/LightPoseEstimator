import functools
import logging

import cv2
import numpy as np
import torch
import trimesh

from .pose import Pose

logger = logging.getLogger(__name__)

_OCCLUSION_CHUNK_ELEMENTS = 1 << 24  # rays x triangles tested per step
_FOLD_SEARCH_MAX_RADIUS = 10.0  # ~84 degrees off-axis
_FOLD_SEARCH_STEPS = 100_001


def as_dist_coeffs(dist_coeffs: np.ndarray | None) -> np.ndarray:
    if dist_coeffs is None:
        return np.zeros((5, 1), dtype=np.float32)
    return np.asarray(dist_coeffs, dtype=np.float32).reshape(-1, 1)


@functools.lru_cache(maxsize=32)
def _max_valid_radius(coeffs: tuple[float, ...]) -> float:
    k = np.zeros(8)
    k[: min(len(coeffs), 8)] = coeffs[:8]
    k1, k2, _, _, k3, k4, k5, k6 = k
    r = np.linspace(0.0, _FOLD_SEARCH_MAX_RADIUS, _FOLD_SEARCH_STEPS)
    r2 = r * r
    distorted = r * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
    turning = np.flatnonzero(np.diff(distorted) <= 0)
    return float(r[turning[0]]) if turning.size else np.inf


def max_valid_radius(dist_coeffs: np.ndarray | None) -> float:
    """
    Largest undistorted radius sqrt(x^2 + y^2) / z at which the radial distortion model
    is still monotonic. Past it, cv2.projectPoints folds points back into the image.
    """
    if dist_coeffs is None:
        return np.inf
    coeffs = tuple(float(c) for c in np.asarray(dist_coeffs).reshape(-1))
    return _max_valid_radius(coeffs)


def ray_rotation(translation: np.ndarray) -> np.ndarray:
    """Smallest rotation turning the camera's optical axis (+z) onto the ray to ``translation``."""
    ray = np.asarray(translation, dtype=np.float64).reshape(3)
    ray = ray / np.linalg.norm(ray)
    axis = np.cross([0.0, 0.0, 1.0], ray)
    sin_angle = np.linalg.norm(axis)
    if sin_angle < 1e-12:
        return np.eye(3)
    rotvec = axis / sin_angle * np.arctan2(sin_angle, ray[2])
    return cv2.Rodrigues(rotvec)[0]


def allocentric_rotation(pose: Pose) -> np.ndarray:
    """
    Object rotation as seen along the viewing ray (what a crop shows), rather than
    relative to the optical axis. Inverse: ``ray_rotation(t) @ allocentric``.
    """
    return ray_rotation(pose.get_tvec()).T @ pose.quaternion.to_rotation_matrix()


def camera_frame_points(points: np.ndarray, pose: Pose) -> np.ndarray:
    rot_matrix, _ = cv2.Rodrigues(pose.get_rvec())
    return (rot_matrix @ np.asarray(points).T + pose.get_tvec().reshape(3, 1)).T


def projectable_mask(
    points: np.ndarray,
    pose: Pose,
    dist_coeffs: np.ndarray | None = None,
    depth_eps: float = 1e-6,
) -> np.ndarray:
    """Points in front of the camera and inside the region where the distortion model is valid."""
    camera_points = camera_frame_points(points, pose)
    depth = camera_points[:, 2]
    in_front = depth > depth_eps
    radius = np.full(len(camera_points), np.inf)
    radius[in_front] = np.hypot(camera_points[in_front, 0], camera_points[in_front, 1]) / depth[in_front]
    return in_front & (radius <= max_valid_radius(dist_coeffs))


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


def face_normals_at_points(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    try:
        _, _, face_ids = mesh.nearest.on_surface(points)
        return mesh.face_normals[face_ids]
    except BaseException:
        try:
            _, _, face_ids = trimesh.proximity.closest_point(mesh, points)
            return mesh.face_normals[face_ids]
        except BaseException:
            logger.warning(
                "Surface normal lookup failed; using vertex normals for %d points.",
                points.shape[0],
            )
            normals = np.empty((points.shape[0], 3), dtype=np.float64)
            chunk = 8
            vertices = mesh.vertices.astype(np.float64)
            vertex_normals = mesh.vertex_normals
            for i in range(0, points.shape[0], chunk):
                batch = points[i : i + chunk]
                dist = np.linalg.norm(vertices[None, :, :] - batch[:, None, :], axis=2)
                normals[i : i + chunk] = vertex_normals[np.argmin(dist, axis=1)]
            return normals


class MeshOccluder:
    """
    Exact ray-occlusion test against every triangle of a mesh.

    All rays of one query start at the same camera position, so the Moller-Trumbore
    ray/triangle test reduces to three matrix products of the ray directions with
    per-triangle vectors. That makes testing the full mesh fast enough without a
    ray-tracing backend, on the GPU when one is available.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        device: torch.device | None = None,
        tolerance: float | None = None,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        triangles = torch.from_numpy(np.array(mesh.triangles, dtype=np.float32)).to(self.device)
        self.v0 = triangles[:, 0]
        self.e1 = triangles[:, 1] - triangles[:, 0]
        self.e2 = triangles[:, 2] - triangles[:, 0]
        self.normal = torch.linalg.cross(self.e2, self.e1)
        self.tolerance = (
            tolerance if tolerance is not None else max(float(mesh.scale) * 1e-3, 1e-5)
        )

    @torch.no_grad()
    def visible(self, camera_position: np.ndarray, points: np.ndarray) -> np.ndarray:
        """True for points with no mesh surface between them and the camera."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if points.shape[0] == 0:
            return np.zeros(0, dtype=bool)

        origin = torch.as_tensor(
            np.asarray(camera_position, dtype=np.float64).reshape(3),
            dtype=torch.float32,
            device=self.device,
        )
        # Ray i runs from the camera (t=0) to point i (t=1).
        rays = torch.as_tensor(points, dtype=torch.float32, device=self.device) - origin
        t_max = 1.0 - self.tolerance / rays.norm(dim=1).clamp_min(1e-12)
        occluded = torch.zeros(rays.shape[0], dtype=torch.bool, device=self.device)

        n_triangles = self.v0.shape[0]
        ray_chunk = max(1, min(rays.shape[0], _OCCLUSION_CHUNK_ELEMENTS // max(n_triangles, 1)))
        tri_chunk = max(1, _OCCLUSION_CHUNK_ELEMENTS // ray_chunk)
        for t0 in range(0, n_triangles, tri_chunk):
            to_origin = origin - self.v0[t0:t0 + tri_chunk]
            e1 = self.e1[t0:t0 + tri_chunk]
            e2 = self.e2[t0:t0 + tri_chunk]
            u_vec = torch.linalg.cross(e2, to_origin)
            v_vec = torch.linalg.cross(to_origin, e1)
            t_num = (e2 * v_vec).sum(dim=1)
            normal = self.normal[t0:t0 + tri_chunk]
            for r0 in range(0, rays.shape[0], ray_chunk):
                d = rays[r0:r0 + ray_chunk]
                inv_det = 1.0 / (d @ normal.T)
                u = (d @ u_vec.T) * inv_det
                v = (d @ v_vec.T) * inv_det
                t = t_num * inv_det
                # Parallel rays give inf/nan and fail every comparison.
                hit = (
                    (u >= -1e-6)
                    & (v >= -1e-6)
                    & (u + v <= 1.0 + 1e-6)
                    & (t > 0.0)
                    & (t < t_max[r0:r0 + ray_chunk, None])
                )
                occluded[r0:r0 + ray_chunk] |= hit.any(dim=1)
        return (~occluded).cpu().numpy()


def keypoint_visibility(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    pose: Pose,
    occluder: MeshOccluder | None = None,
    normals: np.ndarray | None = None,
    depth_eps: float = 1e-6,
    facing_eps: float = 0.0,
) -> np.ndarray:
    """
    Per-keypoint visibility from the pose camera: in front, facing, unoccluded.
    Pass ``occluder`` and ``normals`` when calling repeatedly for the same mesh and points.
    """
    points = np.asarray(points, dtype=np.float64)
    visible = np.zeros(points.shape[0], dtype=bool)

    in_front = camera_frame_points(points, pose)[:, 2] > depth_eps
    if not np.any(in_front):
        return visible

    cam_origin = camera_origin_in_object_frame(pose)
    if normals is None:
        normals = face_normals_at_points(mesh, points)
    facing = np.einsum("ij,ij->i", normals, cam_origin - points) > facing_eps
    candidates = in_front & facing
    if not np.any(candidates):
        return visible

    if occluder is None:
        occluder = MeshOccluder(mesh)
    visible[candidates] = occluder.visible(cam_origin, points[candidates])
    return visible
