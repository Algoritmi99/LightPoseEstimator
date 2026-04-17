from __future__ import annotations

import logging
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional

import numpy as np
import trimesh


@dataclass
class CanonicalKeypoint:
    keypoint_id: int
    position: np.ndarray
    semantic_label: Optional[str] = None
    visibility_prior: Optional[float] = None
    stability_prior: Optional[float] = None


@dataclass
class CandidatePool:
    points: np.ndarray
    normals: np.ndarray
    neighbor_indices: np.ndarray
    local_curvature: np.ndarray


@dataclass
class CandidateScores:
    visibility: np.ndarray
    external_surface: np.ndarray
    distinctiveness: np.ndarray
    geometric_usefulness: np.ndarray
    stability: np.ndarray
    total: np.ndarray


class MeshKeypointSelector:
    """
    Select canonical 3D keypoints for a mesh.

    Pipeline:
    1) Sample a dense candidate pool on the CAD surface.
    2) Simulate many viewpoints and estimate per-candidate visibility/stability.
    3) Score candidates by visibility, distinctiveness, geometry, and stability.
    4) Filter weak candidates and select final K with spread-aware sampling.
    5) Return fixed canonical keypoints in object coordinates.
    """

    def __init__(
        self,
        candidate_count: int = 1200,
        final_keypoint_count: int = 24,
        num_viewpoints: int = 240,
        neighbor_count: int = 16,
        min_visibility: float = 0.08,
        min_external_visibility: float = 0.15,
        random_seed: Optional[int] = None,
    ) -> None:
        self.candidate_count = int(candidate_count)
        self.final_keypoint_count = int(final_keypoint_count)
        self.num_viewpoints = int(num_viewpoints)
        self.neighbor_count = int(neighbor_count)
        self.min_visibility = float(min_visibility)
        self.min_external_visibility = float(min_external_visibility)
        self.rng = np.random.default_rng(random_seed)

    def select_keypoints(
        self,
        mesh: trimesh.Trimesh,
        keypoint_count: Optional[int] = None,
        semantic_labels: Optional[list[str]] = None,
    ) -> list[CanonicalKeypoint]:
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("mesh must be a trimesh.Trimesh instance")

        if mesh.vertices.shape[0] < 4 or mesh.faces.shape[0] < 4:
            raise ValueError("mesh is too small to select robust keypoints")

        mesh = mesh.copy()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()

        k_final = int(keypoint_count or self.final_keypoint_count)
        if k_final <= 0:
            raise ValueError("keypoint_count must be > 0")

        candidates = self._sample_candidates(mesh, self.candidate_count)
        visibility_ratio, stability_ratio = self._simulate_coverage(mesh, candidates)
        external_surface_ratio = self._estimate_external_surface_visibility(mesh, candidates)
        distinctiveness = self._compute_distinctiveness(candidates)
        geometric_usefulness = self._compute_geometric_usefulness(candidates, mesh)

        total_score = (
            0.30 * visibility_ratio
            + 0.20 * external_surface_ratio
            + 0.20 * distinctiveness
            + 0.15 * geometric_usefulness
            + 0.15 * stability_ratio
        )
        scores = CandidateScores(
            visibility=visibility_ratio,
            external_surface=external_surface_ratio,
            distinctiveness=distinctiveness,
            geometric_usefulness=geometric_usefulness,
            stability=stability_ratio,
            total=self._normalize(total_score),
        )

        keep_mask = self._filter_candidates(scores)
        if keep_mask.sum() < k_final:
            # Fall back to best global candidates if filtering is too strict.
            keep_mask = np.zeros_like(keep_mask, dtype=bool)
            top_idx = np.argsort(scores.total)[-max(k_final * 2, k_final) :]
            keep_mask[top_idx] = True

        filtered_points = candidates.points[keep_mask]
        filtered_scores = scores.total[keep_mask]
        selected_local = self._farthest_point_select(
            filtered_points,
            filtered_scores,
            k=k_final,
            mesh_scale=max(float(mesh.scale), 1e-6),
        )

        selected_global = np.flatnonzero(keep_mask)[selected_local]
        selected_global = selected_global[np.argsort(scores.total[selected_global])[::-1]]

        output: list[CanonicalKeypoint] = []
        for out_id, idx in enumerate(selected_global.tolist()):
            label = semantic_labels[out_id] if semantic_labels and out_id < len(semantic_labels) else None
            output.append(
                CanonicalKeypoint(
                    keypoint_id=out_id,
                    position=candidates.points[idx].copy(),
                    semantic_label=label,
                    visibility_prior=float(scores.visibility[idx]),
                    stability_prior=float(scores.stability[idx]),
                )
            )
        return output

    def show_keypoints(
        self,
        mesh: trimesh.Trimesh,
        keypoints: list[CanonicalKeypoint],
        marker_radius: Optional[float] = None,
        show_ids: bool = False,
    ) -> trimesh.Scene:
        """
        Visualize the mesh and selected keypoints in a trimesh scene.

        Parameters
        ----------
        mesh:
            Input object mesh.
        keypoints:
            Selected keypoints to draw.
        marker_radius:
            Sphere marker radius. If None, uses a mesh-scale dependent value.
        show_ids:
            If True, adds tiny axis markers to make each point easier to inspect.

        Returns
        -------
        trimesh.Scene
            Scene containing the mesh and keypoint markers.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("mesh must be a trimesh.Trimesh instance")

        scene = trimesh.Scene()
        mesh_vis = mesh.copy()
        scene.add_geometry(mesh_vis, geom_name="mesh")

        scale = max(float(mesh.scale), 1e-6)
        radius = marker_radius if marker_radius is not None else 0.015 * scale
        radius = max(float(radius), 1e-6)

        for kp in keypoints:
            marker = trimesh.creation.uv_sphere(radius=radius, count=[12, 12])
            marker.visual.vertex_colors = np.array([230, 40, 40, 255], dtype=np.uint8)
            marker.apply_translation(np.asarray(kp.position, dtype=np.float64))
            scene.add_geometry(marker, geom_name=f"keypoint_{kp.keypoint_id}")

            if show_ids:
                axis = trimesh.creation.axis(origin_size=radius * 0.15, axis_length=radius * 1.7)
                axis.apply_translation(np.asarray(kp.position, dtype=np.float64))
                scene.add_geometry(axis, geom_name=f"keypoint_axis_{kp.keypoint_id}")

        scene.show()
        return scene

    def _sample_candidates(self, mesh: trimesh.Trimesh, count: int) -> CandidatePool:
        if count < 32:
            raise ValueError("candidate_count should be at least 32")

        sample_count = count
        points = np.empty((0, 3), dtype=np.float64)
        face_idx = np.empty((0,), dtype=np.int64)

        with self._quiet_trimesh_logs():
            try:
                points, face_idx = trimesh.sample.sample_surface_even(mesh, sample_count)
            except BaseException:
                points, face_idx = trimesh.sample.sample_surface(mesh, sample_count)

            # If even sampling cannot place enough separated points, top up with
            # regular surface sampling so downstream always sees the target count.
            missing = sample_count - int(points.shape[0])
            if missing > 0:
                extra_points, extra_faces = trimesh.sample.sample_surface(mesh, missing)
                points = np.vstack([points, extra_points])
                face_idx = np.concatenate([face_idx, extra_faces], axis=0)

        # Keep exactly requested count in a stable/randomized way.
        if points.shape[0] > sample_count:
            sel = self.rng.choice(points.shape[0], size=sample_count, replace=False)
            points = points[sel]
            face_idx = face_idx[sel]

        normals = mesh.face_normals[face_idx]
        normals = self._safe_normalize(normals)

        neighbor_indices = self._k_nearest_indices(points, k=min(self.neighbor_count, points.shape[0] - 1))
        local_curvature = self._local_curvature(points, neighbor_indices)

        return CandidatePool(
            points=points.astype(np.float64),
            normals=normals.astype(np.float64),
            neighbor_indices=neighbor_indices,
            local_curvature=local_curvature,
        )

    @staticmethod
    @contextmanager
    def _quiet_trimesh_logs():
        trimesh_logger = logging.getLogger("trimesh")
        old_level = trimesh_logger.level
        trimesh_logger.setLevel(logging.ERROR)
        try:
            yield
        finally:
            trimesh_logger.setLevel(old_level)

    def _simulate_coverage(
        self, mesh: trimesh.Trimesh, candidates: CandidatePool
    ) -> tuple[np.ndarray, np.ndarray]:
        points = candidates.points
        normals = candidates.normals
        n = points.shape[0]

        directions = self._fibonacci_sphere(self.num_viewpoints)
        centroid = mesh.centroid
        mesh_scale = max(float(mesh.scale), 1e-6)
        radius = 2.5 * mesh_scale

        visible_count = np.zeros(n, dtype=np.float64)
        stable_count = np.zeros(n, dtype=np.float64)
        valid_views = 0

        for view_dir in directions:
            distance_jitter = self.rng.uniform(1.4, 3.0)
            cam_pos = centroid + (radius * distance_jitter) * view_dir

            # Basic face orientation check (back-facing points are not visible).
            to_cam = cam_pos[None, :] - points
            to_cam_dir = self._safe_normalize(to_cam)
            facing = np.einsum("ij,ij->i", normals, to_cam_dir) > 0.02

            if not np.any(facing):
                continue

            ray_visible = self._raycast_visible(mesh, cam_pos, points)
            visible = facing & ray_visible

            # Simulated random crops/truncations in projected coordinates.
            in_crop = self._simulate_crop(points, centroid, view_dir)
            # Simulated partial occluder (sphere between camera and object center).
            occluded = self._simulate_occluder(cam_pos, centroid, points, mesh_scale)
            visible = visible & in_crop & (~occluded)

            visible_count += visible.astype(np.float64)
            valid_views += 1

            # Lighting perturbation for stability/identifiability.
            light_dir = self._safe_normalize(
                view_dir[None, :] + self.rng.normal(0.0, 0.35, size=(1, 3))
            )[0]
            normal_light = np.clip(np.einsum("ij,j->i", normals, light_dir), 0.0, 1.0)
            detection_prob = 0.25 + 0.75 * normal_light
            detected = visible & (self.rng.random(n) < detection_prob)
            stable_count += detected.astype(np.float64)

        if valid_views == 0:
            return np.zeros(n), np.zeros(n)

        visibility_ratio = visible_count / valid_views
        stability_ratio = stable_count / np.maximum(visible_count, 1.0)
        return self._normalize(visibility_ratio), self._normalize(stability_ratio)

    def _compute_distinctiveness(self, candidates: CandidatePool) -> np.ndarray:
        points = candidates.points
        normals = candidates.normals
        neigh_idx = candidates.neighbor_indices
        curv = candidates.local_curvature

        neigh_points = points[neigh_idx]  # [N, K, 3]
        local_vecs = neigh_points - points[:, None, :]
        local_dists = np.linalg.norm(local_vecs, axis=2)

        local_stats = np.stack(
            [
                np.mean(local_dists, axis=1),
                np.std(local_dists, axis=1),
                np.max(local_dists, axis=1),
                curv,
                normals[:, 0],
                normals[:, 1],
                normals[:, 2],
            ],
            axis=1,
        )
        descriptors = self._normalize_rows(local_stats)

        n = descriptors.shape[0]
        nearest = np.full(n, np.inf, dtype=np.float64)
        chunk = 256
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            a = descriptors[i:j]
            dmat = np.linalg.norm(a[:, None, :] - descriptors[None, :, :], axis=2)
            row_ids = np.arange(j - i)
            dmat[row_ids, i + row_ids] = np.inf
            nearest[i:j] = np.min(dmat, axis=1)

        return self._normalize(nearest)

    def _compute_geometric_usefulness(
        self, candidates: CandidatePool, mesh: trimesh.Trimesh
    ) -> np.ndarray:
        points = candidates.points
        curv = candidates.local_curvature

        centroid = mesh.centroid
        scale = max(float(mesh.scale), 1e-6)
        radial = np.linalg.norm(points - centroid[None, :], axis=1) / scale

        # Reward points away from extreme center clustering and with non-planar locality.
        geom = 0.55 * self._normalize(radial) + 0.45 * self._normalize(curv)
        return self._normalize(geom)

    def _filter_candidates(self, scores: CandidateScores) -> np.ndarray:
        vis_ok = scores.visibility >= self.min_visibility
        ext_ok = scores.external_surface >= self.min_external_visibility
        stab_ok = scores.stability >= np.quantile(scores.stability, 0.15)
        total_ok = scores.total >= np.quantile(scores.total, 0.35)
        return vis_ok & ext_ok & stab_ok & total_ok

    def _estimate_external_surface_visibility(
        self, mesh: trimesh.Trimesh, candidates: CandidatePool
    ) -> np.ndarray:
        """
        Estimate how likely each candidate belongs to externally visible surface.
        Uses strict ray visibility from outside views. If ray tracing backend is
        unavailable, falls back to an outward-normal heuristic.
        """
        points = candidates.points
        normals = candidates.normals
        n = points.shape[0]
        centroid = mesh.centroid
        mesh_scale = max(float(mesh.scale), 1e-6)

        # Conservative fallback: keep points whose normals point outward.
        outward_dir = self._safe_normalize(points - centroid[None, :])
        outward_score = (np.einsum("ij,ij->i", normals, outward_dir) > 0.0).astype(np.float64)

        probe_views = max(24, min(96, self.num_viewpoints // 2))
        directions = self._fibonacci_sphere(probe_views)
        radius = 3.0 * mesh_scale

        visible_count = np.zeros(n, dtype=np.float64)
        valid_views = 0
        ray_backend_ok = False

        for view_dir in directions:
            cam_pos = centroid + radius * view_dir
            to_cam = cam_pos[None, :] - points
            to_cam_dir = self._safe_normalize(to_cam)
            facing = np.einsum("ij,ij->i", normals, to_cam_dir) > 0.05
            if not np.any(facing):
                continue

            ray_visible = self._raycast_visible(mesh, cam_pos, points, allow_fallback=False)
            if ray_visible is None:
                # Ray backend unavailable in this environment.
                continue

            ray_backend_ok = True
            visible = facing & ray_visible
            visible_count += visible.astype(np.float64)
            valid_views += 1

        if ray_backend_ok and valid_views > 0:
            return self._normalize(visible_count / valid_views)
        return outward_score

    def _farthest_point_select(
        self, points: np.ndarray, quality: np.ndarray, k: int, mesh_scale: float
    ) -> np.ndarray:
        n = points.shape[0]
        if n == 0:
            return np.array([], dtype=np.int64)
        if n <= k:
            return np.arange(n, dtype=np.int64)

        selected = [int(np.argmax(quality))]
        min_dist = np.linalg.norm(points - points[selected[0]][None, :], axis=1)
        min_dist[selected[0]] = 0.0

        while len(selected) < k:
            spread = self._normalize(min_dist)
            plane_gain = np.zeros(n, dtype=np.float64)
            if len(selected) >= 3:
                spts = points[selected]
                c = np.mean(spts, axis=0)
                centered = spts - c[None, :]
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                plane_normal = vh[-1]
                plane_gain = np.abs(np.einsum("ij,j->i", points - c[None, :], plane_normal)) / mesh_scale
                plane_gain = self._normalize(plane_gain)

            objective = 0.60 * spread + 0.25 * plane_gain + 0.15 * quality
            objective[selected] = -np.inf
            nxt = int(np.argmax(objective))
            selected.append(nxt)

            d = np.linalg.norm(points - points[nxt][None, :], axis=1)
            min_dist = np.minimum(min_dist, d)

        return np.array(selected, dtype=np.int64)

    def _raycast_visible(
        self,
        mesh: trimesh.Trimesh,
        camera_position: np.ndarray,
        points: np.ndarray,
        allow_fallback: bool = True,
    ) -> Optional[np.ndarray]:
        n = points.shape[0]
        origins = np.repeat(camera_position[None, :], n, axis=0)
        vec = points - origins
        dist = np.linalg.norm(vec, axis=1)
        dirs = self._safe_normalize(vec)

        try:
            loc, idx_ray, _ = mesh.ray.intersects_location(
                ray_origins=origins,
                ray_directions=dirs,
                multiple_hits=False,
            )
        except BaseException:
            # Fallback is used only for score robustness, not for strict external checks.
            if allow_fallback:
                return np.ones(n, dtype=bool)
            return None

        visible = np.zeros(n, dtype=bool)
        if idx_ray.shape[0] == 0:
            return visible

        hit_dist = np.linalg.norm(loc - origins[idx_ray], axis=1)
        tol = max(float(mesh.scale) * 1e-3, 1e-5)
        visible[idx_ray] = np.abs(hit_dist - dist[idx_ray]) <= tol
        return visible

    def _simulate_crop(
        self, points: np.ndarray, center: np.ndarray, view_dir: np.ndarray
    ) -> np.ndarray:
        up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.abs(np.dot(up_ref, view_dir)) > 0.9:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        right = self._safe_normalize(np.cross(view_dir, up_ref)[None, :])[0]
        up = self._safe_normalize(np.cross(right, view_dir)[None, :])[0]

        rel = points - center[None, :]
        x = np.einsum("ij,j->i", rel, right)
        y = np.einsum("ij,j->i", rel, up)

        x_lim = np.max(np.abs(x)) + 1e-9
        y_lim = np.max(np.abs(y)) + 1e-9

        crop_ratio = self.rng.uniform(0.65, 1.0)
        shift_x = self.rng.uniform(-0.25, 0.25) * x_lim
        shift_y = self.rng.uniform(-0.25, 0.25) * y_lim

        in_crop = (np.abs(x - shift_x) <= crop_ratio * x_lim) & (
            np.abs(y - shift_y) <= crop_ratio * y_lim
        )
        return in_crop

    def _simulate_occluder(
        self,
        camera_position: np.ndarray,
        object_center: np.ndarray,
        points: np.ndarray,
        mesh_scale: float,
    ) -> np.ndarray:
        if self.rng.random() > 0.35:
            return np.zeros(points.shape[0], dtype=bool)

        cam_to_obj = object_center - camera_position
        axis = self._safe_normalize(cam_to_obj[None, :])[0]
        depth = self.rng.uniform(0.3, 0.8) * np.linalg.norm(cam_to_obj)
        base_center = camera_position + depth * axis

        lateral = self.rng.normal(0.0, 0.20 * mesh_scale, size=3)
        occ_center = base_center + lateral
        occ_radius = self.rng.uniform(0.08, 0.20) * mesh_scale

        ray = points - camera_position[None, :]
        ray_len = np.linalg.norm(ray, axis=1) + 1e-12
        ray_dir = ray / ray_len[:, None]

        w = occ_center[None, :] - camera_position[None, :]
        t = np.einsum("ij,ij->i", w, ray_dir)
        t = np.clip(t, 0.0, ray_len)
        closest = camera_position[None, :] + ray_dir * t[:, None]
        d = np.linalg.norm(closest - occ_center[None, :], axis=1)

        return d <= occ_radius

    @staticmethod
    def _k_nearest_indices(points: np.ndarray, k: int) -> np.ndarray:
        dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        return np.argpartition(dist, kth=min(k, dist.shape[1] - 1), axis=1)[:, :k]

    @staticmethod
    def _local_curvature(points: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
        n = points.shape[0]
        curvature = np.zeros(n, dtype=np.float64)
        for i in range(n):
            neigh = points[neighbor_indices[i]]
            c = np.mean(neigh, axis=0)
            cov = (neigh - c[None, :]).T @ (neigh - c[None, :]) / max(neigh.shape[0] - 1, 1)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.clip(eigvals, 1e-12, None)
            curvature[i] = eigvals[0] / np.sum(eigvals)
        return curvature

    @staticmethod
    def _fibonacci_sphere(num: int) -> np.ndarray:
        i = np.arange(num, dtype=np.float64)
        phi = np.pi * (3.0 - np.sqrt(5.0))
        y = 1.0 - (2.0 * i + 1.0) / num
        r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def _safe_normalize(v: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(v, axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        return v / denom

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        lo = np.min(x)
        hi = np.max(x)
        if hi <= lo + 1e-12:
            return np.zeros_like(x, dtype=np.float64)
        return (x - lo) / (hi - lo)

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        mu = np.mean(x, axis=0, keepdims=True)
        sigma = np.std(x, axis=0, keepdims=True) + 1e-12
        return (x - mu) / sigma
