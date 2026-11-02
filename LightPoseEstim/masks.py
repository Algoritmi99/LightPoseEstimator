"""
Object silhouettes rendered from the mesh at each frame's labelled pose, cached on disk, and a
training augmentation that shows the object in front of other backgrounds.

Background randomization keeps the keypoint net from memorising the scenes it was trained on:
on the merged Astrobee data, unseen recordings with the ISS interior behind the target did much
worse than unseen ones with a plain background, at the same poses.
"""
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import trimesh
from PIL import Image

from .crop import Box, crop_and_resize
from .pose import Pose
from .projection import max_valid_radius, project_points, projectable_mask

logger = logging.getLogger(__name__)

_SMALL_BOX = 8           # triangles whose pixel bounding box fits in 8x8 are rasterised in bulk
_CHUNK = 40_000          # triangles per bulk step (bounds memory to ~20 MB)
BACKGROUND_SIZE = 256    # side of the cached background patches


def _camera_points(points: np.ndarray, pose: Pose) -> list[np.ndarray]:
    """Camera-frame x, y, z of ``points``, written out rather than as a matrix product: that
    would start a multithreaded BLAS pool in every mask worker process and oversubscribe the CPU."""
    rotation, _ = cv2.Rodrigues(np.asarray(pose.get_rvec(), np.float64))
    t = np.asarray(pose.get_tvec(), np.float64).ravel()
    px, py, pz = points[:, 0], points[:, 1], points[:, 2]
    return [rotation[k, 0] * px + rotation[k, 1] * py + rotation[k, 2] * pz + t[k] for k in range(3)]


def _project_valid(points: np.ndarray, pose: Pose, camera_intrinsics: np.ndarray, dist_coeffs: np.ndarray | None):
    """
    ``projectable_mask`` and ``project_points`` for many points at once: OpenCV's distortion
    model with up to five coefficients (k1, k2, p1, p2, k3) in numpy, several times faster than
    cv2.projectPoints here. Returns (valid mask, [N, 2] pixel coordinates, NaN where invalid).
    """
    coeffs = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, np.float64).ravel()
    if len(coeffs) > 5 and np.any(coeffs[5:]):
        valid = projectable_mask(points, pose, dist_coeffs)
        uv = np.full((len(points), 2), np.nan)
        uv[valid] = project_points(points[valid], pose, camera_intrinsics, dist_coeffs)
        return valid, uv
    k1, k2, p1, p2, k3 = np.pad(coeffs[:5], (0, 5 - len(coeffs[:5])))
    cx, cy, cz = _camera_points(points, pose)
    in_front = cz > 1e-6
    safe_z = np.where(in_front, cz, 1.0)
    x, y = cx / safe_z, cy / safe_z
    r2 = x * x + y * y
    valid = in_front & (np.sqrt(r2) <= max_valid_radius(dist_coeffs))
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = np.asarray(camera_intrinsics, np.float64)
    uv = np.stack([K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=1)
    uv[~valid] = np.nan
    return valid, uv


def silhouette_mask(
    mesh: trimesh.Trimesh,
    pose: Pose,
    camera_intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Boolean [H, W] mask of the pixels whose centre the projected mesh covers (the union of its
    triangles, so no depth test is needed). Triangles with a vertex past the distortion fold
    are left out, as everywhere else in the pipeline.
    """
    height, width = shape
    vertices = np.asarray(mesh.vertices)
    valid, uv = _project_valid(vertices, pose, camera_intrinsics, dist_coeffs)
    faces = mesh.faces[valid[mesh.faces].all(axis=1)]
    a, b, c = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    tri = np.stack([a, b, c], axis=1)                       # [F, 3, 2]
    # Pixel centres are the integer coordinates (OpenCV convention).
    lo = np.ceil(np.minimum(np.minimum(a, b), c))
    hi = np.floor(np.maximum(np.maximum(a, b), c))
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [width - 1, height - 1])
    span = hi - lo + 1
    covers = (span > 0).all(axis=1)
    tri, lo, span = tri[covers], lo[covers], span[covers]
    mask = np.zeros((height, width), dtype=bool)

    # Most triangles cover one or two pixel centres, so test them in groups by box size
    # (1x1, 2x2, 4x4, 8x8 candidates) rather than all against 8x8.
    extent = span.max(axis=1)
    small = extent <= _SMALL_BOX
    cross = lambda p, q, r: (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    previous = 0
    for side in (1, 2, 4, _SMALL_BOX):
        group = (extent > previous) & (extent <= side)
        previous = side
        offsets = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="xy"), -1).reshape(-1, 2)
        t_all, o_all, s_all = tri[group], lo[group], span[group]
        for start in range(0, len(t_all), max(1, _CHUNK * 4 // side ** 2)):
            stop = start + max(1, _CHUNK * 4 // side ** 2)
            t, o, s = t_all[start:stop], o_all[start:stop], s_all[start:stop]
            points = o[:, None, :] + offsets[None]           # [N, side^2, 2]
            in_box = (offsets[None] < s[:, None, :]).all(axis=2)
            a, b, c = t[:, None, 0], t[:, None, 1], t[:, None, 2]
            e0, e1, e2 = cross(a, b, points), cross(b, c, points), cross(c, a, points)
            inside = in_box & (((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0)))
            hit = points[inside].astype(np.int64)
            mask[hit[:, 1], hit[:, 0]] = True

    big = tri[~small]
    if len(big):
        canvas = np.zeros((height, width), dtype=np.uint8)
        for triangle in np.round(big * 16).astype(np.int32):  # 4 fractional bits
            cv2.fillConvexPoly(canvas, triangle, 1, shift=4)
        mask |= canvas.astype(bool)
    return mask


def _mask_path(cache_dir: Path, row) -> Path:
    """One file per frame, named after its image and a digest of its pose and camera, so a
    changed label or calibration is simply a cache miss."""
    digest = hashlib.sha1()
    for value in (row["pose"].get_tvec(), row["pose"].get_rvec(), row["camera_intrinsics"], row["distortion_coeffs"]):
        digest.update(np.asarray(value, np.float64).tobytes())
    return Path(cache_dir) / row["recording"] / f"{Path(row['image']).stem}-{digest.hexdigest()[:10]}.png"


def _mesh_digest(mesh: trimesh.Trimesh) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


_worker_mesh: trimesh.Trimesh | None = None


def _single_threaded() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _init_mask_worker(vertices: np.ndarray, faces: np.ndarray) -> None:
    global _worker_mesh
    _single_threaded()
    _worker_mesh = trimesh.Trimesh(vertices, faces, process=False)


def _render_mask_job(job) -> None:
    path, pose_values, camera_intrinsics, dist_coeffs, shape = job
    mask = silhouette_mask(_worker_mesh, Pose(*pose_values), camera_intrinsics, dist_coeffs, shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and renamed, so an interrupted run leaves no partial file.
    partial = path.with_name(path.stem + ".partial.png")
    cv2.imwrite(str(partial), mask.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_BILEVEL, 1])
    partial.replace(path)


def build_mask_cache(data: pd.DataFrame, mesh: trimesh.Trimesh, cache_dir: Path, workers: int) -> Path:
    """
    Render and store the silhouette of every frame in ``data`` that has none yet (one bilevel
    PNG each, a few KB) under ``cache_dir``. A different mesh starts the cache over.
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "mesh.json"
    meta = {"mesh": _mesh_digest(mesh)}
    if meta_path.is_file() and json.loads(meta_path.read_text()) != meta:
        logger.info("The mesh changed since %s was filled; rendering all masks again.", cache_dir)
        for old in cache_dir.glob("*/*.png"):
            old.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta) + "\n")
    jobs = []
    for _, row in data.iterrows():
        path = _mask_path(cache_dir, row)
        if path.is_file():
            continue
        with Image.open(row["image"]) as image:
            shape = (image.height, image.width)
        q, t = row["pose"].quaternion, row["pose"].position
        jobs.append((path, (t.x, t.y, t.z, q.qx, q.qy, q.qz, q.qw),
                     np.asarray(row["camera_intrinsics"], np.float64), np.asarray(row["distortion_coeffs"], np.float64), shape))
    if not jobs:
        logger.info("All %d object masks are cached in %s.", len(data), cache_dir)
        return cache_dir
    logger.info("Rendering %d object masks to %s with %d processes.", len(jobs), cache_dir, workers)
    from tqdm import tqdm
    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        initializer=_init_mask_worker,
        initargs=(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
    ) as pool:
        for _ in tqdm(pool.map(_render_mask_job, jobs, chunksize=16), total=len(jobs), desc="Object masks", unit="frame"):
            pass
    return cache_dir


def read_mask(cache_dir: Path, row) -> np.ndarray:
    """Cached boolean silhouette of a frame."""
    mask = cv2.imread(str(_mask_path(Path(cache_dir), row)), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"No cached mask for {row['image']} in {cache_dir}.")
    return mask > 0


def _background_patch_job(job) -> np.ndarray | None:
    image_path, bayer_pattern, mask_path, seed = job
    from .dataloader import read_image
    rng = np.random.default_rng(seed)
    gray = np.asarray(read_image({"image": image_path, "bayer_pattern": bayer_pattern}).convert("L"))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    # Keep clear of the object, including a margin for its shadow and the mask's edge.
    blocked = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8))
    integral = cv2.integral(blocked)            # int32 sums of the 0/1 image
    height, width = gray.shape
    for _ in range(30):
        side = int(rng.uniform(0.12, 0.6) * min(height, width))
        x = int(rng.integers(0, width - side + 1))
        y = int(rng.integers(0, height - side + 1))
        if integral[y + side, x + side] - integral[y, x + side] - integral[y + side, x] + integral[y, x] == 0:
            patch = gray[y:y + side, x:x + side]
            return cv2.resize(patch, (BACKGROUND_SIZE, BACKGROUND_SIZE), interpolation=cv2.INTER_AREA)
    return None


def build_background_pool(data: pd.DataFrame, mask_dir: Path, path: Path, count: int, workers: int, seed: int = 0) -> Path:
    """
    Square grayscale patches of ``data``'s images that don't show the object, stored as one
    [count, 256, 256] uint8 array at ``path``. Pass only training frames, so the scenes of
    validation and test recordings never appear in training.
    """
    path = Path(path)
    if path.is_file():
        logger.info("Using the background patches cached in %s.", path)
        return path
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(data), size=min(len(data), int(count * 1.3)), replace=False)
    jobs = []
    for k, i in enumerate(picks):
        row = data.iloc[i]
        jobs.append((row["image"], row.get("bayer_pattern"), _mask_path(Path(mask_dir), row), seed * 1_000_003 + k))
    patches = []
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_single_threaded) as pool:
        for patch in pool.map(_background_patch_job, jobs, chunksize=8):
            if patch is not None:
                patches.append(patch)
            if len(patches) == count:
                break
    if not patches:
        raise RuntimeError("Found no object-free background patches.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.stack(patches))
    logger.info("Saved %d background patches to %s.", len(patches), path)
    return path


class BackgroundCompositor:
    """
    With probability ``p``, replace everything around the object in a training crop: by an
    object-free patch of another training image, or (``plain_share`` of the time) by a dark,
    noisy plain background like a render without scenery. The object's pixels are kept, so
    the crop's keypoint and rotation targets stay valid.
    """

    def __init__(self, mask_dir: Path, pool_path: Path, p: float = 0.5, plain_share: float = 0.25):
        self.mask_dir = Path(mask_dir)
        self.pool_path = Path(pool_path)
        self.p = p
        self.plain_share = plain_share
        self._pool = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pool"] = None     # each worker maps the file itself
        return state

    def _patches(self) -> np.ndarray:
        if self._pool is None:
            self._pool = np.load(self.pool_path, mmap_mode="r")
        return self._pool

    def __call__(self, crop: torch.Tensor, row, box: Box) -> torch.Tensor:
        # torch's RNG is seeded per DataLoader worker, unlike numpy's global one.
        if float(torch.rand(())) >= self.p:
            return crop
        size = crop.shape[-1]
        mask = torch.from_numpy(read_mask(self.mask_dir, row).astype(np.float32))[None]
        alpha = crop_and_resize(mask, box, size)
        if float(torch.rand(())) < self.plain_share:
            level = 0.04 + 0.16 * float(torch.rand(()))
            background = level + torch.randn(1, size, size) * (0.005 + 0.02 * float(torch.rand(())))
        else:
            patches = self._patches()
            patch = torch.from_numpy(np.array(patches[int(torch.randint(len(patches), ()))], dtype=np.float32) / 255.0)
            patch = torch.rot90(patch, int(torch.randint(4, ())), dims=(0, 1))
            if float(torch.rand(())) < 0.5:
                patch = patch.flip(1)
            background = patch[None]
            if background.shape[-1] != size:
                background = torch.nn.functional.interpolate(background[None], size=(size, size), mode="bilinear", align_corners=False)[0]
        return (alpha * crop + (1.0 - alpha) * background).clamp(0.0, 1.0)
