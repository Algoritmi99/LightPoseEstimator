"""
Decoded images kept in memory-mapped files, so training decodes each image once rather than
every epoch. Large images (EO-1's 2048x2048 PNGs) decode far slower than the networks train,
which leaves the GPU waiting. Each cache holds exactly what a dataset computes from the image:

- ``RoiInputCache``: every frame resized for the ROI detector (float32 [C, h, w]), as
  ImageROIDataset resizes it before its augmentation, which works on the resized image;
- ``KeypointRegionCache``: the part of every frame (uint8, full resolution) that any crop of
  ImageKPHeatmapROIDataset can take, its ROI box moved and scaled as far as ``_jitter_roi`` goes.

Each is built once under the cache directory, with a process pool decoding every frame once, and
reused while the images (paths, sizes, modification times, order) and its settings stay the same.
The OS keeps the files in RAM (page cache), shared by every DataLoader worker; if memory runs
short they are read back from disk, still far faster than decoding.
"""
import hashlib
import json
import logging
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh

logger = logging.getLogger(__name__)


def _digest(data: pd.DataFrame, settings: dict, labels: bool = False) -> str:
    """The frames (image files, in order; with ``labels``, poses and cameras too) and settings."""
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True).encode())
    for _, row in data.iterrows():
        stat = os.stat(row["image"])
        digest.update(f"{row['image']}\0{stat.st_size}\0{stat.st_mtime_ns}\0{row.get('bayer_pattern')}\n".encode())
        if labels:
            pose = row["pose"]
            for value in (pose.get_tvec(), pose.get_rvec(), row["camera_intrinsics"], row["distortion_coeffs"]):
                digest.update(np.asarray(value, np.float64).tobytes())
    return digest.hexdigest()[:16]


def _ready(directory: Path, prefix: str) -> bool:
    """Whether ``directory`` holds a finished cache; finished caches of other frames or settings
    with the same ``prefix`` are removed (they can be tens of GB)."""
    for stale in directory.parent.glob(prefix + "*"):
        if stale != directory:
            shutil.rmtree(stale, ignore_errors=True)
    return (directory / "index.npz").is_file()


def _init_worker() -> None:
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


# ROI inputs ------------------------------------------------------------------------------

class RoiInputCache:
    """Frame index -> (ROI input float [C, h, w], full image (height, width))."""

    def __init__(self, directory: Path):
        index = np.load(Path(directory) / "index.npz")
        self.channels, self.sizes = index["channels"], index["sizes"]
        self.input_size = tuple(int(v) for v in index["input_size"])  # (width, height)
        self._path = Path(directory) / "inputs.npy"
        self._inputs = None  # opened in each process that reads it

    def __getstate__(self):
        return {**self.__dict__, "_inputs": None}

    def get(self, index: int) -> tuple[torch.Tensor, tuple[int, int]]:
        if self._inputs is None:
            self._inputs = np.load(self._path, mmap_mode="r")
        # A copy: augmentation may work in place.
        image = torch.from_numpy(np.array(self._inputs[index, : self.channels[index]]))
        height, width = self.sizes[index]
        return image, (int(height), int(width))


_roi_inputs = None
_roi_buffer: torch.Tensor | None = None


def _init_roi_worker(path: str) -> None:
    global _roi_inputs
    _init_worker()
    _roi_inputs = np.load(path, mmap_mode="r+")


def _roi_input_job(job) -> tuple[int, int, tuple[int, int]]:
    global _roi_buffer
    from .dataloader import pixels_to_tensor, read_pixels
    from .roi.roi_detector import resize_for_roi

    index, path, pattern, input_size = job
    pixels = read_pixels({"image": path, "bayer_pattern": pattern})
    shape = (1, *pixels.shape) if pixels.ndim == 2 else (pixels.shape[2], *pixels.shape[:2])
    if _roi_buffer is None or tuple(_roi_buffer.shape) != shape:
        _roi_buffer = torch.empty(shape, dtype=torch.float32)
    image = resize_for_roi(pixels_to_tensor(pixels, _roi_buffer), input_size)
    _roi_inputs[index, : image.shape[0]] = image.numpy()
    return index, image.shape[0], pixels.shape[:2]


def build_roi_input_cache(data: pd.DataFrame, input_size: tuple[int, int], cache_dir: Path, workers: int) -> RoiInputCache:
    """The ROI input of every frame of ``data`` (rows in order), resized to ``input_size`` (width, height)."""
    width, height = input_size
    prefix = f"roi-{width}x{height}-"
    directory = Path(cache_dir) / (prefix + _digest(data, {"input_size": [width, height]}))
    if _ready(directory, prefix):
        logger.info("ROI inputs of %d frames are cached in %s.", len(data), directory)
        return RoiInputCache(directory)
    partial = directory.with_name(directory.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    # Room for 3 channels per frame; a gray frame fills one and leaves the rest of the
    # (sparse) file unwritten, taking no space.
    np.lib.format.open_memmap(partial / "inputs.npy", mode="w+", dtype=np.float32, shape=(len(data), 3, height, width))
    jobs = [(i, str(row["image"]), row.get("bayer_pattern") if isinstance(row.get("bayer_pattern"), str) else None, (width, height))
            for i, (_, row) in enumerate(data.iterrows())]
    channels = np.zeros(len(data), np.int8)
    sizes = np.zeros((len(data), 2), np.int32)
    logger.info("Caching the ROI inputs of %d frames in %s (%.1f GB at most) with %d processes.",
                len(data), directory, len(data) * 3 * height * width * 4 / 1e9, workers)
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init_roi_worker, initargs=(str(partial / "inputs.npy"),)) as pool:
        for index, count, size in tqdm(pool.map(_roi_input_job, jobs, chunksize=8), total=len(jobs), desc="ROI inputs", unit="frame"):
            channels[index], sizes[index] = count, size
    np.savez(partial / "index.npz", channels=channels, sizes=sizes, input_size=np.array([width, height]))
    partial.rename(directory)
    return RoiInputCache(directory)


# Keypoint regions ------------------------------------------------------------------------

class KeypointRegionCache:
    """Frame index -> (uint8 region [h, w] or [h, w, 3], its top-left (x, y) in the image,
    full image (height, width))."""

    def __init__(self, directory: Path):
        index = np.load(Path(directory) / "index.npz")
        self.offsets, self.shapes, self.origins, self.sizes = index["offsets"], index["shapes"], index["origins"], index["sizes"]
        self._path = Path(directory) / "regions.bin"
        self._pixels = None  # opened in each process that reads it

    def __getstate__(self):
        return {**self.__dict__, "_pixels": None}

    def get(self, index: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        if self._pixels is None:
            self._pixels = np.memmap(self._path, dtype=np.uint8, mode="r")
        h, w, c = (int(v) for v in self.shapes[index])
        start = int(self.offsets[index])
        region = self._pixels[start:start + h * w * c].reshape((h, w) if c == 1 else (h, w, c))
        x0, y0 = (int(v) for v in self.origins[index])
        height, width = (int(v) for v in self.sizes[index])
        return region, (x0, y0), (height, width)


def keypoint_region(roi, height: int, width: int, shift: float, scale: float) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1), end exclusive: every pixel a crop box of ``roi`` can cover, the box
    moved by up to ``shift`` and scaled by up to ``scale`` (as _jitter_roi does), then clamped
    to the image as crop.crop_box does."""
    from .crop import crop_box

    if roi.w <= 0 or roi.h <= 0:
        return crop_box(roi, height, width)
    reach = shift + (1.0 + scale) / 2.0
    x0 = max(0, math.floor(roi.cx - reach * roi.w) - 1)
    y0 = max(0, math.floor(roi.cy - reach * roi.h) - 1)
    x1 = min(width, math.ceil(roi.cx + reach * roi.w) + 1)
    y1 = min(height, math.ceil(roi.cy + reach * roi.h) + 1)
    if x1 <= x0 or y1 <= y0:
        return crop_box(roi, height, width)
    return x0, y0, x1, y1


_region_pixels = None


def _init_region_worker(path: str) -> None:
    global _region_pixels
    _init_worker()
    _region_pixels = np.memmap(path, dtype=np.uint8, mode="r+")


def _region_job(job) -> tuple[int, int]:
    from .dataloader import read_pixels

    index, path, pattern, (x0, y0, x1, y1), offset = job
    region = np.ascontiguousarray(read_pixels({"image": path, "bayer_pattern": pattern})[y0:y1, x0:x1])
    _region_pixels[offset:offset + region.size] = region.reshape(-1)
    return index, 1 if region.ndim == 2 else region.shape[2]


def build_keypoint_region_cache(
    data: pd.DataFrame,
    mesh: trimesh.Trimesh,
    margin: float,
    cache_dir: Path,
    workers: int,
) -> KeypointRegionCache:
    """The keypoint region of every frame of ``data`` (rows in order), for ROIs with ``margin``."""
    from PIL import Image
    from .dataloader import JITTER_SCALE, JITTER_SHIFT, _get_2d_roi
    from .masks import _mesh_digest

    settings = {"margin": margin, "shift": JITTER_SHIFT, "scale": JITTER_SCALE, "mesh": _mesh_digest(mesh)}
    prefix = f"keypoint-regions-m{margin:g}-"
    directory = Path(cache_dir) / (prefix + _digest(data, settings, labels=True))
    if _ready(directory, prefix):
        logger.info("Keypoint regions of %d frames are cached in %s.", len(data), directory)
        return KeypointRegionCache(directory)

    jobs, shapes, origins, sizes, offsets = [], [], [], [], []
    offset = 0
    for i, (_, row) in enumerate(data.iterrows()):
        with Image.open(row["image"]) as image:
            height, width = image.height, image.width
        roi = _get_2d_roi((height, width), mesh, row["pose"], row["camera_intrinsics"], row["distortion_coeffs"], margin)
        x0, y0, x1, y1 = keypoint_region(roi, height, width, JITTER_SHIFT, JITTER_SCALE)
        pattern = row.get("bayer_pattern") if isinstance(row.get("bayer_pattern"), str) else None
        jobs.append((i, str(row["image"]), pattern, (x0, y0, x1, y1), offset))
        shapes.append((y1 - y0, x1 - x0))
        origins.append((x0, y0))
        sizes.append((height, width))
        offsets.append(offset)
        offset += (y1 - y0) * (x1 - x0) * 3  # room for RGB; gray frames leave the rest unwritten (sparse)

    partial = directory.with_name(directory.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    np.memmap(partial / "regions.bin", dtype=np.uint8, mode="w+", shape=(max(offset, 1),))
    channels = np.zeros(len(data), np.int32)
    logger.info("Caching the keypoint regions of %d frames in %s (%.1f GB at most) with %d processes.",
                len(data), directory, offset / 1e9, workers)
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init_region_worker, initargs=(str(partial / "regions.bin"),)) as pool:
        for index, count in tqdm(pool.map(_region_job, jobs, chunksize=8), total=len(jobs), desc="Keypoint regions", unit="frame"):
            channels[index] = count
    np.savez(
        partial / "index.npz",
        offsets=np.asarray(offsets, np.int64),
        shapes=np.column_stack([np.asarray(shapes, np.int32).reshape(-1, 2), channels]),
        origins=np.asarray(origins, np.int32).reshape(-1, 2),
        sizes=np.asarray(sizes, np.int32).reshape(-1, 2),
    )
    partial.rename(directory)
    return KeypointRegionCache(directory)
