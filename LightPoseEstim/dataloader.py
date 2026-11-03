import copy
import fnmatch
import json
import logging
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm

from .crop import HEATMAP_STRIDE, Box, crop_and_resize, crop_box, image_to_crop
from .pose import Pose
from .projection import (
    MeshOccluder,
    allocentric_rotation,
    camera_frame_points,
    face_normals_at_points,
    keypoint_visibility,
    project_points,
    projectable_mask,
)
from .roi import BBox, ROI, bbox_to_roi, normalize_roi
from .roi.roi_detector import ROI_INPUT_SIZE, resize_for_roi
from .keypoint_selector import CanonicalKeypoint
from .bayer import BAYER_TO_GRAY
from .gpu_augment import BatchPhotometric

logger = logging.getLogger(__name__)

DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_SPLIT_SEED = 0


@dataclass(frozen=True)
class RecordingSplit:
    """Train/val/test recording names. Whole recordings go to one split, so near-duplicate frames never leak."""
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def indices(self, data: pd.DataFrame, part: str) -> list[int]:
        """Row positions of ``data`` whose recording is in ``part`` (train, val or test)."""
        recordings = set(getattr(self, part))
        return np.flatnonzero(data["recording"].isin(recordings).to_numpy()).tolist()

    def to_dict(self) -> dict:
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test)}

    @classmethod
    def from_dict(cls, data: dict) -> "RecordingSplit":
        return cls(tuple(data["train"]), tuple(data["val"]), tuple(data["test"]))


def split_by_recording(
    data: pd.DataFrame,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> RecordingSplit:
    recordings = sorted(data["recording"].unique())
    np.random.default_rng(seed).shuffle(recordings)
    n_test = round(len(recordings) * test_fraction)
    n_val = round(len(recordings) * val_fraction)
    if n_test + n_val >= len(recordings):
        raise ValueError(
            f"Cannot split {len(recordings)} recordings into "
            f"val={val_fraction} and test={test_fraction} with any left for training."
        )
    if val_fraction > 0 and n_val == 0:
        raise ValueError(f"val_fraction={val_fraction} selects no recordings out of {len(recordings)}.")
    return RecordingSplit(
        train=tuple(sorted(recordings[n_test + n_val:])),
        val=tuple(sorted(recordings[n_test:n_test + n_val])),
        test=tuple(sorted(recordings[:n_test])),
    )


def _find_image(image_dir: Path, timestamp, suffix: str = "") -> list[Path]:
    base = Decimal(str(timestamp))
    for offset in (Decimal("0"), Decimal("0.008"), Decimal("-0.008")):
        text = format(base + offset, "f").rstrip("0").rstrip(".")
        stem = text.replace(".", "_")
        stems = [stem] if "_" in stem else [stem, f"{stem}_0"]
        for candidate in stems:
            image_path = image_dir / f"{candidate}{suffix}.png"
            if image_path.exists():
                return [image_path]
    return []


def read_image(row) -> Image.Image:
    """
    RGB image of a dataset row. Rows whose ``bayer_pattern`` names a pattern hold the raw sensor
    mosaic; it is converted to grayscale (replicated to RGB) first.
    """
    pattern = row.get("bayer_pattern")
    if not isinstance(pattern, str):
        return Image.open(row["image"]).convert("RGB")
    with Image.open(row["image"]) as image:
        raw = np.asarray(image.convert("L"))
    return Image.fromarray(cv2.cvtColor(raw, BAYER_TO_GRAY[pattern])).convert("RGB")


def _all_channels_equal(pixels: np.ndarray) -> bool:
    # A few rows first: a colour image almost always shows it there already.
    sample = pixels[::16]
    if not ((sample[..., 0] == sample[..., 1]).all() and (sample[..., 1] == sample[..., 2]).all()):
        return False
    return bool((pixels[..., 0] == pixels[..., 1]).all() and (pixels[..., 1] == pixels[..., 2]).all())


def read_pixels(row) -> np.ndarray:
    """
    uint8 pixels of a dataset row, those ``read_image(row)`` has: [H, W] for gray images (RGB
    files whose channels are all equal included), else [H, W, 3] RGB. PNGs are decoded with
    OpenCV, about twice as fast as PIL on large images and the same pixels (PNG is lossless);
    anything else, and PNGs OpenCV reads otherwise (e.g. 16-bit), with PIL as before.
    """
    pattern = row.get("bayer_pattern")
    path = str(row["image"])
    pixels = None
    if not isinstance(pattern, str) and path.lower().endswith(".png"):
        decoded = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if decoded is not None and decoded.dtype == np.uint8:
            if decoded.ndim == 2:
                pixels = decoded
            elif decoded.shape[2] in (3, 4):
                pixels = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB if decoded.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    if pixels is None:
        with Image.open(path) as image:
            if isinstance(pattern, str):
                return cv2.cvtColor(np.asarray(image.convert("L")), BAYER_TO_GRAY[pattern])
            pixels = np.asarray(image) if image.mode == "L" else np.asarray(image.convert("RGB"))
    if pixels.ndim == 3 and _all_channels_equal(pixels):
        pixels = pixels[..., 0]
    return pixels


def pixels_to_tensor(pixels: np.ndarray, out: torch.Tensor | None = None) -> torch.Tensor:
    """
    uint8 [H, W] or [H, W, C] pixels as float [C, H, W] in [0, 1], the values torchvision's
    ToTensor gives. ``out``, a float tensor of that shape, is filled instead of a new one: for
    a large image, fresh memory costs more than the conversion itself.
    """
    # Decoded PIL pixels are read-only; torch needs a writable array (copied only if it isn't).
    source = torch.from_numpy(np.require(pixels, requirements=("W",)))
    source = source.unsqueeze(0) if source.ndim == 2 else source.permute(2, 0, 1)
    if out is None:
        out = torch.empty(source.shape, dtype=torch.float32)
    return out.copy_(source).div_(255)


def read_image_tensor(row) -> torch.Tensor:
    """
    Float [C, H, W] image in [0, 1] of a dataset row, with the values ``T.ToTensor()(read_image(row))``
    has. Gray images, including RGB files whose channels are all equal, come back with C = 1:
    the training transforms work per channel and keep gray images gray, so ``_as_rgb`` at the
    end gives the network the same input for a third of the work.
    """
    return pixels_to_tensor(read_pixels(row))


def _as_rgb(image: torch.Tensor) -> torch.Tensor:
    """A one-channel image as three (a view; batching copies it)."""
    return image.expand(3, -1, -1) if image.shape[0] == 1 else image


def _apply_photometric(photometric: T.Compose, image: torch.Tensor, low_light: float) -> torch.Tensor:
    # Colour jitter and blur keep a gray image gray (torchvision skips saturation and hue on one
    # channel; on three equal ones they change values by under 1e-4), so one channel stays one.
    # Simulated low light adds noise per channel, so it gets all three.
    if low_light > 0:
        image = _as_rgb(image)
    return photometric(image)


def init_worker(_worker_id: int) -> None:
    """DataLoader worker setup: one OpenCV thread, as torch already uses in workers."""
    cv2.setNumThreads(1)


def _find_distorted_image_dir(directory: Path) -> Path:
    candidates = sorted(
        p for p in directory.glob("*imgs")
        if p.is_dir() and not p.name.endswith("_undistorted")
    )
    if not candidates:
        raise StopIteration(f"No distorted image directory in {directory}")
    return candidates[0]


def _find_undistorted_image_dir(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("*imgs_undistorted"))
    if not candidates:
        return None
    return candidates[0]


def _get_2d_roi(
    img_shape: tuple,
    mesh: trimesh.Trimesh,
    pose: Pose,
    camera_intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    margin: float = 1.2,
) -> ROI:
    height, width = img_shape[-2:]
    # The projected convex hull bounds the projected mesh exactly; the 3D bounding box
    # is much looser on non-box shapes (about 2x the area on a satellite with a panel).
    hull_points = np.asarray(mesh.convex_hull.vertices)

    camera_points = camera_frame_points(hull_points, pose)
    in_front_mask = camera_points[:, 2] > 1e-6
    if int(np.sum(in_front_mask)) < 2:
        return ROI(0.0, 0.0, 0.0, 0.0)

    projectable = projectable_mask(hull_points, pose, dist_coeffs)
    image_points = [np.empty((0, 2), dtype=np.float32)]
    if projectable.any():
        image_points.append(
            project_points(hull_points[projectable], pose, camera_intrinsics, dist_coeffs)
        )
    # Corners past the distortion fold would be projected back into the image.
    # They lie outside it along their own direction, so push them past the edge.
    beyond_fold = in_front_mask & ~projectable
    if beyond_fold.any():
        focal = np.diag(camera_intrinsics)[:2]
        direction = camera_points[beyond_fold, :2] / camera_points[beyond_fold, 2:3] * focal
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        image_points.append(camera_intrinsics[:2, 2] + direction * float(width + height))
    image_points = np.concatenate(image_points)

    x1 = float(image_points[:, 0].min())
    x2 = float(image_points[:, 0].max())
    y1 = float(image_points[:, 1].min())
    y2 = float(image_points[:, 1].max())
    if x2 < 0 or y2 < 0 or x1 > width - 1 or y1 > height - 1:
        return ROI(0.0, 0.0, 0.0, 0.0)

    x1 = float(np.clip(x1, 0, width - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    y2 = float(np.clip(y2, 0, height - 1))

    roi = bbox_to_roi(BBox(x1, y1, x2, y2))
    roi.w *= margin
    roi.h *= margin
    return roi


def _gaussian_heatmap(height: int, width: int, cx: float, cy: float, sigma: float = 3.0) -> torch.Tensor:
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    return torch.from_numpy(heatmap)


class RandomLowLight:
    """
    Simulate a poorly lit target, since real dim frames are rare (3% of EO-1 is backlit):
    scale the signal above the image's dark level down (log-uniform, to ~3%, about as dark
    as backlit frames) while keeping that level, as a camera's noise floor stays, bend the
    tone curve, add sensor noise and re-quantise to 8 bits, which is what erases detail at
    low intensities. ``noise`` is the noise std range at the network's input resolution.

    Opt-in (``low_light`` on the datasets): on EO-1, darkening 30% of training images gave no
    gain on dim test frames and cost some accuracy on well-lit ones. Backlit targets are lit
    only at their edges, which darkening a lit image does not reproduce.
    """

    def __init__(
        self,
        p: float = 0.3,
        gain: tuple[float, float] = (0.03, 0.6),
        gamma: tuple[float, float] = (0.8, 1.5),
        noise: tuple[float, float] = (0.005, 0.02),
    ):
        self.p, self.gain, self.gamma, self.noise = p, gain, gamma, noise

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # torch's RNG is seeded per DataLoader worker, unlike numpy's global one.
        if float(torch.rand(())) >= self.p:
            return image
        u = torch.rand(3).tolist()
        gain = float(np.exp(np.log(self.gain[0]) + u[0] * np.log(self.gain[1] / self.gain[0])))
        gamma = self.gamma[0] + u[1] * (self.gamma[1] - self.gamma[0])
        sigma = self.noise[0] + u[2] * (self.noise[1] - self.noise[0])
        image = image.clamp(0.0, 1.0)
        # Dark level: median of the border, which is almost always background.
        border = torch.cat([image[:, :8].flatten(1), image[:, -8:].flatten(1),
                            image[:, :, :8].flatten(1), image[:, :, -8:].flatten(1)], dim=1)
        floor = border.median(dim=1).values[:, None, None]
        image = torch.minimum(image, floor) + (image - floor).clamp_min(0.0).pow(gamma) * gain
        image = image + torch.randn_like(image) * sigma
        return (image.clamp(0.0, 1.0) * 255.0).round() / 255.0


def _photometric_augmentation(
    low_light: float = 0.0,
    noise: tuple[float, float] = (0.005, 0.02),
) -> T.Compose:
    """Colour jitter and blur; with ``low_light`` > 0, also darken that share of images."""
    transforms = [
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
    ]
    if low_light > 0:
        transforms.append(RandomLowLight(p=low_light, noise=noise))
    return T.Compose(transforms)


# _jitter_roi's limits: the ROI centre moves by up to this share of the box, its size by this share.
JITTER_SHIFT = 0.2
JITTER_SCALE = 0.25


def _jitter_roi(roi: ROI, shift: float = JITTER_SHIFT, scale: float = JITTER_SCALE) -> ROI:
    """
    Random shift and scale, so training crops look like crops from a predicted ROI.
    The defaults cover the ROI detector's measured validation errors (centre off by up to
    ~0.23 of the box size and size off by ~20% at the 90th percentile).
    """
    if roi.w <= 0 or roi.h <= 0:
        return roi
    # torch's RNG is seeded per DataLoader worker, unlike numpy's global one.
    r = (torch.rand(3) * 2.0 - 1.0).tolist()
    factor = 1.0 + scale * r[2]
    return ROI(roi.cx + shift * roi.w * r[0], roi.cy + shift * roi.h * r[1], roi.w * factor, roi.h * factor)


class ImagePoseDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_undistorted"]).convert("RGB")
        image = self.transform(image)
        return image, row["pose"]


class ImageROIDataset(Dataset):
    """Images resized to ``input_size`` (width, height) with normalized (cx, cy, w, h) box targets."""

    def __init__(
        self,
        data: pd.DataFrame,
        mesh: trimesh.Trimesh,
        margin: float = 1.4,
        augment: bool = False,
        input_size: tuple[int, int] = ROI_INPUT_SIZE,
        low_light: float = 0.0,
        cache=None,
    ):
        """``cache`` (an ``image_cache.RoiInputCache`` of ``data``) supplies the resized images
        instead of decoding every frame each time: the same values."""
        if cache is not None and (len(cache.sizes) != len(data) or cache.input_size != tuple(input_size)):
            raise ValueError("The ROI input cache is for other frames or another input size.")
        self.df = data
        self.mesh = mesh
        self.margin = margin
        self.cache = cache
        self.input_size = tuple(input_size)
        self.augment = augment
        self.low_light = low_light
        # Downscaling to the ROI input averages sensor noise away, so add less of it.
        self.photometric = _photometric_augmentation(low_light, noise=(0.002, 0.008))
        # The same for whole batches on their device, for a trainer to apply in place of
        # ``photometric`` (see with_augmentation).
        self.batch_augmentation = BatchPhotometric(low_light=low_light, noise=(0.002, 0.008))
        self.batch_photometric = False

    def with_augmentation(self, augment: bool, batch_photometric: bool = False) -> "ImageROIDataset":
        """Shallow copy sharing the loaded data, with zoom/flip/colour augmentation on or off.
        With ``batch_photometric``, the colour jitter, blur and low light are left to
        ``batch_augmentation``, for the trainer to apply to whole batches."""
        dataset = copy.copy(self)
        dataset.augment = augment
        dataset.batch_photometric = batch_photometric
        return dataset

    def __len__(self):
        return len(self.df)

    def __getstate__(self):
        # Each DataLoader worker gets its own conversion buffer.
        return {**self.__dict__, "_buffer": None}

    def _full_image(self, pixels: np.ndarray) -> torch.Tensor:
        """The whole image as float [C, H, W], in a buffer reused from sample to sample: only its
        resized copy leaves this dataset, and converting into memory already in use is far
        cheaper than into fresh memory for a large image."""
        shape = (1, *pixels.shape) if pixels.ndim == 2 else (pixels.shape[2], *pixels.shape[:2])
        if getattr(self, "_buffer", None) is None or tuple(self._buffer.shape) != shape:
            self._buffer = torch.empty(shape, dtype=torch.float32)
        return pixels_to_tensor(pixels, self._buffer)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.cache is not None:
            image, (height, width) = self.cache.get(idx)
        else:
            pixels = read_pixels(row)
            height, width = pixels.shape[:2]
            image = resize_for_roi(self._full_image(pixels), self.input_size)

        roi = _get_2d_roi(
            (height, width),
            self.mesh,
            row["pose"],
            row["camera_intrinsics"],
            row["distortion_coeffs"],
            self.margin,
        )
        # As normalize_roi does for an image of this size.
        target = torch.tensor([roi.cx, roi.cy, roi.w, roi.h], dtype=torch.float32) * torch.tensor(
            [1 / width, 1 / height, 1 / width, 1 / height], dtype=torch.float32
        )
        if self.augment:
            image, target = self._augment(image, target)
        return _as_rgb(image), target

    def _augment(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # torch's RNG is seeded per DataLoader worker, unlike numpy's global one.
        cx, cy, w, h = target.tolist()
        _, height, width = image.shape
        if w > 0 and h > 0:
            # Zoom into a random window (75-100% of the image) that still holds the whole box.
            scale = 1.0 - 0.25 * float(torch.rand(()))
            crop_w, crop_h = round(scale * width), round(scale * height)
            x_lo, x_hi = max(0.0, cx + w / 2 - crop_w / width), min(cx - w / 2, 1.0 - crop_w / width)
            y_lo, y_hi = max(0.0, cy + h / 2 - crop_h / height), min(cy - h / 2, 1.0 - crop_h / height)
            if x_lo <= x_hi and y_lo <= y_hi:
                x0 = round((x_lo + (x_hi - x_lo) * float(torch.rand(()))) * width)
                y0 = round((y_lo + (y_hi - y_lo) * float(torch.rand(()))) * height)
                image = F.interpolate(
                    image[:, y0:y0 + crop_h, x0:x0 + crop_w].unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                cx, w = (cx - x0 / width) * width / crop_w, w * width / crop_w
                cy, h = (cy - y0 / height) * height / crop_h, h * height / crop_h
            if float(torch.rand(())) < 0.5:
                image = image.flip(-1)
                cx = 1.0 - cx
        if not self.batch_photometric:
            image = _apply_photometric(self.photometric, image, self.low_light)
        return image, torch.tensor([cx, cy, w, h], dtype=image.dtype)


class ImageKPHeatmapDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        mesh: trimesh.Trimesh,
        keypoints: list[CanonicalKeypoint],
        margin: float = 1.4,
        heatmap_sigma: float = 2.0,
        device=None,
    ):
        """``device`` (a torch device or its name) runs the keypoint visibility ray casts; the
        default is CUDA if available."""
        self.df = data
        self.mesh = mesh
        self.margin = margin
        self.keypoints = keypoints
        self.heatmap_sigma = heatmap_sigma
        self.visibility = self._compute_visibility(device)

    def _keypoint_positions(self) -> np.ndarray:
        return np.stack([kp.position for kp in self.keypoints], axis=0).astype(np.float32)

    def _compute_visibility(self, device=None) -> np.ndarray:
        """Per-sample, per-keypoint visibility (in front, facing, unoccluded, projectable)."""
        points = self._keypoint_positions()
        normals = face_normals_at_points(self.mesh, points)
        occluder = MeshOccluder(self.mesh, device=device)
        visibility = np.zeros((len(self.df), len(points)), dtype=bool)
        rows = zip(self.df["pose"], self.df["distortion_coeffs"])
        for i, (pose, dist_coeffs) in enumerate(
            tqdm(rows, total=len(self.df), desc="Keypoint visibility", unit="sample")
        ):
            visibility[i] = keypoint_visibility(
                self.mesh, points, pose, occluder=occluder, normals=normals
            ) & projectable_mask(points, pose, dist_coeffs)
        logger.info(
            "Keypoints visible in %.1f%% of sample/keypoint pairs (before image bounds).",
            100.0 * visibility.mean() if visibility.size else 0.0,
        )
        return visibility

    def _load_pixels(self, row) -> np.ndarray:
        return read_pixels(row)

    def _build_kp_heatmaps(self, idx: int, row, box: Box, out_size: tuple[int, int]) -> torch.Tensor:
        """
        Gaussian targets at heatmap resolution (sigma in heatmap cells) for the keypoints
        that are visible and inside ``box`` once it is resized to ``out_size`` (width, height).
        """
        out_w, out_h = out_size
        height = -(-out_h // HEATMAP_STRIDE)
        width = -(-out_w // HEATMAP_STRIDE)
        image_points = project_points(
            self._keypoint_positions(),
            row["pose"],
            row["camera_intrinsics"],
            row["distortion_coeffs"],
        )
        cells = image_to_crop(image_points, box, out_size, stride=HEATMAP_STRIDE)
        inside = (
            (cells[:, 0] >= -0.5) & (cells[:, 0] <= width - 0.5) &
            (cells[:, 1] >= -0.5) & (cells[:, 1] <= height - 0.5)
        )

        heatmaps = torch.zeros(len(self.keypoints), height, width)
        for i in np.flatnonzero(self.visibility[idx] & inside):
            x, y = cells[i]
            heatmaps[i] = _gaussian_heatmap(height, width, float(x), float(y), self.heatmap_sigma)
        return heatmaps

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = self._load_pixels(row)
        height, width = pixels.shape[:2]
        heatmaps = self._build_kp_heatmaps(idx, row, (0, 0, width, height), (width, height))
        return _as_rgb(pixels_to_tensor(pixels)), heatmaps, self._rotation_target(row)

    @staticmethod
    def _rotation_target(row) -> torch.Tensor:
        return torch.from_numpy(allocentric_rotation(row["pose"]).astype(np.float32))


class ImageKPHeatmapROIDataset(ImageKPHeatmapDataset):
    def __getstate__(self):
        return {**self.__dict__, "_buffer": None}

    def __init__(
        self,
        data: pd.DataFrame,
        mesh: trimesh.Trimesh,
        keypoints: list[CanonicalKeypoint],
        crop_size: int = 256,
        margin: float = 1.4,
        heatmap_sigma: float = 2.0,
        augment: bool = False,
        low_light: float = 0.0,
        background=None,
        random_erasing: float = 0.0,
        symmetry=None,
        turn_weights: np.ndarray | None = None,
        regions=None,
        device=None,
    ):
        """
        ``regions`` (an ``image_cache.KeypointRegionCache`` of ``data``) supplies the image parts
        crops come from instead of decoding every frame each time: the same crops. ``device``
        runs the keypoint visibility ray casts (default: CUDA if available).

        ``background`` (a ``masks.BackgroundCompositor``) replaces the scene around the object in
        some augmented crops; ``random_erasing`` is the share of augmented crops that get a random
        rectangle of noise, so no single part of the object is relied on. With ``symmetry`` (a
        ``symmetry.KeypointSymmetry`` for keypoints in symmetric sets), each item also carries
        the keypoint labellings its crop may use, for ``SymmetricKeypointLoss``, and with
        ``turn_weights`` (one per row of ``data``, symmetry.twin_balance_weights) augmented items
        also carry their turn-loss weight; unaugmented (validation) items weigh 1.
        """
        super().__init__(data, mesh, keypoints, margin=margin, heatmap_sigma=heatmap_sigma, device=device)
        if regions is not None and len(regions.sizes) != len(data):
            raise ValueError("The keypoint region cache is for other frames.")
        self.regions = regions
        self._buffer: torch.Tensor | None = None
        self.crop_size = crop_size
        self.augment = augment
        self.low_light = low_light
        self.photometric = _photometric_augmentation(low_light)
        self.background = background
        self.symmetry = symmetry
        self.turn_weights = turn_weights
        self.erasing = (
            T.RandomErasing(p=random_erasing, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random")
            if random_erasing > 0 else None
        )
        # The same for whole batches on their device, for a trainer to apply in place of
        # ``photometric`` and ``erasing`` (see with_augmentation).
        self.batch_augmentation = BatchPhotometric(low_light=low_light, erasing=random_erasing)
        self.batch_photometric = False

    def with_augmentation(self, augment: bool, batch_photometric: bool = False) -> "ImageKPHeatmapROIDataset":
        """Shallow copy sharing the loaded data, with ROI jitter and colour/blur augmentation on or off.
        With ``batch_photometric``, the colour jitter, blur, low light and random erasing are left
        to ``batch_augmentation``, for the trainer to apply to whole batches."""
        dataset = copy.copy(self)
        dataset.augment = augment
        dataset.batch_photometric = batch_photometric
        return dataset

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels, origin = None, (0, 0)
        if self.regions is not None:
            pixels, origin, (height, width) = self.regions.get(idx)
        else:
            pixels = self._load_pixels(row)
            height, width = pixels.shape[:2]

        roi = _get_2d_roi(
            (height, width),
            self.mesh,
            row["pose"],
            row["camera_intrinsics"],
            row["distortion_coeffs"],
            self.margin,
        )
        if self.augment:
            roi = _jitter_roi(roi)
        box = crop_box(roi, height, width)
        x1, y1, x2, y2 = box
        ox, oy = origin
        if not (ox <= x1 and oy <= y1 and x2 <= ox + pixels.shape[1] and y2 <= oy + pixels.shape[0]):
            # Outside the cached region (not expected): decode the image as without a cache.
            pixels, (ox, oy) = self._load_pixels(row), (0, 0)
        # Only the box is converted to float (the same values as cropping the converted image),
        # into memory reused from sample to sample: a large box costs more to allocate afresh.
        box_pixels = pixels[y1 - oy:y2 - oy, x1 - ox:x2 - ox]
        channels = 1 if box_pixels.ndim == 2 else box_pixels.shape[2]
        size = channels * (y2 - y1) * (x2 - x1)
        if self._buffer is None or self._buffer.numel() < size:
            self._buffer = torch.empty(size, dtype=torch.float32)
        box_image = pixels_to_tensor(box_pixels, self._buffer[:size].view(channels, y2 - y1, x2 - x1))
        crop = crop_and_resize(box_image, (0, 0, x2 - x1, y2 - y1), self.crop_size)
        if self.augment:
            if self.background is not None:
                crop = self.background(crop, row, box)
            if not self.batch_photometric:
                crop = _apply_photometric(self.photometric, crop, self.low_light)
                if self.erasing is not None:
                    crop = self.erasing(crop)
        heatmaps = self._build_kp_heatmaps(idx, row, box, (self.crop_size, self.crop_size))
        rotation = self._rotation_target(row)
        if self.symmetry is not None:
            allowed = torch.from_numpy(self.symmetry.allowed_turns(rotation.numpy()))
            if self.symmetry.distinctness is None:
                return _as_rgb(crop), heatmaps, rotation, allowed
            pose = row["pose"]
            informativeness = self.symmetry.distinctness.informativeness(pose.quaternion.to_rotation_matrix(), pose.get_tvec())
            item = (_as_rgb(crop), heatmaps, rotation, allowed, torch.from_numpy(informativeness.astype(np.float32)))
            if self.turn_weights is not None:
                weight = float(self.turn_weights[idx]) if self.augment else 1.0
                item += (torch.tensor(weight, dtype=torch.float32),)
            return item
        return _as_rgb(crop), heatmaps, rotation


class DataLoader:
    """
    Recordings layout (the Astrobee datasets):
    - One mesh file at the dataset root, in mm, with the pose frame at its bbox centre.
    - Per recording: one CSV (poses), one JSON (K, D), and a ``*imgs`` folder.
      ``*imgs_undistorted`` is optional. When it is absent, ``image_undistorted`` is omitted.

    Subclasses read other layouts into the same ``data`` columns (id, recording, pose,
    camera_intrinsics, distortion_coeffs, image), so every dataset and trainer works on them.
    """

    FORMAT = "recordings"
    DEFAULT_MESH_SCALE = 0.001
    DEFAULT_RECENTER_MESH = True

    def __init__(self, path: Path, mesh_scale: float | None = None, recenter_mesh: bool | None = None):
        """``None`` uses the layout's defaults (DEFAULT_MESH_SCALE, DEFAULT_RECENTER_MESH)."""
        self.path = Path(path)
        self.mesh_scale = self.DEFAULT_MESH_SCALE if mesh_scale is None else mesh_scale
        self.recenter_mesh = self.DEFAULT_RECENTER_MESH if recenter_mesh is None else recenter_mesh
        self.data = self._load_data()
        self.mesh = self._load_mesh()

    def image_size(self) -> tuple[int, int]:
        """(width, height) of the first image; all images of a dataset share one camera."""
        with Image.open(self.data.iloc[0]["image"]) as image:
            return image.size

    def _find_mesh_file(self) -> Path:
        files = []
        for ext in trimesh.available_formats():
            files.extend(self.path.glob(f"*.{ext}"))
        assert len(files) == 1, f"Found {len(files)} mesh files in {self.path}; expected 1."
        return files[0]

    def _read_mesh(self, path: Path) -> trimesh.Trimesh:
        return trimesh.load_mesh(path)

    def _find_image(self, image_dir: Path, timestamp, suffix: str = "") -> list[Path]:
        return _find_image(image_dir, timestamp, suffix)

    def _load_mesh(self):
        mesh = self._read_mesh(self._find_mesh_file())
        if self.mesh_scale != 1.0:
            mesh.apply_scale(self.mesh_scale)
        if self.recenter_mesh:
            bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
            mesh.apply_translation(-bbox_center)
            logger.info("Recentered mesh by %s.", (-bbox_center).tolist())
        return mesh

    def _load_data(self):
        rows = []
        total_rows = 0
        for directory in self.path.glob("*/"):
            if not directory.is_dir():
                continue
            try:
                imgs = _find_distorted_image_dir(directory)
            except StopIteration as exc:
                logger.warning("Skipping %s: %s", directory.name, exc)
                continue
            imgs_undistorted = _find_undistorted_image_dir(directory)
            if imgs_undistorted is None:
                logger.info(
                    "No undistorted images in %s; skipping image_undistorted.",
                    directory.name,
                )

            json_files = list(directory.glob("*.json"))
            assert len(json_files) == 1, (
                f"Found {len(json_files)} json files in {directory}; expected 1."
            )
            with open(json_files[0], "r") as f:
                intrinsics = json.load(f)
            camera_intrinsics = np.array(intrinsics["K"], dtype=np.float32)
            distortion_coeffs = np.array(intrinsics["D"], dtype=np.float32)

            csv_files = list(directory.glob("*.csv"))
            assert len(csv_files) == 1, (
                f"Found {len(csv_files)} csv files in {directory}; expected 1."
            )
            pose_file = pd.read_csv(csv_files[0])

            for _, row in pose_file.iterrows():
                total_rows += 1
                pose = Pose(
                    row["x"], row["y"], row["z"],
                    row["x.1"], row["y.1"], row["z.1"], row["w"],
                )

                img_files = self._find_image(imgs, row["Timestamp"])
                if len(img_files) != 1:
                    logger.warning(
                        "Skipping row in %s at timestamp %s: found %d distorted image(s).",
                        directory.name,
                        row["Timestamp"],
                        len(img_files),
                    )
                    continue

                sample = {
                    "id": len(rows),
                    "recording": directory.name,
                    "timestamp": float(row["Timestamp"]),
                    "pose": pose,
                    "camera_intrinsics": camera_intrinsics,
                    "distortion_coeffs": distortion_coeffs,
                    "image": img_files[0],
                }
                if imgs_undistorted is not None:
                    img_undistorted_files = _find_image(
                        imgs_undistorted, row["Timestamp"], "_undistorted"
                    )
                    if len(img_undistorted_files) == 1:
                        sample["image_undistorted"] = img_undistorted_files[0]
                    else:
                        logger.warning(
                            "No undistorted image in %s at timestamp %s.",
                            directory.name,
                            row["Timestamp"],
                        )
                rows.append(sample)

        logger.info(
            "Loaded %d samples from %d CSV rows in %s.",
            len(rows),
            total_rows,
            self.path,
        )
        return pd.DataFrame(rows)

    def split_by_recording(
        self,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        test_fraction: float = DEFAULT_TEST_FRACTION,
        seed: int = DEFAULT_SPLIT_SEED,
    ) -> "RecordingSplit":
        return split_by_recording(self.data, val_fraction, test_fraction, seed)

    def exclude_recordings(self, patterns: list[str]) -> list[str]:
        """Drop the recordings whose name matches any shell-style pattern; returns their names."""
        names = sorted(self.data["recording"].unique())
        excluded = [name for name in names if any(fnmatch.fnmatchcase(name, p) for p in patterns)]
        if excluded:
            self.data = self.data[~self.data["recording"].isin(excluded)].reset_index(drop=True)
            logger.info("Excluded %d recordings (%s); %d frames remain.", len(excluded), ", ".join(patterns), len(self.data))
        return excluded

    def get_pose_dataset(self):
        if "image_undistorted" not in self.data.columns:
            raise ValueError("Dataset has no undistorted images.")
        return ImagePoseDataset(self.data)

    def get_roi_dataset(
        self,
        margin: float = 1.4,
        input_size: tuple[int, int] = ROI_INPUT_SIZE,
        low_light: float = 0.0,
        cache=None,
    ):
        return ImageROIDataset(self.data, self.mesh, margin=margin, input_size=input_size, low_light=low_light, cache=cache)

    def get_kp_heatmap_dataset(self, keypoints: list[CanonicalKeypoint], margin: float = 1.4, device=None):
        return ImageKPHeatmapDataset(self.data, self.mesh, keypoints, margin=margin, device=device)

    def get_kp_heatmap_roi_dataset(
        self,
        keypoints: list[CanonicalKeypoint],
        crop_size: int = 256,
        margin: float = 1.4,
        low_light: float = 0.0,
        background=None,
        random_erasing: float = 0.0,
        symmetry=None,
        turn_weights: np.ndarray | None = None,
        regions=None,
        device=None,
    ):
        return ImageKPHeatmapROIDataset(
            self.data,
            self.mesh,
            keypoints,
            crop_size=crop_size,
            margin=margin,
            low_light=low_light,
            background=background,
            random_erasing=random_erasing,
            symmetry=symmetry,
            turn_weights=turn_weights,
            regions=regions,
            device=device,
        )


class EO1DataLoader(DataLoader):
    """
    Single-folder layout of independent renders (the Polimi EO-1 dataset):
    - one glTF/GLB mesh at the root, in metres, with the pose frame at the model origin,
    - ``camera_intrinsics.json`` with K (pinhole, no distortion),
    - ``pose_labels.json`` with, per frame: timestamp k, class, visibility, translation
      and rotation (``x_C = R_CT X_T + t``),
    - ``dataset/main_camera_<k>.000.png``.

    Frames are independent random poses, so for splitting they are grouped into blocks of
    ``FRAMES_PER_BLOCK`` consecutive frames (stored as the ``recording`` column); that keeps
    the saved split short without letting near-duplicates leak between splits.
    """

    FORMAT = "eo1"
    DEFAULT_MESH_SCALE = 1.0
    DEFAULT_RECENTER_MESH = False
    LABELS_FILE = "pose_labels.json"
    INTRINSICS_FILE = "camera_intrinsics.json"
    IMAGE_DIR = "dataset"
    FRAMES_PER_BLOCK = 50
    # glTF stores meshes Y-up; the labels use the model's Z-up frame: (x, y, z) -> (x, -z, y).
    # Checked against the renders: this axis map and the label quaternion convention below
    # are the only combination whose projected silhouette matches the target (IoU 0.77 against
    # the lit pixels, next best 0.44).
    GLTF_TO_MODEL_FRAME = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    @classmethod
    def matches(cls, path: Path) -> bool:
        path = Path(path)
        return (path / cls.LABELS_FILE).is_file() and (path / cls.IMAGE_DIR).is_dir()

    def _read_mesh(self, path: Path) -> trimesh.Trimesh:
        loaded = trimesh.load(path)
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
        if path.suffix.lower() in (".glb", ".gltf"):
            mesh.apply_transform(self.GLTF_TO_MODEL_FRAME)
        return mesh

    @staticmethod
    def _png_complete(path: Path) -> bool:
        """A PNG written to the end finishes with an IEND chunk; truncated files don't."""
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - 12))
            return b"IEND" in f.read()

    @staticmethod
    def _label_pose(label: dict) -> Pose:
        # The labels give q_CT scalar-last with R_CT = quat2dcm(q) (MATLAB Aerospace), which is
        # the transpose of scipy's from_quat(q).as_matrix(); Pose expects scipy's convention.
        q = Rotation.from_quat([label["rotation"][axis] for axis in "xyzw"]).inv().as_quat()
        t = label["translation"]
        return Pose(t["x"], t["y"], t["z"], q[0], q[1], q[2], q[3])

    def _load_data(self):
        with open(self.path / self.INTRINSICS_FILE) as f:
            intrinsics = json.load(f)
        camera_intrinsics = np.array(intrinsics["K"], dtype=np.float32)
        distortion_coeffs = np.array(intrinsics.get("D", np.zeros(5)), dtype=np.float32)
        with open(self.path / self.LABELS_FILE) as f:
            labels = json.load(f)["poses"]

        images = {}
        for image_path in (self.path / self.IMAGE_DIR).glob("*.png"):
            match = re.fullmatch(r".*_(\d+(?:\.\d+)?)", image_path.stem)
            if match:
                images[round(float(match.group(1)), 3)] = image_path

        rows = []
        for label in labels:
            timestamp = round(float(label["timestamp"]), 3)
            image_path = images.get(timestamp)
            if image_path is None:
                logger.warning("Skipping frame %s: no image in %s.", timestamp, self.path / self.IMAGE_DIR)
                continue
            if not self._png_complete(image_path):
                logger.warning("Skipping frame %s: %s is truncated.", timestamp, image_path.name)
                continue
            rows.append({
                "id": len(rows),
                "recording": f"block_{int(timestamp) // self.FRAMES_PER_BLOCK:03d}",
                "pose": self._label_pose(label),
                "camera_intrinsics": camera_intrinsics,
                "distortion_coeffs": distortion_coeffs,
                "image": image_path,
                "class": label.get("class"),
                "visibility": label.get("visibility"),
            })
        logger.info("Loaded %d samples from %d labels in %s.", len(rows), len(labels), self.path)
        return pd.DataFrame(rows)


class GodotDataLoader(DataLoader):
    """
    Recordings layout rendered in Godot (the merged Astrobee ``full_dataset``):
    - one glTF/GLB mesh at the root, in metres, read in the Z-up model frame the poses refer to
      (glTF's Y-up axes converted as for EO-1) with the pose frame at its bounding-box centre,
    - per recording a ``_godot_imgs`` folder, one CSV and one JSON as in ``DataLoader``:
      ``rec_run_*`` images are named after their time stamp (``1088_2.png``), numbered sets
      (``164_*``, ``644_*``, ...) after their frame (``0000000040.jpg`` is the row with Timestamp 40),
    - ``*_bayer`` recordings store the raw BG sensor mosaic; ``read_image`` converts it to grayscale.
    """

    FORMAT = "godot"
    DEFAULT_MESH_SCALE = 1.0
    DEFAULT_RECENTER_MESH = True
    IMAGE_DIR = "_godot_imgs"
    MESH_SUFFIXES = (".glb", ".gltf")
    BAYER_SUFFIX = "_bayer"
    BAYER_PATTERN = "BG"

    @classmethod
    def matches(cls, path: Path) -> bool:
        path = Path(path)
        if not path.is_dir():
            return False
        entries = list(path.iterdir())
        has_gltf = any(p.is_file() and p.suffix.lower() in cls.MESH_SUFFIXES for p in entries)
        return has_gltf and any((p / cls.IMAGE_DIR).is_dir() for p in entries if p.is_dir())

    def _read_mesh(self, path: Path) -> trimesh.Trimesh:
        # Geometry only: the materials' textures would add hundreds of MB.
        mesh = trimesh.load(path, force="mesh", skip_materials=True)
        if path.suffix.lower() in self.MESH_SUFFIXES:
            mesh.apply_transform(EO1DataLoader.GLTF_TO_MODEL_FRAME)
        return mesh

    def _find_image(self, image_dir: Path, timestamp, suffix: str = "") -> list[Path]:
        stamp = Decimal(str(timestamp))
        if stamp == stamp.to_integral_value():
            for extension in (".jpg", ".png"):
                path = image_dir / f"{int(stamp):010d}{suffix}{extension}"
                if path.exists():
                    return [path]
        return super()._find_image(image_dir, timestamp, suffix)

    def _load_data(self):
        data = super()._load_data()
        if data.empty:
            return data
        bayer = data["recording"].str.endswith(self.BAYER_SUFFIX)
        data["bayer_pattern"] = np.where(bayer, self.BAYER_PATTERN, None)
        logger.info(
            "%d samples in %d raw Bayer recordings are converted to grayscale when read.",
            int(bayer.sum()),
            data.loc[bayer, "recording"].nunique(),
        )
        return data


DATASET_LOADERS: dict[str, type[DataLoader]] = {
    DataLoader.FORMAT: DataLoader,
    EO1DataLoader.FORMAT: EO1DataLoader,
    GodotDataLoader.FORMAT: GodotDataLoader,
}


def open_dataset(
    path: Path,
    dataset_format: str | None = None,
    mesh_scale: float | None = None,
    recenter_mesh: bool | None = None,
) -> DataLoader:
    """Load a dataset with the loader for its layout (detected when ``dataset_format`` is None)."""
    if dataset_format is None:
        dataset_format = next(
            (cls.FORMAT for cls in (EO1DataLoader, GodotDataLoader) if cls.matches(path)),
            DataLoader.FORMAT,
        )
    loader_cls = DATASET_LOADERS[dataset_format]
    logger.info("Reading %s as the %r layout.", path, dataset_format)
    return loader_cls(Path(path), mesh_scale=mesh_scale, recenter_mesh=recenter_mesh)
