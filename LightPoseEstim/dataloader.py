import copy
import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import trimesh
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .crop import HEATMAP_STRIDE, Box, crop_and_resize, crop_box, image_to_crop
from .pose import Pose
from .projection import (
    MeshOccluder,
    camera_frame_points,
    face_normals_at_points,
    keypoint_visibility,
    mesh_bbox_corners,
    project_points,
    projectable_mask,
)
from .roi import BBox, ROI, bbox_to_roi, normalize_roi
from .keypoint_selector import CanonicalKeypoint

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
    bbox_3d = mesh_bbox_corners(mesh)

    camera_points = camera_frame_points(bbox_3d, pose)
    in_front_mask = camera_points[:, 2] > 1e-6
    if int(np.sum(in_front_mask)) < 2:
        return ROI(0.0, 0.0, 0.0, 0.0)

    projectable = projectable_mask(bbox_3d, pose, dist_coeffs)
    image_points = [np.empty((0, 2), dtype=np.float32)]
    if projectable.any():
        image_points.append(
            project_points(bbox_3d[projectable], pose, camera_intrinsics, dist_coeffs)
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


def _jitter_roi(roi: ROI, shift: float = 0.1, scale: float = 0.15) -> ROI:
    """Random shift and scale, so training crops look like crops from a predicted ROI."""
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
    def __init__(self, data: pd.DataFrame, mesh: trimesh.Trimesh, margin: float = 1.4):
        self.df = data
        self.mesh = mesh
        self.margin = margin
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image"]).convert("RGB")
        image = self.transform(image)

        roi = _get_2d_roi(
            image.shape,
            self.mesh,
            row["pose"],
            row["camera_intrinsics"],
            row["distortion_coeffs"],
            self.margin,
        )

        return image, normalize_roi(
            image.unsqueeze(0),
            torch.tensor(
                [roi.cx, roi.cy, roi.w, roi.h],
                dtype=image.dtype,
                device=image.device,
            ),
        )


class ImageKPHeatmapDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        mesh: trimesh.Trimesh,
        keypoints: list[CanonicalKeypoint],
        margin: float = 1.4,
        heatmap_sigma: float = 2.0,
    ):
        self.df = data
        self.mesh = mesh
        self.margin = margin
        self.keypoints = keypoints
        self.heatmap_sigma = heatmap_sigma
        self.transform = T.Compose([T.ToTensor()])
        self.visibility = self._compute_visibility()

    def _keypoint_positions(self) -> np.ndarray:
        return np.stack([kp.position for kp in self.keypoints], axis=0).astype(np.float32)

    def _compute_visibility(self) -> np.ndarray:
        """Per-sample, per-keypoint visibility (in front, facing, unoccluded, projectable)."""
        points = self._keypoint_positions()
        normals = face_normals_at_points(self.mesh, points)
        occluder = MeshOccluder(self.mesh)
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

    def _load_image(self, row) -> torch.Tensor:
        image = Image.open(row["image"]).convert("RGB")
        return self.transform(image)

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
        image = self._load_image(row)
        height, width = image.shape[-2:]
        heatmaps = self._build_kp_heatmaps(idx, row, (0, 0, width, height), (width, height))
        return image, heatmaps


class ImageKPHeatmapROIDataset(ImageKPHeatmapDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        mesh: trimesh.Trimesh,
        keypoints: list[CanonicalKeypoint],
        crop_size: int = 256,
        margin: float = 1.4,
        heatmap_sigma: float = 2.0,
        augment: bool = False,
    ):
        super().__init__(data, mesh, keypoints, margin=margin, heatmap_sigma=heatmap_sigma)
        self.crop_size = crop_size
        self.augment = augment
        self.photometric = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
        ])

    def with_augmentation(self, augment: bool) -> "ImageKPHeatmapROIDataset":
        """Shallow copy sharing the loaded data, with ROI jitter and colour/blur augmentation on or off."""
        dataset = copy.copy(self)
        dataset.augment = augment
        return dataset

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._load_image(row)
        height, width = image.shape[-2:]

        roi = _get_2d_roi(
            image.shape,
            self.mesh,
            row["pose"],
            row["camera_intrinsics"],
            row["distortion_coeffs"],
            self.margin,
        )
        if self.augment:
            roi = _jitter_roi(roi)
        box = crop_box(roi, height, width)
        crop = crop_and_resize(image, box, self.crop_size)
        if self.augment:
            crop = self.photometric(crop)
        heatmaps = self._build_kp_heatmaps(idx, row, box, (self.crop_size, self.crop_size))
        return crop, heatmaps


class DataLoader:
    def __init__(self, path: Path, mesh_scale: float = 1.0, recenter_mesh: bool = False):
        """
        Load a pose-estimation dataset from a directory.

        Expected layout:
        - One mesh file at the dataset root.
        - Per recording: one CSV (poses), one JSON (K, D), and a ``*imgs`` folder.
          ``*imgs_undistorted`` is optional. When it is absent, ``image_undistorted`` is omitted.
        """
        self.path = path
        self.mesh_scale = mesh_scale
        self.recenter_mesh = recenter_mesh
        self.data = self._load_data()
        self.mesh = self._load_mesh()

    def _load_mesh(self):
        files = []
        for ext in trimesh.available_formats():
            files.extend(self.path.glob(f"*.{ext}"))
        assert len(files) == 1, f"Found {len(files)} mesh files in {self.path}; expected 1."
        mesh = trimesh.load_mesh(files[0])
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

                img_files = _find_image(imgs, row["Timestamp"])
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

    def get_pose_dataset(self):
        if "image_undistorted" not in self.data.columns:
            raise ValueError("Dataset has no undistorted images.")
        return ImagePoseDataset(self.data)

    def get_roi_dataset(self, margin: float = 1.4):
        return ImageROIDataset(self.data, self.mesh, margin=margin)

    def get_kp_heatmap_dataset(self, keypoints: list[CanonicalKeypoint], margin: float = 1.4):
        return ImageKPHeatmapDataset(self.data, self.mesh, keypoints, margin=margin)

    def get_kp_heatmap_roi_dataset(
        self,
        keypoints: list[CanonicalKeypoint],
        crop_size: int = 256,
        margin: float = 1.4,
    ):
        return ImageKPHeatmapROIDataset(
            self.data,
            self.mesh,
            keypoints,
            crop_size=crop_size,
            margin=margin,
        )
