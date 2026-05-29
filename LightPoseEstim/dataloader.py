import json
import logging
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import trimesh
from PIL import Image
from torch.utils.data import Dataset

from .pose import Pose
from .projection import mesh_bbox_corners, project_points
from .roi import BBox, ROI, bbox_to_roi, normalize_roi

logger = logging.getLogger(__name__)


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


def _find_undistorted_image_dir(directory: Path) -> Path:
    candidates = sorted(directory.glob("*imgs_undistorted"))
    if not candidates:
        raise StopIteration(f"No undistorted image directory in {directory}")
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

    rot_matrix, _ = cv2.Rodrigues(pose.get_rvec())
    camera_points = (rot_matrix @ bbox_3d.T + pose.get_tvec().reshape(3, 1)).T
    in_front_mask = camera_points[:, 2] > 1e-6
    if int(np.sum(in_front_mask)) < 2:
        logger.warning(
            "Object is behind the camera; returning zero ROI. tvec=%s",
            pose.get_tvec().ravel().tolist(),
        )
        return ROI(0.0, 0.0, 0.0, 0.0)

    image_points = project_points(
        bbox_3d[in_front_mask],
        pose,
        camera_intrinsics,
        dist_coeffs,
    )

    x1 = float(np.clip(image_points[:, 0].min(), 0, width - 1))
    x2 = float(np.clip(image_points[:, 0].max(), 0, width - 1))
    y1 = float(np.clip(image_points[:, 1].min(), 0, height - 1))
    y2 = float(np.clip(image_points[:, 1].max(), 0, height - 1))

    roi = bbox_to_roi(BBox(x1, y1, x2, y2))
    roi.w *= margin
    roi.h *= margin
    return roi


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


class DataLoader:
    def __init__(self, path: Path, mesh_scale: float = 1.0, recenter_mesh: bool = False):
        """
        Load a pose-estimation dataset from a directory.

        Expected layout:
        - One mesh file at the dataset root.
        - Per recording: one CSV (poses), one JSON (K, D), ``*imgs`` and ``*_undistorted`` folders.
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
                imgs_undistorted = _find_undistorted_image_dir(directory)
            except StopIteration as exc:
                logger.warning("Skipping %s: %s", directory.name, exc)
                continue

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
                img_undistorted_files = _find_image(
                    imgs_undistorted, row["Timestamp"], "_undistorted"
                )

                if len(img_files) != 1:
                    logger.warning(
                        "Skipping row in %s at timestamp %s: found %d distorted image(s).",
                        directory.name,
                        row["Timestamp"],
                        len(img_files),
                    )
                    continue
                if len(img_undistorted_files) != 1:
                    logger.warning(
                        "Skipping row in %s at timestamp %s: found %d undistorted image(s).",
                        directory.name,
                        row["Timestamp"],
                        len(img_undistorted_files),
                    )
                    continue

                rows.append({
                    "id": len(rows),
                    "pose": pose,
                    "camera_intrinsics": camera_intrinsics,
                    "distortion_coeffs": distortion_coeffs,
                    "image": img_files[0],
                    "image_undistorted": img_undistorted_files[0],
                })

        logger.info(
            "Loaded %d samples from %d CSV rows in %s.",
            len(rows),
            total_rows,
            self.path,
        )
        return pd.DataFrame(rows)

    def get_pose_dataset(self):
        return ImagePoseDataset(self.data)

    def get_roi_dataset(self, margin: float = 1.4):
        return ImageROIDataset(self.data, self.mesh, margin=margin)
