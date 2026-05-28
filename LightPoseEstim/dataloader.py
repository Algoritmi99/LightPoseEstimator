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

from LightPoseEstim.pose import Pose
from LightPoseEstim.roi import BBox, bbox_to_roi, normalize_roi

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

def _get_2d_roi(img_shape: tuple,
                mesh: trimesh.Trimesh,
                pose: Pose,
                camera_intrinsics: np.ndarray,
                dist_coeffs: np.ndarray | None = None,
                margin: float = 1.2
                ):
    height, width = img_shape[-2:]
    bounds = mesh.bounds
    xmin, ymin, zmin = bounds[0]
    xmax, ymax, zmax = bounds[1]

    bbox_3d = np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],

        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
    ], dtype=np.float32)

    dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

    image_points, _ = cv2.projectPoints(
        bbox_3d,
        pose.get_rvec(),
        pose.get_tvec(),
        camera_intrinsics,
        dist_coeffs,
    )
    image_points = image_points.squeeze(1)

    x1 = image_points[:, 0].min()
    x2 = image_points[:, 0].max()

    y1 = image_points[:, 1].min()
    y2 = image_points[:, 1].max()

    x1 = float(np.clip(x1, 0, width - 1))
    x2 = float(np.clip(x2, 0, width - 1))

    y1 = float(np.clip(y1, 0, height - 1))
    y2 = float(np.clip(y2, 0, height - 1))

    bbox = BBox(x1, y1, x2, y2)
    roi = bbox_to_roi(bbox)
    roi.w *= margin
    roi.h *= margin
    return roi


class ImagePoseDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        undist_image = Image.open(row["image_undistorted"]).convert("RGB")
        undist_image = self.transform(undist_image)

        pose = row["pose"]

        return undist_image, pose


class ImageROIDataset(Dataset):
    def __init__(self, data: pd.DataFrame,
                 mesh: trimesh.Trimesh,
                 camera_intrinsics: np.ndarray,
                 dist_coeffs: np.ndarray | None = None,
                 margin: float = 1.2):
        self.df = data
        self.mesh = mesh
        self.camera_intrinsics = camera_intrinsics
        self.dist_coeffs = dist_coeffs
        self.margin = margin
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        undist_image = Image.open(row["image_undistorted"]).convert("RGB")
        undist_image = self.transform(undist_image)

        pose = row["pose"]
        roi = _get_2d_roi(
            undist_image.shape,
            self.mesh,
            pose,
            self.camera_intrinsics,
            self.dist_coeffs,
            self.margin
        )

        return undist_image, normalize_roi(
            undist_image, torch.tensor(
                [roi.cx, roi.cy, roi.w, roi.h],
                dtype=undist_image.dtype,
                device=undist_image.device
            )
        )


class DataLoader:
    def __init__(self, path: Path):
        """
        Loads a Pose Estimation dataset from a directory.
        :param path: path to the raw dataset directory with the expected structure:
        The root directory should contain a single mesh file.
        Each directory in the root directory should contain a single csv file that contains the pose data
        and a single json file that contains the camera intrinsics.
        Each directory in the root directory should contain a directory named matching "*imgs" and "*imgs_undistorted"
        The images should be named in a way that starts with the index of the pose in the csv file.
        """
        self.path = path
        self.data = self._load_data()
        self.mesh = self._load_mesh()
        self.camera_intrinsics = self._load_camera_intrinsics()

    def _load_camera_intrinsics(self):
        json_files = list(self.path.glob("*.json"))

        assert len(json_files) == 1, (
            f"Found {len(json_files)} json files in {self.path}. Expected 1."
        )

        with open(json_files[0], "r") as f:
            data = json.load(f)

        return np.array(data["K"], dtype=np.float32)

    def _load_mesh(self):
        files = []
        for ext in trimesh.available_formats():
            files.extend(self.path.glob(f"*.{ext}"))
        assert len(files) == 1, f"Found {len(files)} mesh files in {self.path}; expected 1."
        return trimesh.load_mesh(files[0])

    def _load_data(self):
        rows = []
        c = 0
        for directory in self.path.glob("*/"):
            try:
                imgs = next(directory.glob("*imgs"))
                imgs_undistorted = next(directory.glob("*imgs_undistorted"))
            except StopIteration:
                logger.warning(
                    "Skipping directory %s because it does not contain imgs or imgs_undistorted.",
                    directory.name,
                )
                continue


            csv_files = list(directory.glob("*.csv"))
            assert len(csv_files) == 1, (
                f"Found {len(csv_files)} csv files in {directory}; expected 1."
            )
            pose_file = pd.read_csv(csv_files[0])

            for idx, row in pose_file.iterrows():
                c += 1
                pose = Pose(row["x"], row["y"], row["z"], row["x.1"], row["y.1"], row["z.1"], row["w"])

                img_files = _find_image(imgs, row["Timestamp"])
                img_undistorted_files = _find_image(imgs_undistorted, row["Timestamp"], "_undistorted")

                if len(img_files) != 1:
                    logger.warning(
                        "Skipping row in %s at timestamp %s: found %d image(s) in %s; expected 1.",
                        directory.name,
                        row["Timestamp"],
                        len(img_files),
                        imgs,
                    )
                    continue
                if len(img_undistorted_files) != 1:
                    logger.warning(
                        "Skipping row in %s at timestamp %s: found %d undistorted image(s) in %s; expected 1.",
                        directory.name,
                        row["Timestamp"],
                        len(img_undistorted_files),
                        imgs_undistorted,
                    )
                    continue

                rows.append({
                    "id": len(rows),
                    "pose": pose,
                    "image": img_files[0],
                    "image_undistorted": img_undistorted_files[0],
                })

        logger.info("Loaded %d data points from a total of %d rows in %s.", len(rows), c, self.path)
        return pd.DataFrame(rows)

    def get_pose_dataset(self):
        return ImagePoseDataset(self.data)

    def get_roi_dataset(self):
        return ImageROIDataset(self.data, self.mesh, self.camera_intrinsics)
