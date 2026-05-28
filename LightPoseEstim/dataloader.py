import logging
from pathlib import Path
from decimal import Decimal
from PIL import Image

import trimesh
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T

from LightPoseEstim.pose import Pose

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


class DataLoader:
    def __init__(self, path: Path):
        """
        Loads a Pose Estimation dataset from a directory.
        :param path: path to the raw dataset directory with the expected structure:
        The root directory should contain a single mesh file.
        Each directory in the root directory should contain a single csv file that contains the pose data.
        Each directory in the root directory should contain a directory named matching "*imgs" and "*imgs_undistorted"
        The images should be named in a way that starts with the index of the pose in the csv file.
        """
        self.path = path
        self.data = self._load_data()
        self.mesh = self._load_mesh()

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
