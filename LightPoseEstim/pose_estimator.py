from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision.transforms import ToTensor

from .keypoint_selector import (
    CanonicalKeypoint,
    load_canonical_keypoints,
    paths_from_keypoint_run_dir,
)
from .kp.keypoint_heatmap import KeypointHeatmapNet
from .pose import Pose
from .projection import as_dist_coeffs
from .roi import ROI, roi_to_bbox
from .roi.roi_detector import ROIDetector


@dataclass
class PoseEstimationResult:
    pose: Pose | None
    roi: ROI
    image_points: np.ndarray
    confidence: np.ndarray


class PoseEstimator:
    def __init__(
        self,
        roi_detector: ROIDetector,
        keypoint_net: KeypointHeatmapNet,
        keypoints: list[CanonicalKeypoint],
        camera_intrinsics: np.ndarray,
        distortion_coeffs: np.ndarray | None = None,
        crop_size: int = 256,
        peak_threshold: float = 0.1,
        device: torch.device | None = None,
    ):
        self.roi_detector = roi_detector
        self.keypoint_net = keypoint_net
        self.keypoints = keypoints
        self.object_points = np.stack(
            [kp.position for kp in keypoints], axis=0
        ).astype(np.float32)
        self.camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float32)
        self.distortion_coeffs = distortion_coeffs
        self.crop_size = crop_size
        self.peak_threshold = peak_threshold
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.to_tensor = ToTensor()
        self.roi_detector.to(self.device).eval()
        self.keypoint_net.to(self.device).eval()

    @classmethod
    def load(
        cls,
        roi_detector_path: str | Path,
        camera_intrinsics: np.ndarray,
        keypoint_run_dir: str | Path | None = None,
        keypoint_net_path: str | Path | None = None,
        keypoints: list[CanonicalKeypoint] | None = None,
        keypoints_path: str | Path | None = None,
        distortion_coeffs: np.ndarray | None = None,
        crop_size: int = 256,
        peak_threshold: float = 0.1,
        device: torch.device | None = None,
    ) -> "PoseEstimator":
        if keypoint_run_dir is not None:
            if keypoint_net_path is not None or keypoints_path is not None:
                raise ValueError(
                    "Provide keypoint_run_dir or explicit net/keypoints paths, not both."
                )
            run_paths = paths_from_keypoint_run_dir(keypoint_run_dir)
            keypoint_net_path = run_paths.net_checkpoint
            keypoints_path = run_paths.keypoints
        elif keypoint_net_path is None:
            raise ValueError("Provide keypoint_run_dir or keypoint_net_path.")

        if keypoints is None:
            if keypoints_path is None:
                raise ValueError("Provide keypoints or keypoints_path.")
            keypoints = load_canonical_keypoints(keypoints_path)
        elif keypoints_path is not None:
            raise ValueError("Provide only one of keypoints or keypoints_path.")
        return cls(
            roi_detector=ROIDetector.load(roi_detector_path),
            keypoint_net=KeypointHeatmapNet.load(
                keypoint_net_path,
                num_keypoints=len(keypoints),
            ),
            keypoints=keypoints,
            camera_intrinsics=camera_intrinsics,
            distortion_coeffs=distortion_coeffs,
            crop_size=crop_size,
            peak_threshold=peak_threshold,
            device=device,
        )

    def estimate(self, image) -> PoseEstimationResult | list[PoseEstimationResult]:
        batch = self._to_batch(image)
        results = []
        for sample in batch:
            results.append(self._estimate_one(sample))
        if len(results) == 1 and not isinstance(image, (list, tuple)):
            if not (
                isinstance(image, torch.Tensor)
                and image.ndim == 4
                and image.shape[0] > 1
            ):
                return results[0]
        return results

    def _estimate_one(self, image: torch.Tensor) -> PoseEstimationResult:
        image = image.to(self.device)
        roi = self.roi_detector.inference(image.unsqueeze(0))[0]
        crop, crop_box = self._crop_and_resize(image, roi, self.crop_size)

        coordinates, confidence = self.keypoint_net.inference(
            crop.unsqueeze(0),
            threshold=self.peak_threshold,
        )
        image_points = self._map_to_full_image(
            coordinates[0].cpu().numpy(),
            crop_box,
            self.crop_size,
        )
        confidence = confidence[0].cpu().numpy()
        pose = self._solve_pose(image_points, confidence)
        return PoseEstimationResult(
            pose=pose,
            roi=roi,
            image_points=image_points,
            confidence=confidence,
        )

    def _solve_pose(
        self,
        image_points: np.ndarray,
        confidence: np.ndarray,
    ) -> Pose | None:
        valid = confidence >= self.peak_threshold
        if int(valid.sum()) < 4:
            return None

        object_points = self.object_points[valid]
        points_2d = image_points[valid].reshape(-1, 1, 2).astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            points_2d,
            self.camera_intrinsics,
            as_dist_coeffs(self.distortion_coeffs),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = Rotation.from_matrix(rotation).as_quat()
        t = tvec.reshape(3)
        return Pose(float(t[0]), float(t[1]), float(t[2]), qx, qy, qz, qw)

    @staticmethod
    def _crop_and_resize(
        image: torch.Tensor,
        roi: ROI,
        crop_size: int,
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        _, height, width = image.shape
        bbox = roi_to_bbox(roi)
        x1 = int(max(0, np.floor(bbox.x1)))
        y1 = int(max(0, np.floor(bbox.y1)))
        x2 = int(min(width, np.ceil(bbox.x2)))
        y2 = int(min(height, np.ceil(bbox.y2)))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)

        crop = image[:, y1:y2, x1:x2]
        crop = F.interpolate(
            crop.unsqueeze(0),
            size=(crop_size, crop_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return crop, (x1, y1, x2, y2)

    @staticmethod
    def _map_to_full_image(
        coordinates: np.ndarray,
        crop_box: tuple[int, int, int, int],
        crop_size: int,
    ) -> np.ndarray:
        x1, y1, x2, y2 = crop_box
        crop_w = max(x2 - x1, 1)
        crop_h = max(y2 - y1, 1)
        scale_x = crop_w / crop_size
        scale_y = crop_h / crop_size
        x = x1 + coordinates[:, 0] * scale_x
        y = y1 + coordinates[:, 1] * scale_y
        return np.stack([x, y], axis=1).astype(np.float32)

    def _to_batch(self, image) -> list[torch.Tensor]:
        if isinstance(image, (list, tuple)):
            return [self._to_tensor(item) for item in image]
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            return [sample for sample in image]
        return [self._to_tensor(image)]

    def _to_tensor(self, image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                return image.float()
            raise ValueError("Expected image tensor with shape [C, H, W].")
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in (1, 3):
                return torch.from_numpy(image).float()
            image = Image.fromarray(image)
        return self.to_tensor(image)
