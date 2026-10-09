import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision.transforms import ToTensor

from .keypoint_selector import (
    CanonicalKeypoint,
    load_canonical_keypoints,
    paths_from_keypoint_run_dir,
)
from .crop import crop_and_resize, crop_box, crop_to_image
from .kp.keypoint_heatmap import DEFAULT_PEAK_THRESHOLD, KeypointHeatmapNet
from .pose import Pose
from .projection import as_dist_coeffs, project_points, projectable_mask
from .roi import ROI, roi_to_bbox
from .roi.roi_detector import ROIDetector

ROI_CHECKPOINT_RELATIVE = Path("roi") / "roi_detector.pth"
RUN_CONFIG_RELATIVE = Path("config.json")

# PnP outlier threshold: this many heatmap cells, mapped to full-image pixels.
_REPROJECTION_ERROR_CELLS = 2.0
_MIN_REPROJECTION_ERROR_PX = 4.0
# 4 points always fit some pose, so a 5th is needed to reject outliers at all.
_MIN_PNP_INLIERS = 5


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
        peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
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
        peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
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

    @classmethod
    def load_run(
        cls,
        run_dir: str | Path,
        camera_intrinsics: np.ndarray,
        distortion_coeffs: np.ndarray | None = None,
        peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
        device: torch.device | None = None,
    ) -> "PoseEstimator":
        run_dir = Path(run_dir)
        with (run_dir / RUN_CONFIG_RELATIVE).open() as handle:
            config = json.load(handle)
        return cls.load(
            roi_detector_path=run_dir / ROI_CHECKPOINT_RELATIVE,
            camera_intrinsics=camera_intrinsics,
            keypoint_run_dir=run_dir,
            distortion_coeffs=distortion_coeffs,
            crop_size=int(config["crop_size"]),
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
        _, height, width = image.shape
        roi = self.roi_detector.inference(image.unsqueeze(0))[0]
        if self._roi_outside_frame(roi, height, width):
            return self._discarded_pose(roi)

        box = crop_box(roi, height, width)
        crop = crop_and_resize(image, box, self.crop_size)
        cells, confidence = self.keypoint_net.inference(
            crop.unsqueeze(0),
            threshold=self.peak_threshold,
        )
        stride = self.keypoint_net.stride
        image_points = crop_to_image(
            cells[0].cpu().numpy(),
            box,
            (self.crop_size, self.crop_size),
            stride=stride,
        ).astype(np.float32)
        confidence = confidence[0].cpu().numpy()
        cell_px = stride * max(box[2] - box[0], box[3] - box[1]) / self.crop_size
        max_error = max(_MIN_REPROJECTION_ERROR_PX, _REPROJECTION_ERROR_CELLS * cell_px)
        pose = self._solve_pose(image_points, confidence, max_error)
        if pose is not None and self._pose_outside_fov(pose, height, width):
            pose = None
        return PoseEstimationResult(
            pose=pose,
            roi=roi,
            image_points=image_points,
            confidence=confidence,
        )

    def _discarded_pose(self, roi: ROI) -> PoseEstimationResult:
        count = len(self.keypoints)
        return PoseEstimationResult(
            pose=None,
            roi=roi,
            image_points=np.zeros((count, 2), dtype=np.float32),
            confidence=np.zeros(count, dtype=np.float32),
        )

    @staticmethod
    def _roi_outside_frame(roi: ROI, height: int, width: int) -> bool:
        if roi.w <= 1.0 or roi.h <= 1.0:
            return True
        bbox = roi_to_bbox(roi)
        x1 = max(bbox.x1, 0.0)
        y1 = max(bbox.y1, 0.0)
        x2 = min(bbox.x2, float(width))
        y2 = min(bbox.y2, float(height))
        return x2 - x1 <= 1.0 or y2 - y1 <= 1.0

    def _pose_outside_fov(self, pose: Pose, height: int, width: int) -> bool:
        projectable = projectable_mask(self.object_points, pose, self.distortion_coeffs)
        if not np.any(projectable):
            return True
        projected = project_points(
            self.object_points[projectable],
            pose,
            self.camera_intrinsics,
            self.distortion_coeffs,
        )
        inside = (
            (projected[:, 0] >= 0) & (projected[:, 0] < width)
            & (projected[:, 1] >= 0) & (projected[:, 1] < height)
        )
        return not bool(np.any(inside))

    def _solve_pose(
        self,
        image_points: np.ndarray,
        confidence: np.ndarray,
        max_error: float,
    ) -> Pose | None:
        """RANSAC PnP; keypoints reprojecting more than ``max_error`` px away are outliers."""
        valid = confidence >= self.peak_threshold
        if int(valid.sum()) < _MIN_PNP_INLIERS:
            return None

        object_points = self.object_points[valid]
        points_2d = image_points[valid].reshape(-1, 1, 2).astype(np.float32)
        dist_coeffs = as_dist_coeffs(self.distortion_coeffs)
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points,
            points_2d,
            self.camera_intrinsics,
            dist_coeffs,
            iterationsCount=100,
            reprojectionError=max_error,
            confidence=0.99,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not success:
            return None

        # With 4 or 5 points OpenCV solves directly and reports every point as an
        # inlier, so count inliers from the reprojection error here.
        reprojected, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_intrinsics, dist_coeffs
        )
        inliers = np.linalg.norm(reprojected - points_2d, axis=2).ravel() <= max_error
        if int(inliers.sum()) < _MIN_PNP_INLIERS:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inliers],
            points_2d[inliers],
            self.camera_intrinsics,
            dist_coeffs,
            rvec,
            tvec,
        )

        rotation, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = Rotation.from_matrix(rotation).as_quat()
        t = tvec.reshape(3)
        return Pose(float(t[0]), float(t[1]), float(t[2]), qx, qy, qz, qw)

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
