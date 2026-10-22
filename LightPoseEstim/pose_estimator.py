import copy
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
from .projection import as_dist_coeffs, project_points, projectable_mask, ray_rotation
from .roi import ROI, roi_to_bbox
from .roi.roi_detector import ROI_INPUT_SIZE, ROIDetector

ROI_CHECKPOINT_RELATIVE = Path("roi") / "roi_detector.pth"
RUN_CONFIG_RELATIVE = Path("config.json")

# PnP outlier threshold: this many heatmap cells, mapped to full-image pixels.
_REPROJECTION_ERROR_CELLS = 2.0
_MIN_REPROJECTION_ERROR_PX = 4.0
# 4 points always fit some pose, so a 5th is needed to reject outliers at all.
_MIN_PNP_INLIERS = 5
# Identity and 180-degree turns about the object x, y and z axes.
_SYMMETRY_TURNS = [np.diag(signs).astype(np.float64) for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]


@dataclass
class PoseEstimationResult:
    pose: Pose | None
    roi: ROI
    image_points: np.ndarray
    confidence: np.ndarray
    inliers: int = 0
    # Mean heatmap peak of the PnP inliers, and degrees between the PnP rotation and the
    # keypoint net's coarse whole-crop rotation: both drop/grow when the evidence is poor.
    keypoint_confidence: float = float("nan")
    rotation_disagreement: float = float("nan")
    # Why ``pose`` is None: "no target in view", "too few keypoints", "outside field of
    # view", "rotation disagreement" or "low keypoint confidence".
    rejected: str | None = None


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
        resolve_symmetry: bool = True,
        max_rotation_disagreement: float | None = None,
        min_keypoint_confidence: float | None = None,
    ):
        """
        ``resolve_symmetry`` corrects 180-degree flips with the keypoint net's coarse
        rotation. It assumes the object frame's origin is the object's centre, as for the
        recentered Astrobee mesh; switch it off for objects that are not near-symmetric.

        ``max_rotation_disagreement`` (degrees) and ``min_keypoint_confidence`` reject poses
        built on poor evidence, e.g. a target too dark to see; None skips a check. Calibrate
        them per run with calibration.calibrate_run, as train.py does.
        """
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
        self.resolve_symmetry = resolve_symmetry
        self.max_rotation_disagreement = max_rotation_disagreement
        self.min_keypoint_confidence = min_keypoint_confidence
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
        roi_input_size: tuple[int, int] = ROI_INPUT_SIZE,
        resolve_symmetry: bool = True,
        max_rotation_disagreement: float | None = None,
        min_keypoint_confidence: float | None = None,
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
            roi_detector=ROIDetector.load(roi_detector_path, input_size=roi_input_size),
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
            resolve_symmetry=resolve_symmetry,
            max_rotation_disagreement=max_rotation_disagreement,
            min_keypoint_confidence=min_keypoint_confidence,
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
            roi_input_size=tuple(config.get("roi_input_size", ROI_INPUT_SIZE)),
            resolve_symmetry=bool(config.get("resolve_symmetry", True)),
            max_rotation_disagreement=config.get("max_rotation_disagreement"),
            min_keypoint_confidence=config.get("min_keypoint_confidence"),
        )

    def with_camera(
        self,
        camera_intrinsics: np.ndarray,
        distortion_coeffs: np.ndarray | None = None,
    ) -> "PoseEstimator":
        """The same networks and settings with another camera calibration."""
        estimator = copy.copy(self)
        estimator.camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float32)
        estimator.distortion_coeffs = distortion_coeffs
        return estimator

    def export(
        self,
        out_dir: str | Path,
        device: str | torch.device | None = None,
        verify: bool = True,
    ) -> Path:
        """
        Write everything needed to run this estimator outside Python (e.g. C++ with
        LibTorch) into ``out_dir``:

        - ``roi_detector/forward.pt2``, ``keypoint_net/forward.pt2``: the networks' forward
          passes compiled with torch.export + AOTInductor, for ``device`` (default: the
          estimator's). The ROI detector accepts any image size.
        - ``pose_estimator.json``: keypoints, camera intrinsics and distortion, crop size,
          heatmap stride, peak threshold, PnP settings, symmetry and quality checks.
        - ``README.md``: each step around the networks (ROI box, crop, peak decoding, PnP,
          checks) as formulas with these values, to reimplement as is.

        PnP and the other geometry are not compiled. A CUDA export needs a CUDA toolkit
        (CUDA_HOME). With ``verify``, each package is loaded back and compared with the
        PyTorch network; a mismatch raises.
        """
        from .export import export_pose_estimator

        return export_pose_estimator(self, out_dir, device=device, verify=verify)

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
        cells, confidence, rotation = self.keypoint_net.inference(
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
        result = PoseEstimationResult(
            pose=None,
            roi=roi,
            image_points=image_points,
            confidence=confidence,
        )
        pose, inliers = self._solve_pose(image_points, confidence, max_error)
        if pose is None:
            result.rejected = "too few keypoints"
            return result

        allocentric = rotation[0].cpu().numpy()
        if self.resolve_symmetry:
            pose = self._resolve_symmetry(pose, allocentric)
        result.inliers = int(inliers.sum())
        result.keypoint_confidence = float(confidence[inliers].mean())
        result.rotation_disagreement = self.rotation_disagreement(pose, allocentric)
        if self._pose_outside_fov(pose, height, width):
            result.rejected = "outside field of view"
        elif (
            self.max_rotation_disagreement is not None
            and result.rotation_disagreement > self.max_rotation_disagreement
        ):
            result.rejected = "rotation disagreement"
        elif (
            self.min_keypoint_confidence is not None
            and result.keypoint_confidence < self.min_keypoint_confidence
        ):
            result.rejected = "low keypoint confidence"
        else:
            result.pose = pose
        return result

    def _discarded_pose(self, roi: ROI) -> PoseEstimationResult:
        count = len(self.keypoints)
        return PoseEstimationResult(
            pose=None,
            roi=roi,
            image_points=np.zeros((count, 2), dtype=np.float32),
            confidence=np.zeros(count, dtype=np.float32),
            rejected="no target in view",
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

    @staticmethod
    def rotation_disagreement(pose: Pose, allocentric: np.ndarray) -> float:
        """Degrees between the pose's rotation and the net's coarse (allocentric) rotation."""
        coarse = ray_rotation(pose.get_tvec()) @ allocentric
        relative = pose.quaternion.to_rotation_matrix().T @ coarse
        return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))

    @staticmethod
    def _resolve_symmetry(pose: Pose, allocentric: np.ndarray) -> Pose:
        """
        Keypoints on a near-symmetric object can land on the wrong side, which turns the
        PnP pose 180 degrees about an object axis through the origin (the mesh bbox centre
        when recentered). Of the PnP pose and those turned versions, keep the one closest
        to the network's coarse whole-crop rotation.
        """
        tvec = pose.get_tvec()
        coarse = ray_rotation(tvec) @ allocentric
        rotation = pose.quaternion.to_rotation_matrix()
        candidates = [rotation @ turn for turn in _SYMMETRY_TURNS]
        best = int(np.argmax([np.trace(candidate.T @ coarse) for candidate in candidates]))
        if best == 0:
            return pose
        qx, qy, qz, qw = Rotation.from_matrix(candidates[best]).as_quat()
        t = tvec.reshape(3)
        return Pose(float(t[0]), float(t[1]), float(t[2]), qx, qy, qz, qw)

    def _solve_pose(
        self,
        image_points: np.ndarray,
        confidence: np.ndarray,
        max_error: float,
    ) -> tuple[Pose | None, np.ndarray | None]:
        """
        RANSAC PnP; keypoints reprojecting more than ``max_error`` px away are outliers.
        Returns the pose and the inlier mask over all keypoints.
        """
        valid = confidence >= self.peak_threshold
        if int(valid.sum()) < _MIN_PNP_INLIERS:
            return None, None

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
            return None, None

        # With 4 or 5 points OpenCV solves directly and reports every point as an
        # inlier, so count inliers from the reprojection error here.
        reprojected, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_intrinsics, dist_coeffs
        )
        inliers = np.linalg.norm(reprojected - points_2d, axis=2).ravel() <= max_error
        if int(inliers.sum()) < _MIN_PNP_INLIERS:
            return None, None
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
        inlier_mask = np.zeros(len(confidence), dtype=bool)
        inlier_mask[np.flatnonzero(valid)[inliers]] = True
        return Pose(float(t[0]), float(t[1]), float(t[2]), qx, qy, qz, qw), inlier_mask

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
