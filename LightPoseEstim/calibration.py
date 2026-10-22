"""
Calibrate the pose quality checks of a trained run on its validation frames.

A pose counts as good when its translation is within GOOD_TRANSLATION_FRACTION of the
distance and its rotation within GOOD_ROTATION_DEG. Each check's threshold is set to keep
``keep_fraction`` of the good validation poses, so it adapts to how reliable a run's
networks are: an orientation head that generalises poorly gets a loose threshold, or none.
"""
import copy
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .dataloader import DataLoader, RecordingSplit
from .pose import Pose, pose_errors
from .pose_estimator import RUN_CONFIG_RELATIVE, PoseEstimator

GOOD_TRANSLATION_FRACTION = 0.02
GOOD_ROTATION_DEG = 10.0
DEFAULT_KEEP_FRACTION = 0.99
MIN_GOOD_POSES = 30
# A disagreement check looser than this cannot tell right from wrong poses.
MAX_USEFUL_DISAGREEMENT_DEG = 90.0

logger = logging.getLogger(__name__)


def calibrate_quality_gates(
    estimator: PoseEstimator,
    frames: Iterable[tuple[object, Pose, np.ndarray, np.ndarray | None]],
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
) -> dict:
    """
    ``frames`` yields (image, true pose, camera intrinsics, distortion coefficients).
    Returns the thresholds (None for a check that cannot be calibrated) and the counts used.
    """
    ungated = copy.copy(estimator)
    ungated.max_rotation_disagreement = None
    ungated.min_keypoint_confidence = None
    disagreement, confidence, poses = [], [], 0
    for image, true_pose, camera_intrinsics, distortion_coeffs in frames:
        result = ungated.with_camera(camera_intrinsics, distortion_coeffs).estimate(image)
        if result.pose is None:
            continue
        poses += 1
        translation_error, rotation_error = pose_errors(result.pose, true_pose)
        distance = float(np.linalg.norm(true_pose.get_tvec()))
        if translation_error <= GOOD_TRANSLATION_FRACTION * distance and rotation_error <= GOOD_ROTATION_DEG:
            disagreement.append(result.rotation_disagreement)
            confidence.append(result.keypoint_confidence)

    gates = {"max_rotation_disagreement": None, "min_keypoint_confidence": None}
    summary = {"poses": poses, "good_poses": len(disagreement), "keep_fraction": keep_fraction}
    if len(disagreement) < MIN_GOOD_POSES:
        logger.warning(
            "Only %d good validation poses (need %d); pose quality checks stay off.",
            len(disagreement),
            MIN_GOOD_POSES,
        )
        return {**gates, **summary}
    max_disagreement = float(np.percentile(disagreement, 100.0 * keep_fraction))
    if max_disagreement <= MAX_USEFUL_DISAGREEMENT_DEG:
        gates["max_rotation_disagreement"] = round(max_disagreement, 2)
    else:
        logger.warning(
            "Orientation head disagrees by up to %.0f deg on good poses; rotation check off.",
            max_disagreement,
        )
    gates["min_keypoint_confidence"] = round(float(np.percentile(confidence, 100.0 * (1.0 - keep_fraction))), 4)
    return {**gates, **summary}


def calibrate_run(
    run_dir: Path,
    loader: DataLoader,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    part: str = "val",
) -> dict:
    """Calibrate a run's quality checks on its ``part`` frames and store them in its config."""
    config_path = Path(run_dir) / RUN_CONFIG_RELATIVE
    config = json.loads(config_path.read_text())
    split = RecordingSplit.from_dict(config["split"])
    rows = loader.data.iloc[split.indices(loader.data, part)]
    first = rows.iloc[0]
    estimator = PoseEstimator.load_run(run_dir, first["camera_intrinsics"], first["distortion_coeffs"])
    frames = (
        (row["image"], row["pose"], row["camera_intrinsics"], row["distortion_coeffs"])
        for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Calibrating checks", unit="frame")
    )
    result = calibrate_quality_gates(estimator, frames, keep_fraction)
    config["max_rotation_disagreement"] = result["max_rotation_disagreement"]
    config["min_keypoint_confidence"] = result["min_keypoint_confidence"]
    config["quality_gates"] = {key: result[key] for key in ("poses", "good_poses", "keep_fraction")} | {"part": part}
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    logger.info(
        "Pose quality checks from %d good %s poses: rotation disagreement <= %s deg, "
        "keypoint confidence >= %s",
        result["good_poses"],
        part,
        result["max_rotation_disagreement"],
        result["min_keypoint_confidence"],
    )
    return result
