import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from LightPoseEstim import Pose, PoseEstimator, open_dataset
from LightPoseEstim.dataloader import RecordingSplit, _get_2d_roi
from LightPoseEstim.kp import DEFAULT_PEAK_THRESHOLD
from LightPoseEstim.pose_estimator import RUN_CONFIG_RELATIVE

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"
DEFAULT_RUN_DIR = Path("pose_estimator_20")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pose-estimator run from train.py on its held-out recordings "
            "and report translation/rotation error and how often a pose is returned."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--part",
        choices=("test", "val", "train"),
        default="test",
        help="Which recordings of the run's split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate an evenly spaced subset of this many frames.",
    )
    parser.add_argument(
        "--max-translation-error",
        type=float,
        default=0.05,
        help="Success threshold in meters.",
    )
    parser.add_argument(
        "--max-rotation-error",
        type=float,
        default=5.0,
        help="Success threshold in degrees.",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=DEFAULT_PEAK_THRESHOLD,
        help="Minimum heatmap peak (0-1) for a keypoint to be used. Tune on --part val.",
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cpu or cuda (default: cuda if available).")
    parser.add_argument("--csv", type=Path, default=None, help="Write per-frame results to this CSV file.")
    return parser.parse_args()


def pose_errors(pred: Pose, gt: Pose) -> tuple[float, float]:
    """Translation error in meters and rotation error in degrees."""
    translation = float(np.linalg.norm(pred.get_tvec() - gt.get_tvec()))
    relative = pred.quaternion.to_rotation_matrix().T @ gt.quaternion.to_rotation_matrix()
    cos_angle = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return translation, float(np.degrees(np.arccos(cos_angle)))


def estimator_for(estimators: dict, base: PoseEstimator, row) -> PoseEstimator:
    """Reuse the loaded nets with the camera calibration of this row's recording."""
    key = (row["camera_intrinsics"].tobytes(), row["distortion_coeffs"].tobytes())
    if key not in estimators:
        estimators[key] = PoseEstimator(
            roi_detector=base.roi_detector,
            keypoint_net=base.keypoint_net,
            keypoints=base.keypoints,
            camera_intrinsics=row["camera_intrinsics"],
            distortion_coeffs=row["distortion_coeffs"],
            crop_size=base.crop_size,
            peak_threshold=base.peak_threshold,
            device=base.device,
            resolve_symmetry=base.resolve_symmetry,
        )
    return estimators[key]


def percent(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%" if total else "n/a"


def summarize(values: np.ndarray, scale: float, unit: str) -> str:
    if values.size == 0:
        return "n/a"
    values = values * scale
    return (
        f"median {np.median(values):.2f} {unit}, mean {values.mean():.2f} {unit}, "
        f"p90 {np.percentile(values, 90):.2f} {unit}"
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    with (args.run_dir / RUN_CONFIG_RELATIVE).open() as handle:
        config = json.load(handle)
    loader = open_dataset(
        args.dataset,
        dataset_format=config.get("dataset_format"),
        mesh_scale=config.get("mesh_scale"),
        recenter_mesh=config.get("recenter_mesh"),
    )
    data = loader.data

    if "split" in config:
        split = RecordingSplit.from_dict(config["split"])
    else:
        logging.warning(
            "%s has no recorded split (trained before splitting by recording). "
            "Its training data covers every recording, so these results are optimistic.",
            args.run_dir,
        )
        split = loader.split_by_recording()

    missing = set(getattr(split, args.part)) - set(data["recording"])
    if missing:
        logging.warning(
            "%d %s recording(s) of the run are not in %s: %s",
            len(missing),
            args.part,
            args.dataset,
            ", ".join(sorted(missing)),
        )
    indices = split.indices(data, args.part)
    if not indices:
        raise SystemExit(f"No {args.part} frames of {args.run_dir} found in {args.dataset}.")
    if args.limit is not None and args.limit < len(indices):
        picks = np.unique(np.linspace(0, len(indices) - 1, args.limit).round().astype(int))
        indices = [indices[i] for i in picks]

    first = data.iloc[indices[0]]
    base = PoseEstimator.load_run(
        args.run_dir,
        camera_intrinsics=first["camera_intrinsics"],
        distortion_coeffs=first["distortion_coeffs"],
        peak_threshold=args.peak_threshold,
        device=torch.device(args.device) if args.device else None,
    )
    estimators = {}

    records = []
    for idx in tqdm(indices, unit="frame"):
        row = data.iloc[idx]
        result = estimator_for(estimators, base, row).estimate(row["image"])
        with Image.open(row["image"]) as image:
            width, height = image.size
        gt_roi = _get_2d_roi(
            (height, width),
            loader.mesh,
            row["pose"],
            row["camera_intrinsics"],
            row["distortion_coeffs"],
            margin=1.0,
        )
        translation_error, rotation_error = (
            pose_errors(result.pose, row["pose"])
            if result.pose is not None
            else (float("nan"), float("nan"))
        )
        records.append({
            "recording": row["recording"],
            "image": str(row["image"]),
            "class": row.get("class"),
            "in_view": gt_roi.w > 0 and gt_roi.h > 0,
            "has_pose": result.pose is not None,
            "distance_m": float(np.linalg.norm(row["pose"].get_tvec())),
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
        })

    in_view = np.array([r["in_view"] for r in records])
    has_pose = np.array([r["has_pose"] for r in records])
    translation = np.array([r["translation_error_m"] for r in records])
    rotation = np.array([r["rotation_error_deg"] for r in records])
    distance = np.array([r["distance_m"] for r in records])
    scored = in_view & has_pose
    success = (
        scored
        & (translation <= args.max_translation_error)
        & (rotation <= args.max_rotation_error)
    )
    n_in_view = int(in_view.sum())
    n_out_of_view = int((~in_view).sum())

    print(
        f"\nRun {args.run_dir}, {args.part} split: "
        f"{len({r['recording'] for r in records})} recordings, {len(records)} frames"
    )
    print(
        f"Object in view:     {n_in_view} frames, pose returned for "
        f"{int(has_pose[in_view].sum())} ({percent(int(has_pose[in_view].sum()), n_in_view)})"
    )
    print(
        f"Object out of view: {n_out_of_view} frames, pose returned for "
        f"{int(has_pose[~in_view].sum())} (should be 0)"
    )
    print(f"Translation error:  {summarize(translation[scored], 100.0, 'cm')}")
    print(f"  ... of distance:  {summarize(translation[scored] / distance[scored], 100.0, '%')}")
    print(f"Rotation error:     {summarize(rotation[scored], 1.0, 'deg')}")
    print(
        f"Success (<= {args.max_translation_error * 100:g} cm and <= {args.max_rotation_error:g} deg): "
        f"{int(success.sum())} of {n_in_view} in-view frames ({percent(int(success.sum()), n_in_view)})"
    )

    classes = [r["class"] for r in records]
    if any(c is not None for c in classes):
        for name in sorted({c for c in classes if c is not None}):
            mask = np.array([c == name for c in classes])
            frames, n_in = int(mask.sum()), int((mask & in_view).sum())
            print(
                f"  class {name:10s}: {frames:4d} frames, pose for {percent(int((mask & has_pose).sum()), frames)}, "
                f"success {percent(int((mask & success).sum()), n_in)} of {n_in} in view"
            )

    if args.csv is not None:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        logging.info("Wrote per-frame results to %s", args.csv)


if __name__ == "__main__":
    main()
