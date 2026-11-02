import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from LightPoseEstim import PoseEstimator, open_dataset
from LightPoseEstim.pose import pose_errors
from LightPoseEstim.dataloader import RecordingSplit, _get_2d_roi
from LightPoseEstim.kp import DEFAULT_PEAK_THRESHOLD
from LightPoseEstim.pose_estimator import RUN_CONFIG_RELATIVE
from LightPoseEstim.tracking import TurnTracker, smooth_turns

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
    parser.add_argument(
        "--track",
        action="store_true",
        help=(
            "Also resolve near-symmetric twins over each recording's frames in time order: online "
            "(TurnTracker, past frames only, as on a live stream) and offline (smooth_turns). For runs "
            "with --symmetric-keypoints and recordings that are continuous sequences."
        ),
    )
    parser.add_argument(
        "--exclude-recordings",
        nargs="+",
        default=[],
        metavar="PATTERN",
        help="Also leave out recordings matching these shell-style patterns (the run's own exclusions always apply).",
    )
    parser.add_argument(
        "--no-quality-gates",
        action="store_true",
        help="Ignore the run's calibrated pose quality checks, to compare with and without them.",
    )
    return parser.parse_args()


def within(errors: tuple[float, float], args) -> bool:
    translation_error, rotation_error = errors
    return translation_error <= args.max_translation_error and rotation_error <= args.max_rotation_error


def _sequences(data, indices) -> dict[str, list[int]]:
    """Positions in ``indices`` per recording, in time order, for recordings whose labelled poses
    change smoothly from frame to frame (median rotation change under 20 degrees); recordings of
    independent random poses are left out."""
    by_recording: dict[str, list[int]] = {}
    for position, idx in enumerate(indices):
        by_recording.setdefault(data["recording"].iat[idx], []).append(position)
    sequences = {}
    for name, positions in by_recording.items():
        rows = [data.iloc[indices[p]] for p in positions]
        stamps = [float(r["timestamp"]) if "timestamp" in r and r["timestamp"] == r["timestamp"] else float(k) for k, r in enumerate(rows)]
        positions = [positions[k] for k in np.argsort(stamps, kind="stable")]
        rotations = [data["pose"].iat[indices[p]].quaternion.to_rotation_matrix() for p in positions]
        steps = [np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))) for a, b in zip(rotations[:-1], rotations[1:])]
        if len(positions) > 1 and np.median(steps) < 20.0:
            sequences[name] = positions
    return sequences


def report_tracking(data, indices, results, args) -> None:
    """Resolve twins per recording in time order, online and offline, and print how that scores."""
    sequences = _sequences(data, indices)
    if not sequences:
        print("Tracking: no recording is a continuous sequence.")
        return
    frames = sum(len(p) for p in sequences.values())
    single = sum(
        results[p].pose is not None and within(pose_errors(results[p].pose, data["pose"].iat[indices[p]]), args)
        for positions in sequences.values() for p in positions
    )
    posed_single = sum(results[p].pose is not None for positions in sequences.values() for p in positions)
    print(
        f"Tracking {len(sequences)} continuous recordings ({frames} frames); frame by frame, "
        f"success {percent(single, posed_single)} of {posed_single} poses"
    )
    for name, track in (("online ", "online"), ("offline", "offline")):
        posed = resolved = good = resolved_good = resolved_wrong_twin = unresolved_covered = 0
        for positions in sequences.values():
            rows = [data.iloc[indices[p]] for p in positions]
            stamps = [float(r["timestamp"]) if "timestamp" in r and r["timestamp"] == r["timestamp"] else float(k) for k, r in enumerate(rows)]
            frame_results = [results[p] for p in positions]
            if track == "online":
                tracker = TurnTracker()
                tracked = [tracker.update(r, t) for r, t in zip(frame_results, stamps)]
            else:
                tracked = smooth_turns(frame_results, stamps)
            for row, t in zip(rows, tracked):
                if t.pose is None:
                    continue
                posed += 1
                errors = pose_errors(t.pose, row["pose"])
                ok = within(errors, args)
                good += ok
                if t.resolved:
                    resolved += 1
                    resolved_good += ok
                    resolved_wrong_twin += errors[1] > 90.0
                else:
                    unresolved_covered += any(within(pose_errors(c.pose, row["pose"]), args) for c in t.candidates)
        print(
            f"Tracked {name}: success {percent(good, posed)} of {posed} poses; resolved {percent(resolved, posed)}, "
            f"of which success {percent(resolved_good, resolved)} and {resolved_wrong_twin} on the wrong twin; "
            f"unresolved ones have a right candidate in {percent(unresolved_covered, posed - resolved)}"
        )


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
    loader.exclude_recordings(config.get("exclude_recordings", []) + args.exclude_recordings)
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
    if args.no_quality_gates:
        base.max_rotation_disagreement = None
        base.min_keypoint_confidence = None
    logging.info(
        "Pose quality checks: rotation disagreement <= %s deg, keypoint confidence >= %s",
        base.max_rotation_disagreement,
        base.min_keypoint_confidence,
    )
    records = []
    results = []
    for idx in tqdm(indices, unit="frame"):
        row = data.iloc[idx]
        # The frame's own camera: its calibration and, for raw sensor recordings, Bayer pattern.
        pattern = row.get("bayer_pattern")
        estimator = base.with_camera(row["camera_intrinsics"], row["distortion_coeffs"], pattern if isinstance(pattern, str) else None)
        result = estimator.estimate(row["image"])
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
        results.append(result)
        records.append({
            "recording": row["recording"],
            "image": str(row["image"]),
            "class": row.get("class"),
            "in_view": gt_roi.w > 0 and gt_roi.h > 0,
            "has_pose": result.pose is not None,
            "rejected": result.rejected,
            "distance_m": float(np.linalg.norm(row["pose"].get_tvec())),
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
            "ambiguous": result.ambiguous,
            "candidates": len(result.candidates) if result.pose is not None else 0,
            "candidate_success": result.pose is not None and any(
                within(pose_errors(c.pose, row["pose"]), args) for c in result.candidates
            ),
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
    reasons = {}
    for record in records:
        if record["in_view"] and record["rejected"]:
            reasons[record["rejected"]] = reasons.get(record["rejected"], 0) + 1
    if reasons:
        print("  no pose because:  " + ", ".join(f"{reason} {count}" for reason, count in sorted(reasons.items())))
    print(f"Translation error:  {summarize(translation[scored], 100.0, 'cm')}")
    print(f"  ... of distance:  {summarize(translation[scored] / distance[scored], 100.0, '%')}")
    print(f"Rotation error:     {summarize(rotation[scored], 1.0, 'deg')}")
    print(
        f"Success (<= {args.max_translation_error * 100:g} cm and <= {args.max_rotation_error:g} deg): "
        f"{int(success.sum())} of {n_in_view} in-view frames ({percent(int(success.sum()), n_in_view)})"
    )

    ambiguous = np.array([r["ambiguous"] for r in records])
    if ambiguous.any():
        sure = scored & ~ambiguous
        candidate_success = np.array([r["candidate_success"] for r in records])
        print(
            f"Twins it can't tell apart: {int(ambiguous.sum())} of {int(has_pose.sum())} poses "
            f"({percent(int(ambiguous.sum()), int(has_pose.sum()))}), reported with all candidates; "
            f"one is right in {percent(int((candidate_success & ambiguous).sum()), int(ambiguous.sum()))}"
        )
        print(
            f"  unambiguous poses: success {percent(int((success & sure).sum()), int(sure.sum()))} "
            f"of {int(sure.sum())}"
        )
    if args.track:
        report_tracking(data, indices, results, args)

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
