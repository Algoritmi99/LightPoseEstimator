"""
Check the C++ library against the Python PoseEstimator: the same frames, in time order, through
Python's track() and the C++ `main --track`, compared frame by frame (pose or the reason for
none, the twin chosen, the twins' probabilities, the pose) and on the tracked output (resolved,
measured, the twin reported).

By default the Python side runs the exported network packages, as the C++ side does, so what
differs is the geometry around them (and the OpenCV version); --networks pytorch compares with
the PyTorch networks instead. Both sides run on the device the export is compiled for.

    python LightPoseEstimlib/check_against_python.py --run-dir pose_estimator_astrobee_sym3_30 \
        --export-dir export_sym3_cpu --binary LightPoseEstimlib/build/main \
        --dataset /path/to/full_dataset --frames 200
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from LightPoseEstim import PoseEstimator, open_dataset  # noqa: E402
from LightPoseEstim.dataloader import RecordingSplit  # noqa: E402
from LightPoseEstim.pose import Pose  # noqa: E402

# A pose "agrees" within these; RANSAC can settle on another inlier set on a few frames.
ROTATION_TOLERANCE_DEG = 0.1
TRANSLATION_TOLERANCE_M = 0.001
PROBABILITY_TOLERANCE = 1e-3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True, help="The run exported with PoseEstimator.export.")
    parser.add_argument("--binary", type=Path, required=True, help="The C++ main executable.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--recordings", nargs="+", default=None, help="Default: the run's test recordings.")
    parser.add_argument("--frames", type=int, default=None, help="At most this many frames per recording, from the start.")
    parser.add_argument("--networks", choices=("packages", "pytorch"), default="packages")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def use_packages(estimator: PoseEstimator, export_dir: Path) -> None:
    """Run the exported packages in place of the PyTorch networks."""
    manifest = json.loads((export_dir / "pose_estimator.json").read_text())
    for name, net in (("roi_detector", estimator.roi_detector), ("keypoint_net", estimator.keypoint_net)):
        runner = torch._inductor.aoti_load_package(str(export_dir / manifest["networks"][name]["package"]))

        def forward(x, runner=runner):
            out = runner(x)
            if isinstance(out, torch.Tensor):
                return out
            return out[0] if len(out) == 1 else tuple(out)

        net.forward = forward


def pose_of(item: dict) -> Pose | None:
    if "translation" not in item:
        return None
    return Pose(*item["translation"], *item["quaternion"])


def difference(a: Pose, b: Pose) -> tuple[float, float]:
    """Degrees and metres between two poses."""
    relative = a.quaternion.to_rotation_matrix() @ b.quaternion.to_rotation_matrix().T
    degrees = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
    return degrees, float(np.linalg.norm(np.asarray(a.position.to_tvec(), float) - np.asarray(b.position.to_tvec(), float)))


def run_cpp(args, frames, camera_file: Path, bayer: str | None) -> list[dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for row in frames:
            listing.write(f"{row['image']} {float(row['timestamp']):.6f}\n")
    command = [str(args.binary), "--track", "--threads", str(args.threads), "--list", listing.name, "--camera", str(camera_file)]
    if bayer:
        command += ["--bayer", bayer]
    command.append(str(args.export_dir))
    out = subprocess.run(command, capture_output=True, text=True, env={"OMP_NUM_THREADS": str(args.threads), "PATH": "/usr/bin:/bin"})
    if out.returncode != 0:
        raise RuntimeError(f"C++ main failed:\n{out.stderr}")
    return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]


def compare(python, cpp, stats: dict) -> list[str]:
    """Fold one frame into stats; return notes on what disagrees."""
    notes = []
    frame = python.frame
    p_pose, c_pose = frame.pose, pose_of(cpp)
    stats["frames"] += 1
    same_outcome = (p_pose is None) == (c_pose is None) and (frame.rejected or "") == cpp["rejected"]
    stats["same_outcome"] += same_outcome
    if not same_outcome:
        notes.append(f"outcome: python {frame.rejected or 'pose'} / c++ {cpp['rejected'] or 'pose'}")
    if frame.hypotheses and cpp["hypotheses"]:
        stats["with_twins"] += 1
        same_order = [h.turn for h in frame.hypotheses] == [h["turn"] for h in cpp["hypotheses"]]
        stats["same_twin_order"] += same_order
        if same_order:
            gap = max(abs(h.probability - c["probability"]) for h, c in zip(frame.hypotheses, cpp["hypotheses"]))
            stats["probability_gap"] = max(stats["probability_gap"], gap)
            stats["same_probabilities"] += gap <= PROBABILITY_TOLERANCE
        else:
            notes.append(f"twin order: python {[h.turn for h in frame.hypotheses]} / c++ {[h['turn'] for h in cpp['hypotheses']]}")
    if p_pose is not None and c_pose is not None:
        stats["both_posed"] += 1
        stats["same_turn"] += frame.turn == cpp["turn"]
        degrees, metres = difference(p_pose, c_pose)
        close = degrees <= ROTATION_TOLERANCE_DEG and metres <= TRANSLATION_TOLERANCE_M
        stats["same_pose"] += close
        stats["pose_gaps"].append((degrees, metres))
        if not close:
            notes.append(f"pose differs by {degrees:.3f} deg / {1000 * metres:.2f} mm")

    tracked = cpp["tracked"]
    t_pose = pose_of(tracked)
    stats["tracked_same_outcome"] += (python.pose is None) == (t_pose is None)
    if python.pose is not None and t_pose is not None:
        stats["tracked_both"] += 1
        same = python.resolved == tracked["resolved"] and python.measured == tracked["measured"]
        degrees, metres = difference(python.pose, t_pose)
        same_twin = degrees < 30.0
        stats["tracked_same"] += same and same_twin
        stats["tracked_same_twin"] += same_twin
        if not (same and same_twin):
            notes.append(f"tracked: python resolved={python.resolved} measured={python.measured} / c++ resolved={tracked['resolved']} "
                         f"measured={tracked['measured']}, poses {degrees:.1f} deg apart")
    return notes


def main():
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    loader = open_dataset(args.dataset)
    loader.exclude_recordings(config.get("exclude_recordings", []))
    data = loader.data
    recordings = args.recordings or list(RecordingSplit.from_dict(config["split"]).test)
    first = data[data.recording == recordings[0]].iloc[0]
    device = json.loads((args.export_dir / "pose_estimator.json").read_text())["device"]
    base = PoseEstimator.load_run(args.run_dir, first["camera_intrinsics"], first["distortion_coeffs"], device=device)
    if args.networks == "packages":
        use_packages(base, args.export_dir)

    stats = {key: 0 for key in ("frames", "same_outcome", "with_twins", "same_twin_order", "same_probabilities", "both_posed",
                                "same_turn", "same_pose", "tracked_same_outcome", "tracked_both", "tracked_same", "tracked_same_twin")}
    stats["probability_gap"], stats["pose_gaps"] = 0.0, []
    examples = []
    for recording in recordings:
        frames = data[data.recording == recording].sort_values("timestamp")
        if args.frames:
            frames = frames.head(args.frames)
        row0 = frames.iloc[0]
        pattern = row0.get("bayer_pattern")
        bayer = pattern if isinstance(pattern, str) else None
        estimator = base.with_camera(row0["camera_intrinsics"], row0["distortion_coeffs"], bayer)
        python = [estimator.track(row["image"], float(row["timestamp"])) for _, row in frames.iterrows()]
        rows = [row for _, row in frames.iterrows()]
        cpp = run_cpp(args, rows, Path(row0["image"]).parent.parent / "camera_intrinsics.json", bayer)
        if len(cpp) != len(python):
            raise RuntimeError(f"{recording}: C++ returned {len(cpp)} frames for {len(python)}.")
        before = stats["frames"]
        for row, p, c in zip(rows, python, cpp):
            for note in compare(p, c, stats):
                if len(examples) < 12:
                    examples.append(f"{recording}/{Path(row['image']).name}: {note}")
        print(f"{recording}: {stats['frames'] - before} frames compared")

    s = stats
    pct = lambda a, b: f"{100.0 * a / b:.1f}%" if b else "n/a"
    gaps = np.array(s["pose_gaps"]) if s["pose_gaps"] else np.zeros((1, 2))
    print(f"\nnetworks: {args.networks}; {s['frames']} frames")
    print(f"same outcome (pose, or the same reason for none): {pct(s['same_outcome'], s['frames'])}")
    print(f"same twins in the same order: {pct(s['same_twin_order'], s['with_twins'])}; probabilities within {PROBABILITY_TOLERANCE}: "
          f"{pct(s['same_probabilities'], s['same_twin_order'])} (largest gap {s['probability_gap']:.2e})")
    print(f"both with a pose: {s['both_posed']}; same twin {pct(s['same_turn'], s['both_posed'])}; pose within "
          f"{ROTATION_TOLERANCE_DEG} deg and {1000 * TRANSLATION_TOLERANCE_M:g} mm: {pct(s['same_pose'], s['both_posed'])} "
          f"(median gap {np.median(gaps[:, 0]):.2e} deg / {1000 * np.median(gaps[:, 1]):.2e} mm)")
    print(f"tracked: same outcome {pct(s['tracked_same_outcome'], s['frames'])}; same twin {pct(s['tracked_same_twin'], s['tracked_both'])}; "
          f"same twin, resolved and measured {pct(s['tracked_same'], s['tracked_both'])}")
    if examples:
        print("\nfirst disagreements:")
        print("\n".join("  " + e for e in examples))


if __name__ == "__main__":
    main()
