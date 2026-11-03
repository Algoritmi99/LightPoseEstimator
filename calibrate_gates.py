import argparse
import json
import logging
from pathlib import Path

from LightPoseEstim import open_dataset
from LightPoseEstim.calibration import DEFAULT_KEEP_FRACTION, calibrate_run
from LightPoseEstim.pose_estimator import RUN_CONFIG_RELATIVE


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a trained run's pose quality checks (rotation disagreement and\n"
            "keypoint confidence) on its validation frames and store them in its config.json.\n"
            "train.py does this at the end of training; use this for older runs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory written by train.py.")
    parser.add_argument("--dataset", type=Path, required=True, help="The dataset the run was trained on.")
    parser.add_argument(
        "--keep-fraction",
        type=float,
        default=DEFAULT_KEEP_FRACTION,
        help="Share of good validation poses each check must keep (default: %(default)s).",
    )
    parser.add_argument("--device", default=None, help="e.g. cpu or cuda (default: cuda if available).")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = json.loads((args.run_dir / RUN_CONFIG_RELATIVE).read_text())
    loader = open_dataset(
        args.dataset,
        dataset_format=config.get("dataset_format"),
        mesh_scale=config.get("mesh_scale"),
        recenter_mesh=config.get("recenter_mesh"),
    )
    loader.exclude_recordings(config.get("exclude_recordings", []))
    calibrate_run(args.run_dir, loader, keep_fraction=args.keep_fraction, device=args.device)


if __name__ == "__main__":
    main()
