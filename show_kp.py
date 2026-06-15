import argparse
from pathlib import Path

from LightPoseEstim import (
    DataLoader,
    MeshKeypointSelector,
    load_canonical_keypoints,
    paths_from_keypoint_run_dir,
)

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"
DEFAULT_RUN_DIR = Path("keypoint_heatmap_20")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize saved canonical keypoints on the mesh.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Keypoint training run directory (e.g. keypoint_heatmap_20).",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mesh-scale", type=float, default=0.001)
    parser.add_argument(
        "--no-recenter-mesh",
        action="store_true",
        help="Do not translate mesh so its bbox center is at the origin.",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Draw small axis markers at each keypoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_paths = paths_from_keypoint_run_dir(args.run_dir)
    keypoints = load_canonical_keypoints(run_paths.keypoints)

    loader = DataLoader(
        args.dataset,
        mesh_scale=args.mesh_scale,
        recenter_mesh=not args.no_recenter_mesh,
    )
    MeshKeypointSelector().show_keypoints(
        loader.mesh,
        keypoints,
        show_ids=args.show_ids,
    )


if __name__ == "__main__":
    main()
