import argparse
import logging
from pathlib import Path

from LightPoseEstim import DataLoader, project_mesh_vertices, visualize_roi

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize projected mesh ROI on a dataset sample.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-idx", type=int, default=800)
    parser.add_argument("--margin", type=float, default=1.3)
    parser.add_argument("--mesh-scale", type=float, default=0.001)
    parser.add_argument(
        "--no-recenter-mesh",
        action="store_true",
        help="Do not translate mesh so its bbox center is at the origin.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    loader = DataLoader(
        args.dataset,
        mesh_scale=args.mesh_scale,
        recenter_mesh=not args.no_recenter_mesh,
    )
    roi_dataset = loader.get_roi_dataset(margin=args.margin)
    image, roi = roi_dataset[args.sample_idx]
    row = loader.data.iloc[args.sample_idx]

    projected_points = project_mesh_vertices(
        loader.mesh,
        row["pose"],
        row["camera_intrinsics"],
        row["distortion_coeffs"],
    )

    print(f"sample={args.sample_idx} roi={roi}")
    visualize_roi(image, roi, projected_points=projected_points)


if __name__ == "__main__":
    main()
