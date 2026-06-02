import argparse
import logging
from pathlib import Path

from LightPoseEstim import DataLoader, ROIDetector, visualize_roi

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"
DEFAULT_MODEL = Path("roi_detector_nf_20.pth")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ROI detector on dataset samples."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("-k", type=int, default=200, help="Number of samples to evaluate.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--start-idx", type=int, default=2300)
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

    model = ROIDetector.load(args.model)
    device = next(model.parameters()).device

    loader = DataLoader(
        args.dataset,
        mesh_scale=args.mesh_scale,
        recenter_mesh=not args.no_recenter_mesh,
    )
    roi_dataset = loader.get_roi_dataset(margin=args.margin)

    for i in range(args.k):
        idx = args.start_idx + i
        image, gt_roi = roi_dataset[idx]

        pred_roi = model.inference(image.unsqueeze(0).to(device))[0]
        logging.info("sample=%d gt=%s pred=%s", idx, gt_roi.tolist(), pred_roi)
        visualize_roi(
            image,
            [pred_roi.cx, pred_roi.cy, pred_roi.w, pred_roi.h],
            reference_roi=gt_roi,
            normalized=False,
            reference_normalized=True,
            title=f"sample {idx} (green=pred, blue=gt)",
        )


if __name__ == "__main__":
    main()
