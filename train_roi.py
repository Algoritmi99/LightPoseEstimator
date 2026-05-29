import argparse
import logging
from pathlib import Path

from LightPoseEstim import DataLoader, ROIDetectorTrainer

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"
DEFAULT_SAVE_PATH = Path("roi_detector.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ROI detector on a dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--margin", type=float, default=1.3)
    parser.add_argument("--mesh-scale", type=float, default=0.001)
    parser.add_argument(
        "--no-recenter-mesh",
        action="store_true",
        help="Do not translate mesh so its bbox center is at the origin.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--no-freeze-backbone",
        action="store_true",
        help="Train the full model including the feature extractor (backbone frozen by default).",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help=(
            "Checkpoint path for the best validation loss; "
            "the epoch count is appended before the file extension."
        ),
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging (enabled by default).",
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

    save_path = args.save_path.with_name(
        f"{args.save_path.stem}_{args.epochs}{args.save_path.suffix}"
    )
    freeze_backbone = not args.no_freeze_backbone
    logging.info("Training for %d epochs (freeze_backbone=%s)", args.epochs, freeze_backbone)
    logging.info("Saving checkpoint to %s", save_path)

    trainer = ROIDetectorTrainer(roi_dataset, use_tensorboard=not args.no_tensorboard)
    trainer.train(
        args.epochs,
        save_path=str(save_path),
        freeze_backbone=freeze_backbone,
    )


if __name__ == "__main__":
    main()
