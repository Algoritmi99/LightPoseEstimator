import argparse
import logging
from pathlib import Path

from LightPoseEstim import DataLoader, ROIDetector, ROIDetectorTrainer

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"
DEFAULT_STAGE0_SAVE_PATH = Path("roi_detector.pth")
DEFAULT_STAGE1_SAVE_PATH = Path("roi_detector_nf.pth")


def checkpoint_path(save_path: Path, epochs: int) -> Path:
    return save_path.with_name(f"{save_path.stem}_{epochs}{save_path.suffix}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the ROI detector in two stages: from scratch with a frozen "
            "backbone, then fine-tune the full model from that checkpoint."
        ),
    )
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
        "--save-path",
        type=Path,
        default=DEFAULT_STAGE0_SAVE_PATH,
        help=(
            "Stage-0 checkpoint path for the best validation loss; "
            "the epoch count is appended before the file extension."
        ),
    )
    parser.add_argument(
        "--finetune-save-path",
        type=Path,
        default=DEFAULT_STAGE1_SAVE_PATH,
        help=(
            "Stage-1 checkpoint path for the best validation loss; "
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

    use_tensorboard = not args.no_tensorboard
    stage0_path = checkpoint_path(args.save_path, args.epochs)
    stage1_path = checkpoint_path(args.finetune_save_path, args.epochs)

    logging.info("Stage 0: training for %d epochs (freeze_backbone=True)", args.epochs)
    logging.info("Saving stage-0 checkpoint to %s", stage0_path)
    trainer = ROIDetectorTrainer(
        roi_dataset,
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir="runs/roi_detector",
    )
    trainer.train(
        args.epochs,
        save_path=str(stage0_path),
        freeze_backbone=True,
    )

    logging.info(
        "Stage 1: fine-tuning for %d epochs from %s (freeze_backbone=False, lr=1e-5)",
        args.epochs,
        stage0_path,
    )
    logging.info("Saving stage-1 checkpoint to %s", stage1_path)
    trainer = ROIDetectorTrainer(
        roi_dataset,
        model=ROIDetector.load(stage0_path),
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir="runs/roi_detector_nf",
        optimizer_kwargs=dict(lr=1e-5),
    )
    trainer.train(
        args.epochs,
        save_path=str(stage1_path),
        freeze_backbone=False,
    )


if __name__ == "__main__":
    main()
