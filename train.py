import argparse
import json
import logging
from pathlib import Path

from LightPoseEstim import (
    DataLoader,
    KeypointHeatmapNet,
    KeypointHeatmapTrainer,
    MeshKeypointSelector,
    ROIDetector,
    ROIDetectorTrainer,
    save_canonical_keypoints,
)
from LightPoseEstim.dataloader import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
)
from LightPoseEstim.keypoint_selector import KEYPOINTS_FILENAME, KEYPOINTS_SUBDIR, NET_SUBDIR
from LightPoseEstim.pose_estimator import ROI_CHECKPOINT_RELATIVE, RUN_CONFIG_RELATIVE

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"
DEFAULT_OUTPUT = Path("pose_estimator")
FINE_TUNE_LR = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the ROI detector and keypoint heatmap net, then save a run "
            "directory that PoseEstimator.load_run can load."
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--margin", type=float, default=1.3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--keypoint-count", type=int, default=24)
    parser.add_argument("--mesh-scale", type=float, default=0.001)
    parser.add_argument(
        "--no-recenter-mesh",
        action="store_true",
        help="Do not translate mesh so its bbox center is at the origin.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Share of recordings held out for checkpoint selection.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Share of recordings held out for eval_pipe.py; never used in training.",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--roi-batch-size",
        type=int,
        default=64,
        help="ROI batch size. 64 uses about 24 GB on 1280x960 images.",
    )
    parser.add_argument(
        "--kp-batch-size",
        type=int,
        default=48,
        help="Keypoint batch size. 48 uses about 1.5 GB on 256x256 crops.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Data-loader worker processes. Overlaps image loading with GPU training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Run directory base path. Training writes {stem}_{epochs}/ with the "
            "fine-tuned ROI detector, keypoint net, canonical keypoints, and config."
        ),
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging (enabled by default).",
    )
    return parser.parse_args()


def run_directory(output: Path, epochs: int) -> Path:
    return output.parent / f"{output.stem}_{epochs}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    loader = DataLoader(
        args.dataset,
        mesh_scale=args.mesh_scale,
        recenter_mesh=not args.no_recenter_mesh,
    )
    run_dir = run_directory(args.output, args.epochs)
    stage0_dir = run_dir / "stage0"
    roi_stage0_path = stage0_dir / "roi_detector.pth"
    roi_path = run_dir / ROI_CHECKPOINT_RELATIVE
    keypoint_stage0_path = stage0_dir / "keypoint_heatmap.pth"
    keypoint_net_path = run_dir / NET_SUBDIR / "keypoint_heatmap.pth"
    keypoints_path = run_dir / KEYPOINTS_SUBDIR / KEYPOINTS_FILENAME
    for path in (
        roi_stage0_path,
        roi_path,
        keypoint_stage0_path,
        keypoint_net_path,
        keypoints_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    split = loader.split_by_recording(
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    logging.info(
        "Split by recording: %d train, %d val, %d test",
        len(split.train),
        len(split.val),
        len(split.test),
    )

    use_tensorboard = not args.no_tensorboard
    log_root = Path("runs") / run_dir.name

    roi_dataset = loader.get_roi_dataset(margin=args.margin)
    logging.info(
        "ROI stage 0: training for %d epochs (freeze_backbone=True, batch_size=%d, workers=%d)",
        args.epochs,
        args.roi_batch_size,
        args.workers,
    )
    logging.info("Saving ROI stage-0 checkpoint to %s", roi_stage0_path)
    ROIDetectorTrainer(
        roi_dataset,
        batch_size=args.roi_batch_size,
        num_workers=args.workers,
        split=split,
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=str(log_root / "roi"),
    ).train(
        args.epochs,
        save_path=str(roi_stage0_path),
        freeze_backbone=True,
    )

    logging.info(
        "ROI stage 1: fine-tuning for %d epochs from %s (freeze_backbone=False, lr=%s)",
        args.epochs,
        roi_stage0_path,
        FINE_TUNE_LR,
    )
    logging.info("Saving ROI checkpoint to %s", roi_path)
    ROIDetectorTrainer(
        roi_dataset,
        model=ROIDetector.load(roi_stage0_path),
        batch_size=args.roi_batch_size,
        num_workers=args.workers,
        split=split,
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=str(log_root / "roi_finetune"),
        optimizer_kwargs=dict(lr=FINE_TUNE_LR),
    ).train(
        args.epochs,
        save_path=str(roi_path),
        freeze_backbone=False,
    )

    logging.info("Selecting canonical keypoints from the mesh")
    keypoints = MeshKeypointSelector(
        final_keypoint_count=args.keypoint_count,
    ).select_keypoints(loader.mesh, keypoint_count=args.keypoint_count)
    save_canonical_keypoints(keypoints_path, keypoints)
    logging.info("Loading dataset with images, ROI, and keypoint heatmaps")
    kp_dataset = loader.get_kp_heatmap_roi_dataset(
        keypoints,
        crop_size=args.crop_size,
        margin=args.margin,
    )

    logging.info(
        "Keypoint stage 0: training for %d epochs (freeze_backbone=True, batch_size=%d, workers=%d)",
        args.epochs,
        args.kp_batch_size,
        args.workers,
    )
    logging.info("Saving keypoint stage-0 checkpoint to %s", keypoint_stage0_path)
    KeypointHeatmapTrainer(
        kp_dataset,
        batch_size=args.kp_batch_size,
        num_workers=args.workers,
        split=split,
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=str(log_root / "keypoints"),
    ).train(
        args.epochs,
        save_path=str(keypoint_stage0_path),
        freeze_backbone=True,
    )

    logging.info(
        "Keypoint stage 1: fine-tuning for %d epochs from %s (freeze_backbone=False, lr=%s)",
        args.epochs,
        keypoint_stage0_path,
        FINE_TUNE_LR,
    )
    logging.info("Saving keypoint checkpoint to %s", keypoint_net_path)
    KeypointHeatmapTrainer(
        kp_dataset,
        model=KeypointHeatmapNet.load(
            keypoint_stage0_path,
            num_keypoints=len(keypoints),
        ),
        batch_size=args.kp_batch_size,
        num_workers=args.workers,
        split=split,
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=str(log_root / "keypoints_finetune"),
        optimizer_kwargs=dict(lr=FINE_TUNE_LR),
    ).train(
        args.epochs,
        save_path=str(keypoint_net_path),
        freeze_backbone=False,
    )

    config_path = run_dir / RUN_CONFIG_RELATIVE
    config = {
        "crop_size": args.crop_size,
        "margin": args.margin,
        "mesh_scale": args.mesh_scale,
        "recenter_mesh": not args.no_recenter_mesh,
        "split_seed": args.split_seed,
        "split": split.to_dict(),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    logging.info("Saved pose estimator run to %s", run_dir)


if __name__ == "__main__":
    main()
