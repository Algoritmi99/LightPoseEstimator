import argparse
import json
import logging
from pathlib import Path

import torch

from LightPoseEstim import (
    DataLoader,
    KeypointHeatmapTrainer,
    MeshKeypointSelector,
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum epochs for each network; the learning rate decays to 0 over them.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Stop a network's training after this many epochs without a better val loss.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for keypoint selection and network initialisation.",
    )
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
        help="ROI batch size. Images are downscaled to 320x240, so 64 needs little memory.",
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
            "ROI detector, keypoint net, canonical keypoints, and config."
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
    roi_path = run_dir / ROI_CHECKPOINT_RELATIVE
    keypoint_net_path = run_dir / NET_SUBDIR / "keypoint_heatmap.pth"
    keypoints_path = run_dir / KEYPOINTS_SUBDIR / KEYPOINTS_FILENAME
    for path in (roi_path, keypoint_net_path, keypoints_path):
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

    # One stage per network, whole model trained from the start: in tests this beat a
    # frozen-backbone stage followed by a low-learning-rate fine-tune.
    torch.manual_seed(args.seed)
    logging.info(
        "ROI detector: up to %d epochs (lr=%s, patience=%d, batch_size=%d, workers=%d)",
        args.epochs,
        args.lr,
        args.patience,
        args.roi_batch_size,
        args.workers,
    )
    ROIDetectorTrainer(
        loader.get_roi_dataset(margin=args.margin),
        batch_size=args.roi_batch_size,
        num_workers=args.workers,
        split=split,
        optimizer_kwargs=dict(lr=args.lr),
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=str(log_root / "roi"),
    ).train(args.epochs, save_path=str(roi_path), patience=args.patience)

    logging.info("Selecting canonical keypoints from the mesh")
    keypoints = MeshKeypointSelector(
        final_keypoint_count=args.keypoint_count,
        random_seed=args.seed,
    ).select_keypoints(loader.mesh, keypoint_count=args.keypoint_count)
    save_canonical_keypoints(keypoints_path, keypoints)
    logging.info("Loading dataset with images, ROI, and keypoint heatmaps")
    kp_dataset = loader.get_kp_heatmap_roi_dataset(
        keypoints,
        crop_size=args.crop_size,
        margin=args.margin,
    )

    logging.info(
        "Keypoint net: up to %d epochs (lr=%s, patience=%d, batch_size=%d, workers=%d)",
        args.epochs,
        args.lr,
        args.patience,
        args.kp_batch_size,
        args.workers,
    )
    KeypointHeatmapTrainer(
        kp_dataset,
        batch_size=args.kp_batch_size,
        num_workers=args.workers,
        split=split,
        optimizer_kwargs=dict(lr=args.lr),
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=str(log_root / "keypoints"),
    ).train(args.epochs, save_path=str(keypoint_net_path), patience=args.patience)

    config_path = run_dir / RUN_CONFIG_RELATIVE
    config = {
        "crop_size": args.crop_size,
        "margin": args.margin,
        "mesh_scale": args.mesh_scale,
        "recenter_mesh": not args.no_recenter_mesh,
        "seed": args.seed,
        "lr": args.lr,
        "split_seed": args.split_seed,
        "split": split.to_dict(),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    logging.info("Saved pose estimator run to %s", run_dir)


if __name__ == "__main__":
    main()
