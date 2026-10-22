import argparse
import json
import logging
from pathlib import Path

import torch

from LightPoseEstim import (
    KeypointHeatmapTrainer,
    MeshKeypointSelector,
    ROIDetector,
    ROIDetectorTrainer,
    open_dataset,
    save_canonical_keypoints,
)
from LightPoseEstim.dataloader import (
    DATASET_LOADERS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
)
from LightPoseEstim.calibration import DEFAULT_KEEP_FRACTION, calibrate_run
from LightPoseEstim.keypoint_selector import KEYPOINTS_FILENAME, KEYPOINTS_SUBDIR, NET_SUBDIR
from LightPoseEstim.pose_estimator import ROI_CHECKPOINT_RELATIVE, RUN_CONFIG_RELATIVE

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"
DEFAULT_OUTPUT = Path("pose_estimator")
ROI_INPUT_WIDTH = 320


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the ROI detector and keypoint heatmap net, then save a run\n"
            "directory that PoseEstimator.load_run can load."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python train.py --dataset /path/to/Dataset_2 --epochs 30\n"
            "  python train.py --dataset \"/path/to/EO1 Dataset\" --epochs 60 --patience 20 \\\n"
            "      --output pose_estimator_eo1\n"
            "\n"
            "Evaluate a finished run with eval_pipe.py --run-dir <run directory> --dataset <same dataset>."
        ),
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Dataset root directory (default: %(default)s).",
    )
    data.add_argument(
        "--dataset-format",
        choices=sorted(DATASET_LOADERS),
        default=None,
        help="Dataset layout (default: detected; 'eo1' when pose_labels.json and dataset/ exist).",
    )
    data.add_argument(
        "--mesh-scale",
        type=float,
        default=None,
        help="Mesh units to metres (default: the layout's; 0.001 for recordings, 1 for eo1).",
    )
    data.add_argument(
        "--recenter-mesh",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Translate the mesh so its bbox centre is the origin (default: the layout's; "
            "on for recordings, off for eo1, whose poses are relative to the model origin)."
        ),
    )
    data.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Share of recordings held out for checkpoint selection (default: %(default)s).",
    )
    data.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Share of recordings held out for eval_pipe.py; never used in training (default: %(default)s).",
    )
    data.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed for assigning recordings to train/val/test (default: %(default)s).",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--margin",
        type=float,
        default=1.3,
        help="ROI size relative to the object's projected box (default: %(default)s).",
    )
    model.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Side of the square ROI crop the keypoint net sees, in pixels (default: %(default)s).",
    )
    model.add_argument(
        "--keypoint-count",
        type=int,
        default=24,
        help="Number of mesh keypoints to select and detect (default: %(default)s).",
    )

    training = parser.add_argument_group("training")
    training.add_argument(
        "--epochs",
        type=int,
        default=30,
        help=(
            "Maximum epochs for each network; the learning rate decays to 0 over them, so a "
            "larger value also keeps the rate high for longer (default: %(default)s)."
        ),
    )
    training.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Starting learning rate for both networks (default: %(default)s).",
    )
    training.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Stop a network's training after this many epochs without a better val loss (default: %(default)s).",
    )
    training.add_argument(
        "--low-light-augmentation",
        type=float,
        default=0.0,
        metavar="SHARE",
        help=(
            "Share of training images darkened to simulate poor lighting, 0 = off (default: "
            "%(default)s). At 0.3 on EO-1 it did not help dim frames and cost some accuracy."
        ),
    )
    training.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for keypoint selection and network initialisation (default: %(default)s).",
    )
    training.add_argument(
        "--roi-batch-size",
        type=int,
        default=64,
        help="ROI batch size. Images are downscaled to 320 px wide, so 64 needs little memory (default: %(default)s).",
    )
    training.add_argument(
        "--kp-batch-size",
        type=int,
        default=48,
        help="Keypoint batch size. 48 uses about 1.5 GB on 256x256 crops (default: %(default)s).",
    )
    training.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Data-loader worker processes. Overlaps image loading with GPU training (default: %(default)s).",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Run directory base path. Training writes {stem}_{epochs}/ with the ROI detector, "
            "keypoint net, canonical keypoints, and config, overwriting an existing directory "
            "of that name (default: %(default)s)."
        ),
    )
    output.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging to runs/{run directory name}/ (enabled by default).",
    )
    output.add_argument(
        "--gate-keep-fraction",
        type=float,
        default=DEFAULT_KEEP_FRACTION,
        help=(
            "After training, calibrate the pose quality checks on the validation frames so each "
            "keeps this share of good poses; 0 turns the checks off (default: %(default)s)."
        ),
    )
    return parser.parse_args()


def run_directory(output: Path, epochs: int) -> Path:
    return output.parent / f"{output.stem}_{epochs}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    loader = open_dataset(
        args.dataset,
        dataset_format=args.dataset_format,
        mesh_scale=args.mesh_scale,
        recenter_mesh=args.recenter_mesh,
    )
    image_width, image_height = loader.image_size()
    # Keep the image's aspect ratio at a height the backbone's stride 32 divides.
    roi_input_size = (
        ROI_INPUT_WIDTH,
        max(32, round(ROI_INPUT_WIDTH * image_height / image_width / 32) * 32),
    )
    # Symmetric flips are resolved about the origin, so only when it is the bbox centre.
    resolve_symmetry = loader.recenter_mesh
    logging.info(
        "Dataset %s: %d frames of %dx%d, mesh scale %s, recentered %s, ROI input %dx%d",
        loader.FORMAT,
        len(loader.data),
        image_width,
        image_height,
        loader.mesh_scale,
        loader.recenter_mesh,
        *roi_input_size,
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
        loader.get_roi_dataset(
            margin=args.margin,
            input_size=roi_input_size,
            low_light=args.low_light_augmentation,
        ),
        model=ROIDetector(input_size=roi_input_size),
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
        low_light=args.low_light_augmentation,
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
        "dataset_format": loader.FORMAT,
        "crop_size": args.crop_size,
        "margin": args.margin,
        "mesh_scale": loader.mesh_scale,
        "recenter_mesh": loader.recenter_mesh,
        "roi_input_size": list(roi_input_size),
        "resolve_symmetry": resolve_symmetry,
        "seed": args.seed,
        "lr": args.lr,
        "low_light_augmentation": args.low_light_augmentation,
        "split_seed": args.split_seed,
        "split": split.to_dict(),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if args.gate_keep_fraction > 0:
        calibrate_run(run_dir, loader, keep_fraction=args.gate_keep_fraction)
    logging.info("Saved pose estimator run to %s", run_dir)


if __name__ == "__main__":
    main()
