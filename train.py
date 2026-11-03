import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
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
    RecordingSplit,
)
from LightPoseEstim.kp import KeypointHeatmapNet, KeypointLoss, SymmetricKeypointLoss
from LightPoseEstim.masks import BackgroundCompositor, build_background_pool, build_mask_cache
from LightPoseEstim.projection import allocentric_rotation
from LightPoseEstim.devices import resolve_device
from LightPoseEstim.image_cache import build_keypoint_region_cache, build_roi_input_cache
from LightPoseEstim.symmetry import KeypointSymmetry, TurnDistinctness, choose_reference, twin_balance_weights
from LightPoseEstim.calibration import DEFAULT_KEEP_FRACTION, calibrate_run
from LightPoseEstim.keypoint_selector import KEYPOINTS_FILENAME, KEYPOINTS_SUBDIR, NET_SUBDIR
from LightPoseEstim.pose_estimator import ROI_CHECKPOINT_RELATIVE, RUN_CONFIG_RELATIVE, TURN_DISTINCTNESS_FILENAME

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"
DEFAULT_OUTPUT = Path("pose_estimator")
DEFAULT_WORKERS = min(24, max(2, (os.cpu_count() or 4) * 3 // 4))
DEFAULT_CACHE_DIR = Path("cache")
BACKGROUND_PATCHES = 3000
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
    data.add_argument(
        "--split-from",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Reuse another run's train/val/test recordings instead of drawing a split, so the two "
            "runs are compared on the same held-out data. Recordings it doesn't list go to training."
        ),
    )
    data.add_argument(
        "--force-train",
        nargs="+",
        default=[],
        metavar="PATTERN",
        help=(
            "Put recordings whose name matches a shell-style pattern into training even if the split "
            "holds them out, e.g. '[0-9]*' for the random-pose renders of full_dataset."
        ),
    )
    data.add_argument(
        "--exclude-recordings",
        nargs="+",
        default=[],
        metavar="PATTERN",
        help=(
            "Leave out recordings whose name matches a shell-style pattern, e.g. "
            "'rec_run_2026-05-22-*'. Stored in the run, so eval_pipe.py leaves them out too."
        ),
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
    model.add_argument(
        "--symmetric-keypoints",
        action="store_true",
        help=(
            "Choose keypoints in sets that the object's 180-degree turns map onto each other "
            "(--keypoint-count must be a multiple of 4), let the heatmaps use any of the labellings "
            "the turns induce, and train a classifier that picks the turn. For near-symmetric "
            "objects whose keypoints otherwise swap with their twins on unseen data."
        ),
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
        default=20,
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
        "--background-augmentation",
        type=float,
        default=0.0,
        metavar="SHARE",
        help=(
            "Share of keypoint training crops whose surroundings are replaced by a patch of another "
            "training image or a plain dark background, so the net can't rely on the scenes it "
            "saw; 0 = off (default: %(default)s). Renders object masks once into --cache-dir."
        ),
    )
    training.add_argument(
        "--random-erasing",
        type=float,
        default=0.0,
        metavar="SHARE",
        help="Share of keypoint training crops with a random rectangle of noise, 0 = off (default: %(default)s).",
    )
    training.add_argument(
        "--oversample",
        nargs="+",
        default=[],
        metavar="PATTERN=FACTOR",
        help=(
            "Draw keypoint-net training crops of recordings matching PATTERN FACTOR times as often, "
            "e.g. '[0-9]*=2' for the random-pose renders, which show every view as each twin."
        ),
    )
    training.add_argument(
        "--balance-twins",
        action="store_true",
        help=(
            "With --symmetric-keypoints: weight each crop's turn-classifier loss by how rare its "
            "twin is among training crops with a similar view, so a view seen mostly as one twin "
            "can't teach the classifier which turn it is."
        ),
    )
    training.add_argument(
        "--device",
        default=None,
        help="Device to train on, e.g. cpu, cuda or cuda:1 (default: cuda if available, else cpu).",
    )
    training.add_argument(
        "--cpu-augmentation",
        action="store_true",
        help=(
            "Apply the photometric augmentation (colour jitter, blur, low light, random erasing) to "
            "each image in the loader workers instead of to each training batch on the GPU, which "
            "is the default when training on one. Same distributions either way; on colour images "
            "the CPU route can leave the GPU waiting."
        ),
    )
    training.add_argument(
        "--cache-images",
        action="store_true",
        help=(
            "Decode every image once, into memory-mapped files under --cache-dir that the OS keeps "
            "in RAM (the ROI detector's inputs, and the image parts keypoint crops come from), "
            "instead of every epoch: for large images whose decoding leaves the GPU waiting, such "
            "as EO-1's 2048x2048 PNGs (about 26 GB for its 5000 frames). Training sees the same data."
        ),
    )
    training.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Where object masks and background patches are cached between runs (default: %(default)s).",
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
        default=DEFAULT_WORKERS,
        help=(
            "Data-loader worker processes per loader. Image loading, not the GPU, limits training "
            "speed, so this uses 3/4 of the CPU threads, up to 24; each worker takes about "
            "250 MB (default: %(default)s)."
        ),
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


def split_from_run(run_dir: Path, data) -> RecordingSplit:
    """``run_dir``'s split restricted to the recordings in ``data``; any others go to training."""
    config = json.loads((run_dir / RUN_CONFIG_RELATIVE).read_text())
    present = set(data["recording"])
    parts = {part: [r for r in config["split"][part] if r in present] for part in ("train", "val", "test")}
    listed = set(sum(parts.values(), []))
    new = sorted(present - listed)
    if new:
        logging.warning("%d recordings are not in %s's split; adding them to training: %s", len(new), run_dir, ", ".join(new))
    return RecordingSplit(tuple(sorted(parts["train"] + new)), tuple(parts["val"]), tuple(parts["test"]))


def force_into_training(split: RecordingSplit, patterns: list[str]) -> RecordingSplit:
    """``split`` with every val/test recording matching a pattern moved to training."""
    from fnmatch import fnmatchcase

    moved = lambda name: any(fnmatchcase(name, p) for p in patterns)
    extra = [r for r in split.val + split.test if moved(r)]
    if extra:
        logging.info("Moving %d held-out recordings into training (%s): %s", len(extra), ", ".join(patterns), ", ".join(extra))
    return RecordingSplit(
        tuple(sorted(split.train + tuple(extra))),
        tuple(r for r in split.val if not moved(r)),
        tuple(r for r in split.test if not moved(r)),
    )


def oversampling_weights(recordings, specs: list[str]) -> np.ndarray:
    """Per-row sampling weights from 'PATTERN=FACTOR' specs (the last matching spec wins)."""
    from fnmatch import fnmatchcase

    rules = []
    for spec in specs:
        pattern, _, factor = spec.rpartition("=")
        if not pattern:
            raise SystemExit(f"--oversample expects PATTERN=FACTOR, got {spec!r}.")
        rules.append((pattern, float(factor)))
    weights = np.ones(len(recordings))
    for k, name in enumerate(recordings):
        for pattern, factor in rules:
            if fnmatchcase(name, pattern):
                weights[k] = factor
    return weights


def dataset_cache_key(args) -> str:
    return f"{args.dataset.resolve().name}-{hashlib.sha1(str(args.dataset.resolve()).encode()).hexdigest()[:8]}"


def background_compositor(loader, split: RecordingSplit, args) -> BackgroundCompositor:
    """Object masks for every frame, and background patches from training frames only."""
    dataset_key = dataset_cache_key(args)
    mask_dir = build_mask_cache(loader.data, loader.mesh, args.cache_dir / "masks" / dataset_key, args.workers)
    train_rows = loader.data.iloc[split.indices(loader.data, "train")].reset_index(drop=True)
    split_key = hashlib.sha1("\n".join(split.train).encode()).hexdigest()[:10]
    pool = build_background_pool(
        train_rows,
        mask_dir,
        args.cache_dir / "backgrounds" / f"{dataset_key}-{split_key}-{BACKGROUND_PATCHES}.npy",
        count=BACKGROUND_PATCHES,
        workers=args.workers,
        seed=args.seed,
    )
    return BackgroundCompositor(mask_dir, pool, p=args.background_augmentation)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    loader = open_dataset(
        args.dataset,
        dataset_format=args.dataset_format,
        mesh_scale=args.mesh_scale,
        recenter_mesh=args.recenter_mesh,
    )
    excluded = loader.exclude_recordings(args.exclude_recordings)
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
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device}: CUDA is not available.")
    # The photometric augmentation runs on whole batches on the GPU, unless told otherwise.
    batch_augmentation = device.type == "cuda" and not args.cpu_augmentation
    logging.info(
        "Training on %s; photometric augmentation %s",
        device,
        "on the GPU, per batch" if batch_augmentation else "in the loader workers, per image",
    )
    if args.balance_twins and not args.symmetric_keypoints:
        raise SystemExit("--balance-twins needs --symmetric-keypoints.")
    if args.symmetric_keypoints and args.keypoint_count % 4:
        raise SystemExit("--symmetric-keypoints needs a --keypoint-count that is a multiple of 4.")
    run_dir = run_directory(args.output, args.epochs)
    roi_path = run_dir / ROI_CHECKPOINT_RELATIVE
    keypoint_net_path = run_dir / NET_SUBDIR / "keypoint_heatmap.pth"
    keypoints_path = run_dir / KEYPOINTS_SUBDIR / KEYPOINTS_FILENAME
    for path in (roi_path, keypoint_net_path, keypoints_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    if args.split_from is not None:
        split = split_from_run(args.split_from, loader.data)
    else:
        split = loader.split_by_recording(
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )
    if args.force_train:
        split = force_into_training(split, args.force_train)
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
    roi_cache = None
    if args.cache_images:
        roi_cache = build_roi_input_cache(loader.data, roi_input_size, args.cache_dir / "images" / dataset_cache_key(args), args.workers)
    ROIDetectorTrainer(
        loader.get_roi_dataset(
            margin=args.margin,
            input_size=roi_input_size,
            low_light=args.low_light_augmentation,
            cache=roi_cache,
        ),
        model=ROIDetector(input_size=roi_input_size),
        batch_size=args.roi_batch_size,
        num_workers=args.workers,
        split=split,
        device=device,
        batch_augmentation=batch_augmentation,
        optimizer_kwargs=dict(lr=args.lr),
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=str(log_root / "roi"),
    ).train(args.epochs, save_path=str(roi_path), patience=args.patience)

    logging.info("Selecting canonical keypoints from the mesh")
    selector = MeshKeypointSelector(final_keypoint_count=args.keypoint_count, random_seed=args.seed, device=device)
    if args.symmetric_keypoints:
        keypoints = selector.select_symmetric_keypoints(loader.mesh, orbit_count=args.keypoint_count // 4)
    else:
        keypoints = selector.select_keypoints(loader.mesh, keypoint_count=args.keypoint_count)
    save_canonical_keypoints(keypoints_path, keypoints)
    symmetry = KeypointSymmetry.from_keypoints(keypoints)
    ambiguous_share = None
    if symmetry is not None:
        train_poses = loader.data["pose"].iloc[split.indices(loader.data, "train")]
        symmetry.reference, ambiguous_share = choose_reference(
            np.stack([allocentric_rotation(pose) for pose in train_poses]),
            symmetry.turns,
            symmetry.tolerance_deg,
            seed=args.seed,
        )
        logging.info(
            "Keypoints in %d symmetric sets; %.1f%% of training crops lie within %.0f deg of a "
            "boundary between labellings and may use either.",
            len(keypoints) // len(symmetry.turns),
            100 * ambiguous_share,
            symmetry.tolerance_deg,
        )
        # Where the object differs from its turned twins: the turn classifier learns "can't
        # tell" where none of it faces the camera, and the estimator reports both twins there.
        symmetry.distinctness = TurnDistinctness.from_mesh(loader.mesh, symmetry.turns, seed=args.seed)
        symmetry.distinctness.save(keypoints_path.parent / TURN_DISTINCTNESS_FILENAME)
        logging.info(
            "Surface that differs under each turn: %s",
            ", ".join(f"{100 * row.mean():.0f}%" for row in symmetry.distinctness.differs[1:]),
        )
    logging.info("Loading dataset with images, ROI, and keypoint heatmaps")
    background = None
    if args.background_augmentation > 0:
        background = background_compositor(loader, split, args)
    sample_weights = oversampling_weights(loader.data["recording"], args.oversample) if args.oversample else None
    turn_weights = None
    if args.balance_twins:
        train_rows = split.indices(loader.data, "train")
        weights, share = twin_balance_weights(
            np.stack([allocentric_rotation(pose) for pose in loader.data["pose"].iloc[train_rows]]),
            symmetry,
            sample_weights=sample_weights[train_rows] if sample_weights is not None else None,
        )
        turn_weights = np.ones(len(loader.data))
        turn_weights[train_rows] = weights
        logging.info(
            "Twin balance: turn-loss weights %.2f-%.2f (10th-90th percentile %.2f-%.2f); %.0f%% of "
            "training crops have views seen only as their own twin",
            weights.min(), weights.max(), *np.percentile(weights, [10, 90]), 100 * np.mean(share >= 0.95),
        )
    regions = None
    if args.cache_images:
        regions = build_keypoint_region_cache(loader.data, loader.mesh, args.margin, args.cache_dir / "images" / dataset_cache_key(args), args.workers)
    kp_dataset = loader.get_kp_heatmap_roi_dataset(
        keypoints,
        crop_size=args.crop_size,
        margin=args.margin,
        low_light=args.low_light_augmentation,
        background=background,
        random_erasing=args.random_erasing,
        symmetry=symmetry,
        turn_weights=turn_weights,
        regions=regions,
        device=device,
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
        model=KeypointHeatmapNet(
            num_keypoints=len(keypoints),
            num_turns=len(symmetry.turns) if symmetry is not None else 0,
        ),
        criterion=SymmetricKeypointLoss(symmetry) if symmetry is not None else KeypointLoss(),
        sample_weights=sample_weights,
        batch_size=args.kp_batch_size,
        num_workers=args.workers,
        split=split,
        device=device,
        batch_augmentation=batch_augmentation,
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
        "background_augmentation": args.background_augmentation,
        "random_erasing": args.random_erasing,
        "symmetric_keypoints": symmetry is not None,
        "labelling_reference": symmetry.reference.tolist() if symmetry is not None else None,
        "labelling_tolerance_deg": symmetry.tolerance_deg if symmetry is not None else None,
        "ambiguous_training_share": ambiguous_share,
        "exclude_recordings": args.exclude_recordings,
        "force_train": args.force_train,
        "oversample": args.oversample,
        "balance_twins": args.balance_twins,
        "batch_augmentation": batch_augmentation,
        "excluded_recordings": excluded,
        "split_seed": args.split_seed,
        "split_from": str(args.split_from) if args.split_from is not None else None,
        "split": split.to_dict(),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if args.gate_keep_fraction > 0:
        calibrate_run(run_dir, loader, keep_fraction=args.gate_keep_fraction, device=device)
    logging.info("Saved pose estimator run to %s", run_dir)


if __name__ == "__main__":
    main()
