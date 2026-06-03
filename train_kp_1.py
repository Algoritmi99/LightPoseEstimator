import argparse
import logging
from pathlib import Path

from LightPoseEstim import (
    DataLoader,
    KeypointHeatmapNet,
    KeypointHeatmapTrainer,
    load_canonical_keypoints,
    paths_from_keypoint_model,
    resolve_keypoint_run_paths,
    save_canonical_keypoints,
)

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"
DEFAULT_SAVE_PATH = Path("keypoint_heatmap_nf.pth")
DEFAULT_MODEL = Path("keypoint_heatmap_20")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the keypoint heatmap net on a dataset.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=(
            "Prior training run directory or checkpoint under .../net/; "
            "loads the paired net weights and canonical keypoints."
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--margin", type=float, default=1.3)
    parser.add_argument("--crop-size", type=int, default=256)
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
            "Output base path; creates {stem}_{epochs}/net/ and {stem}_{epochs}/keypoints/ "
            "with the checkpoint and canonical keypoints."
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
    source_paths = paths_from_keypoint_model(args.model)
    keypoints = load_canonical_keypoints(source_paths.keypoints)
    logging.info(
        "Loaded %d keypoints from %s",
        len(keypoints),
        source_paths.keypoints,
    )
    model = KeypointHeatmapNet.load(
        source_paths.net_checkpoint,
        num_keypoints=len(keypoints),
    )
    logging.info("Loaded keypoint net from %s", source_paths.net_checkpoint)
    kp_dataset = loader.get_kp_heatmap_roi_dataset(
        keypoints,
        crop_size=args.crop_size,
        margin=args.margin,
    )

    run_paths = resolve_keypoint_run_paths(args.save_path, args.epochs)
    freeze_backbone = not args.no_freeze_backbone
    save_canonical_keypoints(run_paths.keypoints, keypoints)

    logging.info("Training for %d epochs (freeze_backbone=%s)", args.epochs, freeze_backbone)
    logging.info("Saving run artifacts under %s", run_paths.run_dir)

    trainer = KeypointHeatmapTrainer(
        kp_dataset,
        model=model,
        use_tensorboard=not args.no_tensorboard,
        optimizer_kwargs=dict(lr=1e-5),
    )
    trainer.train(
        args.epochs,
        save_path=str(run_paths.net_checkpoint),
        freeze_backbone=freeze_backbone,
    )


if __name__ == "__main__":
    main()
