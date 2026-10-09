import argparse
import logging
from pathlib import Path

from LightPoseEstim import (
    DataLoader,
    KeypointHeatmapNet,
    KeypointHeatmapTrainer,
    MeshKeypointSelector,
    resolve_keypoint_run_paths,
    save_canonical_keypoints,
)

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "Astrobee"
DEFAULT_STAGE0_SAVE_PATH = Path("keypoint_heatmap.pth")
DEFAULT_STAGE1_SAVE_PATH = Path("keypoint_heatmap_nf.pth")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the keypoint heatmap net in two stages: from scratch with a "
            "frozen backbone, then fine-tune the full model from that checkpoint."
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
        "--save-path",
        type=Path,
        default=DEFAULT_STAGE0_SAVE_PATH,
        help=(
            "Stage-0 output base path; creates {stem}_{epochs}/net/ and "
            "{stem}_{epochs}/keypoints/ with the checkpoint and canonical keypoints."
        ),
    )
    parser.add_argument(
        "--finetune-save-path",
        type=Path,
        default=DEFAULT_STAGE1_SAVE_PATH,
        help=(
            "Stage-1 output base path; creates {stem}_{epochs}/net/ and "
            "{stem}_{epochs}/keypoints/ with the checkpoint and canonical keypoints."
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
    logging.info("Selecting canonical keypoints from the mesh")
    keypoints = MeshKeypointSelector(
        final_keypoint_count=args.keypoint_count,
    ).select_keypoints(loader.mesh, keypoint_count=args.keypoint_count)
    logging.info("Loading dataset with images, ROI, and keypoint heatmaps")
    kp_dataset = loader.get_kp_heatmap_roi_dataset(
        keypoints,
        crop_size=args.crop_size,
        margin=args.margin,
    )

    use_tensorboard = not args.no_tensorboard
    stage0_paths = resolve_keypoint_run_paths(args.save_path, args.epochs)
    stage1_paths = resolve_keypoint_run_paths(args.finetune_save_path, args.epochs)
    save_canonical_keypoints(stage0_paths.keypoints, keypoints)

    logging.info("Stage 0: training for %d epochs (freeze_backbone=True)", args.epochs)
    logging.info("Saving stage-0 run artifacts under %s", stage0_paths.run_dir)
    trainer = KeypointHeatmapTrainer(
        kp_dataset,
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir="runs/keypoint_heatmap",
    )
    trainer.train(
        args.epochs,
        save_path=str(stage0_paths.net_checkpoint),
        freeze_backbone=True,
    )

    logging.info(
        "Stage 1: fine-tuning for %d epochs from %s (freeze_backbone=False, lr=1e-5)",
        args.epochs,
        stage0_paths.net_checkpoint,
    )
    logging.info("Saving stage-1 run artifacts under %s", stage1_paths.run_dir)
    save_canonical_keypoints(stage1_paths.keypoints, keypoints)
    trainer = KeypointHeatmapTrainer(
        kp_dataset,
        model=KeypointHeatmapNet.load(
            stage0_paths.net_checkpoint,
            num_keypoints=len(keypoints),
        ),
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir="runs/keypoint_heatmap_nf",
        optimizer_kwargs=dict(lr=1e-5),
    )
    trainer.train(
        args.epochs,
        save_path=str(stage1_paths.net_checkpoint),
        freeze_backbone=False,
    )


if __name__ == "__main__":
    main()
