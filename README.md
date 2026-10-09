# Light Pose Estimator

Pose estimation and ROI tooling for Astrobee-style datasets (distorted camera images, mesh, and per-frame poses).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place datasets under `data/` (ignored by git).

## Visualize ROI

```bash
python vis_bounding_box.py --dataset data/Astrobee --sample-idx 150 --margin 1.3
```

Options:

- `--mesh-scale` — scale mesh units to match pose (default `0.001` for mm mesh + m poses)
- `--no-recenter-mesh` — skip translating mesh bbox center to origin
- `--margin` — ROI padding factor around projected bbox

## Train ROI detector

Stage 0 (from scratch, backbone frozen by default):

```bash
python train_roi_0.py --dataset data/Astrobee --epochs 20
```

Stage 1 (fine-tune from a prior checkpoint):

```bash
python train_roi_1.py --model roi_detector_20.pth --epochs 20
```

Checkpoints are saved as `{stem}_{epochs}.pth` (e.g. `roi_detector_20.pth`).

Evaluate a trained ROI model:

```bash
python eval_roi.py --model roi_detector_nf_20.pth
```

## Train keypoint heatmap net

Stage 0 selects canonical mesh keypoints, then trains the heatmap network:

```bash
python train_kp_0.py --dataset data/Astrobee --epochs 20 --keypoint-count 24
```

Stage 1 fine-tunes from a prior run directory or checkpoint:

```bash
python train_kp_1.py --model keypoint_heatmap_20 --epochs 20
```

Run artifacts are written under `{save_stem}_{epochs}/`:

```
keypoint_heatmap_20/
  net/keypoint_heatmap.pth
  keypoints/keypoints.npz
```

## Train and evaluate the full pipeline

`train.py` runs both ROI stages and both keypoint stages into one run directory:

```bash
python train.py --dataset data/Astrobee --epochs 20
```

Data is split by recording, not by frame, so near-identical neighbouring frames never land
in both training and validation. By default 70% of recordings train, 15% validate (checkpoint
selection) and 15% are held out for testing (`--val-fraction`, `--test-fraction`, `--split-seed`).
The split is saved in the run's `config.json`. The per-stage scripts above use the same default split.

Evaluate the run on its held-out test recordings:

```bash
python eval_pipe.py --run-dir pose_estimator_20 --dataset data/Astrobee
```

It reports how often a pose is returned, the translation and rotation error, and the share of
frames within `--max-translation-error` (default 5 cm) and `--max-rotation-error` (default 5°).
`--csv results.csv` writes per-frame results.

## Dataset layout

```
data/Astrobee/
  model.stl
  rec_run_*/
    camera_intrinsics.json   # K, D
    __target_in_cam.csv
    *_imgs/
    *_imgs_undistorted/
```
