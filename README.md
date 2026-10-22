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

`train.py` trains the ROI detector, selects keypoints and trains the keypoint net into one
run directory. This is the recommended way to train; the per-stage scripts above still use the
older frozen-backbone schedule.

```bash
python train.py --dataset data/Astrobee --epochs 30
```

Each network trains in one stage with the whole model unfrozen, at `--lr` (default 1e-3)
decaying to 0 over `--epochs`, and stops early after `--patience` epochs without a better
validation loss. `--seed` fixes keypoint selection and initialisation, so runs are comparable.
`--low-light-augmentation 0.3` darkens that share of training images to simulate poor
lighting; it is off by default because on EO-1 it did not help dim frames and cost some accuracy.

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

### Pose quality checks

A pose is rejected (no pose instead of a wrong one) when its evidence is poor, typically a
target too dark to see:

- the PnP rotation disagrees with the keypoint net's coarse whole-crop rotation by more than
  `max_rotation_disagreement` degrees, or
- the PnP inliers' mean heatmap confidence is below `min_keypoint_confidence`.

`train.py` calibrates both on the validation frames after training, so each keeps 99% of good
poses (`--gate-keep-fraction`), and stores them in `config.json`. A run whose orientation head is
unreliable gets no rotation check. For runs trained before this existed:

```bash
python calibrate_gates.py --run-dir pose_estimator_eo1_30 --dataset "EO1 Dataset"
```

`PoseEstimationResult.rejected` says why a frame got no pose; `eval_pipe.py` counts the reasons
and `--no-quality-gates` evaluates without the checks.

## Dataset layouts

`train.py` and `eval_pipe.py` detect the layout (`--dataset-format` overrides it) and take the
mesh units and recentring from it unless `--mesh-scale` / `--[no-]recenter-mesh` are given.
Both layouts are read into the same columns, so every dataset and trainer works on either.

Recordings (`recordings`, the Astrobee data; `DataLoader`): mesh in mm, pose frame at the mesh's
bounding-box centre (so it is scaled by 0.001 and recentred).

```
data/Astrobee/
  model.stl
  rec_run_*/
    camera_intrinsics.json   # K, D
    __target_in_cam.csv
    *_imgs/
    *_imgs_undistorted/
```

Single folder of independent renders (`eo1`, the Polimi EO-1 data; `EO1DataLoader`): mesh in
metres, poses relative to the model origin (not recentred). The glTF mesh is converted from
glTF's Y-up axes to the labels' Z-up model frame, and the labels' MATLAB-convention quaternion
to scipy's. Truncated PNGs are skipped with a warning. Frames are grouped in blocks of 50 for
the train/val/test split.

```
EO1 Dataset/
  EO1.glb
  camera_intrinsics.json     # K (pinhole, no distortion)
  pose_labels.json           # per frame: timestamp, class, visibility, translation, rotation
  dataset/
    main_camera_<k>.000.png
```

```bash
python train.py --dataset "EO1 Dataset" --epochs 30 --output pose_estimator_eo1
```

```bash
python eval_pipe.py --run-dir pose_estimator_eo1_30 --dataset "EO1 Dataset" --max-translation-error 0.5
```

At 10-70 m, translation error is better judged relative to distance; `eval_pipe.py` reports
both, and a per-class breakdown when the labels have classes.
