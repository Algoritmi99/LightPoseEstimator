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

`--background-augmentation 0.5` replaces the surroundings of that share of keypoint training
crops with an object-free patch of another training image (or, a quarter of the time, a dark
plain background), so the keypoint net can't memorise the scenes it was trained on. On the
merged Astrobee data, unseen recordings with the ISS interior behind the target did far worse
than unseen ones with a plain background. The object's silhouette comes from the mesh at each
frame's labelled pose; masks and patches are rendered once into `--cache-dir` (default `cache/`;
about 85 MB of masks and 190 MB of patches for 19k frames) and reused. `--random-erasing 0.3`
puts a random rectangle of noise over that share of keypoint crops.

Data is split by recording, not by frame, so near-identical neighbouring frames never land
in both training and validation. By default 70% of recordings train, 15% validate (checkpoint
selection) and 15% are held out for testing (`--val-fraction`, `--test-fraction`, `--split-seed`).
The split is saved in the run's `config.json`. The per-stage scripts above use the same default split.
`--split-from RUN_DIR` reuses another run's split instead, so two runs are compared on the same
held-out recordings. `--exclude-recordings PATTERN...` leaves out recordings by shell-style name
pattern (e.g. `'rec_run_2026-05-22-*'`, whose renders show the payload window opaque); the run
stores the patterns, and `eval_pipe.py` and `calibrate_gates.py` apply them too.

```bash
python train.py --dataset full_dataset --output pose_estimator_astrobee_bg \
    --split-from pose_estimator_astrobee_full_30 --exclude-recordings 'rec_run_2026-05-22-*' \
    --background-augmentation 0.5 --random-erasing 0.3
```

`--symmetric-keypoints` is for near-symmetric objects like the Astrobee, which looks almost the
same after a 180-degree turn about any axis. On unseen recordings its heatmaps labelled keypoints
with their turned twins, and PnP returned poses flipped by 180 degrees. With the flag:

- keypoints come in sets of four that the turns map onto each other (a surface point and its
  turned copies, kept only where the copies are real counterparts on the surface), so
  `--keypoint-count` must be a multiple of 4;
- the heatmap loss takes the lowest of the four labellings the turns induce, held to a canonical
  one except near the boundaries between them, so the heatmaps never have to tell twins apart;
- a classifier on the keypoint net picks the turn from that labelling to the true one, and PnP
  pairs each heatmap with the 3D keypoint it then shows. It replaces the coarse-rotation symmetry
  correction.

Runs trained this way can't be exported yet (the C++ library doesn't handle the classifier).

Some views can't tell twins apart at all: the Astrobee's two propulsion modules are the same part
turned 180 degrees about the vertical axis, so only its front and back differ under that turn,
and seen with those edge-on the twins look alike. `train.py` saves, per turn, where the surface
differs (`keypoints/turn_distinctness.npz`). The classifier is trained to split its answer
between twins as far as that region faces away from the camera, and the estimator weighs it the
same way. When twins remain plausible, `PoseEstimationResult.candidates` lists each of them with
its pose and probability (`ambiguous` is True); `pose` is the likeliest.

Over an image stream the true pose moves a little from frame to frame while a twin is 180 degrees
away, so each twin can be followed as its own identity, and frames that do show the front or back
settle the others. A twin a frame solved no pose for keeps its last pose (`TrackedPose.measured` is
False when that is the one reported) instead of being swapped for a hypothesis 180 degrees away, so
one frame where PnP misses the true twin can't flip the stream onto the wrong one. Neighbouring
frames see nearly the same view, so their evidence is discounted, and steady contrary evidence can
still overturn a wrong lead. `tracking.TurnTracker` does this online (feed it every result in
order, also those without a pose; it reports the likeliest twin, whether it is `resolved`, and the
candidates until it is), and `tracking.smooth_turns` offline over a whole recording.
`eval_pipe.py --track` scores both on the held-out recordings that are continuous sequences,
including how many frames were resolved on the wrong twin, the mistake a docking controller can't
absorb.

#### On the payload

`PoseEstimator.track(frame, timestamp)` is the one call a navigation loop needs: call it for every
camera frame of the stream, in order. It estimates the frame (one network pass, quality checks
included) and keeps a `TurnTracker` that settles which twin the target is.

```python
estimator = PoseEstimator.load_run(run_dir, camera_intrinsics, distortion_coeffs, bayer_pattern="BG")
while True:
    frame, timestamp = camera.read()   # raw mosaic as a 2-D array, a path or a PIL image
    tracked = estimator.track(frame, timestamp)
    if tracked.position is not None:   # metres, camera frame; valid whether the twin is settled or not
        ...
    if tracked.attitude is not None:   # object-to-camera rotation, once the twin is settled
        ...
```

- `pose` is None when the frame gives no pose and `frame.rejected` says why (9.5% of frames on the
  held-out approaches: rotation disagreement, too few keypoints, low keypoint confidence).
- `position` holds whenever there is a pose: twins differ by a turn about the object's centre, so
  they share it. On the held-out approaches unsettled and even wrong-twin poses had the same
  position error as settled ones, about 1.7% of the distance (median).
- `attitude` is given once the twin is `resolved` (73.7% of frames on the held-out approaches, 2 of
  them on the wrong twin); until then `candidates` lists the twins in play.
- `measured` is False when the frame solved no pose for the reported twin, whose attitude is then
  its last one (0.2% of poses); `position` then comes from the frame's own estimate.
- `reset_tracking()` starts over, e.g. after the target was out of view for long;
  `reset_tracking(frame_interval=..., reset_after=...)` also sets the tracker's timing.
- Leave out `bayer_pattern` for a camera that delivers grayscale or colour images. Each camera made
  with `with_camera` keeps its own stream.

`estimate_tracked(image, last_pose)` instead takes the pose from the caller, e.g. a navigation
filter's prediction for this frame. Of the twins PnP solves it returns the one continuing
`last_pose`, with the quality checks applied to that twin, or no pose ("far from last pose") if it
turned more than `max_rotation_change` (45 degrees) or moved more than `max_translation_change`
(15% of the distance); on the held-out approaches true poses changed by at most 18.5 degrees and
the estimates by at most 20% of the distance between frames 0.4 s apart. It keeps whichever twin
`last_pose` is: fed with its own poses alone, 7 of the 15 held-out approaches started on the wrong
twin and 40% of its poses were, against 2 frames for `track`.

```bash
python train.py --dataset full_dataset --output pose_estimator_astrobee_sym \
    --split-from pose_estimator_astrobee_full_30 --exclude-recordings 'rec_run_2026-05-22-*' \
    --background-augmentation 0.5 --random-erasing 0.3 --keypoint-count 40 --symmetric-keypoints
```

Where the twins do differ, the classifier can still go by which twin the training crops of a view
mostly showed: on held-out recordings, informative frames were misread 16% of the time where
training mostly showed the view as the other twin, against 10% elsewhere. Two options counter this:

- `--force-train PATTERN...` puts recordings whose name matches into training even if the split
  holds them out. `full_dataset`'s numbered sets (`164_*`, `644_*`, `565_*`) render random poses
  and so show views as every twin, unlike the `rec_run_*` trajectories; with `--force-train '[0-9]*'`
  all of them train, and the held-out recordings are all `rec_run_*`. `--oversample PATTERN=FACTOR...`
  draws the keypoint net's training crops of matching recordings that many times as often.
- `--balance-twins` (with `--symmetric-keypoints`) weights each crop's turn-classifier loss so
  that, among training crops with a similar view (within 20 degrees), each twin seen there weighs
  the same in total, counting crops as often as training draws them.

```bash
python train.py --dataset full_dataset --output pose_estimator_astrobee_sym3 \
    --split-from pose_estimator_astrobee_full_30 --exclude-recordings 'rec_run_2026-05-22-*' \
    --force-train '[0-9]*' --oversample '[0-9]*=2' --balance-twins \
    --background-augmentation 0.5 --random-erasing 0.3 --keypoint-count 40 --symmetric-keypoints
```

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
All layouts are read into the same columns, so every dataset and trainer works on any of them.

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

Recordings rendered in Godot (`godot`, the merged Astrobee `full_dataset`; `GodotDataLoader`):
detected by a glTF mesh at the root and `_godot_imgs` folders. The mesh is in metres and, as for
EO-1, converted from glTF's Y-up axes to the Z-up model frame; the pose frame is its bounding-box
centre (recentred). `rec_run_*` images are named after their time stamp like the recordings layout;
numbered sets name image N `NNNNNNNNNN.jpg` for the CSV row with Timestamp N. `*_bayer` recordings
store the raw BG sensor mosaic and are converted to grayscale when read (`read_image`). A
`PoseEstimator` given the camera's `bayer_pattern` (`load_run`, `with_camera`) converts raw frames
the same way, whether passed as paths, PIL images or 2-D arrays.

```
full_dataset/
  astrobee10.glb
  rec_run_*/                 # __target_in_cam.csv, camera_intrinsics.json, _godot_imgs/1088_2.png
  164_back_bw/               # __target_in_cam.csv, camera_intrinsics.json, _godot_imgs/0000000001.jpg
  164_back_bayer/            # same, raw Bayer images
```

```bash
python train.py --dataset /home/algoritmi/data/LightPoseEstimator_data/data/full_dataset --output pose_estimator_full
```

```bash
python eval_pipe.py --run-dir pose_estimator_eo1_30 --dataset "EO1 Dataset" --max-translation-error 0.5
```

At 10-70 m, translation error is better judged relative to distance; `eval_pipe.py` reports
both, and a per-class breakdown when the labels have classes.
