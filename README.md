# Light Pose Estimator

Pose estimation and ROI tooling for Astrobee-style datasets (distorted camera images, mesh, and per-frame poses).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place datasets under `data/` (ignored by git).

## How to: estimate the target's pose in a navigation loop (Python)

`PoseEstimator.track` is the one call a navigation loop needs. Load a trained run once with the
camera's calibration, then call `track` for every camera frame, in order. Each call converts a
raw sensor frame if needed, estimates the frame's pose (one network pass, quality checks
included) and updates the estimator's twin tracker. That tracker settles, over the frames, which
way round the near-symmetric target is: it looks almost the same turned 180 degrees, so a single
frame can't always tell.

```python
import torch
from LightPoseEstim import PoseEstimator

estimator = PoseEstimator.load_run(
    "pose_estimator_astrobee_sym3_30",  # trained run directory
    camera_intrinsics=K,                # 3x3 camera matrix, as in camera_intrinsics.json
    distortion_coeffs=D,                # OpenCV distortion coefficients
    bayer_pattern="BG",                 # only for a camera that delivers its raw sensor mosaic
    device=torch.device("cpu"),
)

while True:
    frame, timestamp = camera.read()    # 2-D array, image file path or PIL image; seconds
    tracked = estimator.track(frame, timestamp)
    if tracked.position is not None:    # metres, camera frame
        ...
    if tracked.attitude is not None:    # rotation, object to camera frame
        ...
```

What `tracked` holds:

- `position`: the target's position whenever the frame gives a pose, also before the twin is
  settled, since the twins share it.
- `attitude`: the target's rotation (a `Quaternion`), only once the twin is settled
  (`resolved`). Before that it is None and `candidates` lists every twin still possible.
- `pose` is None when the frame gave no pose; `frame.rejected` says why.
- `measured` is False when this frame missed the reported twin: its attitude is then the
  previous one, while `position` is this frame's.

Notes:

- Call `track` on every frame of one continuous stream, frames without the target included.
  `reset_tracking()` starts over, e.g. after the target was out of view for long. For unrelated
  single images use `estimate`.
- One estimator object follows one stream. `estimator.with_camera(K, D, bayer_pattern)` makes
  another, with its own stream, for a second camera.
- Expected accuracy, measured on 15 held-out approaches from about 3 m to 0.4 m (Godot renders,
  frames 0.4 s apart): a pose on 90% of frames and an attitude on 74%, 0.1% of the attitudes on
  the wrong twin. Position error 1.8% of the distance (median) and 3.9% (90th percentile);
  attitude error 2.8 degrees (median) and 6.1 degrees (90th percentile). Real camera images may
  do worse.

## How to: estimate the target's pose in a navigation loop (C++)

`LightPoseEstimlib` runs the same pipeline on the payload without Python: the two networks from
an export of the trained run, everything around them in C++. As in Python, `PoseEstimator::track`
is the one call a navigation loop needs. Construct the estimator once, then call `track` for every
camera frame, in order. Each call converts a raw sensor frame if needed, estimates the frame's
pose (quality checks included) and updates the estimator's twin tracker, which settles over the
frames which way round the near-symmetric target is.

1. Export the trained run once, in Python, with the payload camera's calibration. The export holds
   compiled machine code for one device type and CPU: CPU exports need AVX2 by default; for
   another architecture (e.g. ARM) export on that architecture with `cpu_target="native"`.

   ```python
   PoseEstimator.load_run(
       "pose_estimator_astrobee_sym3_30",  # trained run directory
       camera_intrinsics=K,                # 3x3 camera matrix, as in camera_intrinsics.json
       distortion_coeffs=D,                # OpenCV distortion coefficients
       bayer_pattern="BG",                 # only for a camera that delivers its raw sensor mosaic
   ).export("export_sym3_cpu", device="cpu")
   ```

2. Build the library against LibTorch of the same PyTorch version as the export (it records the
   version, and the library refuses another), with Eigen3, OpenCV 4.5.3 or later and nlohmann_json:

   ```bash
   cmake -S LightPoseEstimlib -B LightPoseEstimlib/build -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
   cmake --build LightPoseEstimlib/build -j 8
   ```

   and link it into the navigation software's CMake project:

   ```cmake
   add_subdirectory(path/to/LightPoseEstimlib LightPoseEstimlib)  # build dir needed when outside the project
   target_link_libraries(navigation PRIVATE LightPoseEstimLib)
   ```

3. Call it in the loop:

   ```cpp
   #include "PoseEstimator.h"

   PoseEstimator estimator("export_sym3_cpu");          // throws if the export doesn't fit this LibTorch
   cv::Mat frame;                                       // 8-bit: BGR, grayscale, or the raw mosaic (1 channel)
   double timestamp;                                    // seconds
   while (camera.read(frame, timestamp)) {
       const TrackedPose tracked = estimator.track(frame, timestamp);
       if (const auto position = tracked.position()) {  // Eigen::Vector3d, metres, camera frame
           ...
       }
       if (const auto attitude = tracked.attitude()) {  // Quaternion (qx, qy, qz, qw), object to camera frame
           ...
       }
   }
   ```

What `tracked` holds, as in Python:

- `position()`: the target's position whenever the frame gives a pose, also before the twin is
  settled, since the twins share it.
- `attitude()`: the target's rotation, only once the twin is settled (`resolved`). Before that it
  is empty and `candidates` lists every twin still possible.
- `pose` is empty when the frame gave no pose; `frame->rejected` says why.
- `measured` is false when this frame missed the reported twin: its attitude is then the previous
  one, while `position()` is this frame's.

Notes:

- Call `track` on every frame of one continuous stream, frames without the target included.
  `resetTracking()` starts over, e.g. after the target was out of view for long. For unrelated
  single images use `estimate`; `estimateTracked` is as in Python.
- One estimator follows one stream and is not thread-safe: use one per camera and thread.
  `setCamera(K, D, bayerPattern)` switches to a camera other than the exported one and starts the
  stream over. A frame in RGB order instead of OpenCV's BGR: `track(frame, timestamp, false)`.
- CPU threads: `at::set_num_threads(n)`, and `OMP_NUM_THREADS=n` in the environment (the
  networks' kernels run on the system OpenMP runtime). On a desktop CPU (AMD Ryzen 9 9950X3D) a
  `track` call took 6 ms (median) with 4 or 8 threads; a payload CPU will be slower.
- The accuracy is Python's (see above). `LightPoseEstimlib/check_against_python.py` compares the
  library with Python frame by frame. On the 7 held-out test approaches (1,584 frames) it gave the
  same result on every frame with the same exported networks. Against the PyTorch networks, 94% of
  poses agreed within 0.1 degrees and the twin always; the rest differ by about the method's own
  error, where RANSAC settles on another inlier set. On raw Bayer frames the OpenCV versions
  convert the mosaic one grey level apart on many pixels (5.0 in Python, 4.10 here), which moved
  poses by 0.04 degrees (median).
- `main`, built with the library, runs an export on image files and prints one JSON line per
  frame (`--track` for a stream, `--camera`, `--bayer`): a quick check on the payload itself.
  The export's own README.md spells out every step of the pipeline.

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
Training runs on the GPU when CUDA is available; `--device` picks another (`cpu`, `cuda:1`).
The same holds across the package: every script with `--device`, and every class or function
with a `device` argument, uses CUDA when it is available unless told otherwise.
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

`--cache-images` decodes every image once instead of every epoch, for datasets whose images
decode slower than the networks train and leave the GPU waiting (EO-1's 2048x2048 PNGs). It keeps
what each network needs in memory-mapped files under `--cache-dir`, which the OS holds in RAM,
shared by all loader workers: the ROI detector's resized inputs, and the full-resolution part of
each frame that keypoint crops can come from (their randomly moved and scaled boxes included).
Training sees exactly the same data. For EO-1's 5000 frames that is about 26 GB, built in a
minute or two and reused while the images, labels and settings stay the same.

When training on a GPU, the colour jitter, blur, simulated low light and random erasing are
applied there, to each training batch once the loader workers hand it over, not to each image in
the workers. Every image still gets its own random draws from the same distributions as the
torchvision transforms they replace. On EO-1's 384 px colour keypoint crops they were about 60%
of the workers' CPU time (the hue jitter converts every crop to HSV and back), enough to leave
the GPU waiting. `--cpu-augmentation` applies them in the workers instead, as does training on
the CPU.

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

`PoseEstimator.export` writes such runs for the C++ library, which runs the classifier, the twins and
the tracking as Python does.

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

On the payload, `PoseEstimator.track` runs the estimator and a `TurnTracker` together, one frame
per call: see the how-tos for [Python](#how-to-estimate-the-targets-pose-in-a-navigation-loop-python)
and [C++](#how-to-estimate-the-targets-pose-in-a-navigation-loop-c).

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
