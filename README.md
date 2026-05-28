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
