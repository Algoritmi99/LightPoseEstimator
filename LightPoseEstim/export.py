"""
Export a PoseEstimator for running outside Python (e.g. C++ with LibTorch).

The networks' forward passes are compiled with torch.export + AOTInductor into one
package per network. Everything around them (ROI box, crop, peak decoding, PnP, symmetry
and quality checks) stays plain geometry: its settings go into pose_estimator.json and
each step is spelled out in the generated README.md, so it can be reimplemented as is.
"""
import contextlib
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.export import Dim

from .projection import max_valid_radius

EXPORT_FORMAT_VERSION = 1
MANIFEST_NAME = "pose_estimator.json"
README_NAME = "README.md"
ENTRY_POINT = "forward"
_VERIFY_TOLERANCE = 1e-3


class _Entry(torch.nn.Module):
    """Expose model.<method> as forward(); torch.export only captures forward."""

    def __init__(self, model: torch.nn.Module, method: str):
        super().__init__()
        self.model, self.method = model, method

    def forward(self, *args):
        return getattr(self.model, self.method)(*args)


def export_model(model, entry_points, out_dir, dynamic_shapes=None) -> dict[str, Path]:
    """
    model:          any nn.Module
    entry_points:   {method_name: example_args (tuple of tensors)}
    dynamic_shapes: optional {method_name: dynamic_shapes spec}
    Returns {method_name: package path}.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    packages = {}
    with torch.no_grad():
        for name, args in entry_points.items():
            ep = torch.export.export(_Entry(model, name), args,
                                     dynamic_shapes=(dynamic_shapes or {}).get(name))
            packages[name] = Path(torch._inductor.aoti_compile_and_package(ep, package_path=str(out / f"{name}.pt2")))
    return packages


def _max_difference(expected, actual) -> float:
    expected = expected if isinstance(expected, (tuple, list)) else (expected,)
    actual = actual if isinstance(actual, (tuple, list)) else (actual,)
    return max(float((e - a).abs().max()) for e, a in zip(expected, actual))


@contextlib.contextmanager
def _full_float32():
    """
    Compare in full float32. On CUDA, TF32 convolutions (on by default) alone move outputs by
    ~1e-3 relative, in PyTorch and in the packages alike, which would hide real mismatches.
    """
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


def _verify(package: Path, model: torch.nn.Module, examples: list[torch.Tensor]) -> float:
    """Largest output difference between a package and the eager model over ``examples``."""
    runner = torch._inductor.aoti_load_package(str(package))
    with torch.no_grad(), _full_float32():
        return max(_max_difference(model(x), runner(x)) for x in examples)


def export_pose_estimator(estimator, out_dir: str | Path, device: str | torch.device | None = None, verify: bool = True) -> Path:
    """See PoseEstimator.export."""
    from . import pose_estimator as pe

    out = Path(out_dir)
    device = torch.device(device) if device is not None else estimator.device
    if device.type == "cuda":
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is None:
            raise RuntimeError(
                "Compiling CUDA packages needs a CUDA toolkit (set CUDA_HOME to it), "
                "or export for the CPU with device='cpu'."
            )

    # Copies, so exporting never moves the estimator's own networks to another device.
    roi_detector = copy.deepcopy(estimator.roi_detector).to(device).eval()
    keypoint_net = copy.deepcopy(estimator.keypoint_net).to(device).eval()
    crop_size = int(estimator.crop_size)
    stride = int(keypoint_net.stride)
    heatmap_size = -(-crop_size // stride)
    num_keypoints = len(estimator.keypoints)

    # The ROI detector resizes internally, so it accepts any camera image size.
    roi_example = torch.rand(1, 3, 960, 1280, device=device)
    roi_package = export_model(
        roi_detector,
        {ENTRY_POINT: (roi_example,)},
        out / "roi_detector",
        dynamic_shapes={ENTRY_POINT: {"args": ({2: Dim("height", min=16, max=16384), 3: Dim("width", min=16, max=16384)},)}},
    )[ENTRY_POINT]
    keypoint_package = export_model(
        keypoint_net,
        {ENTRY_POINT: (torch.rand(1, 3, crop_size, crop_size, device=device),)},
        out / "keypoint_net",
    )[ENTRY_POINT]

    verification = None
    if verify:
        verification = {
            "roi_detector_max_abs_difference": _verify(
                roi_package, roi_detector,
                [torch.rand(1, 3, 960, 1280, device=device), torch.rand(1, 3, 2048, 2048, device=device)],
            ),
            "keypoint_net_max_abs_difference": _verify(
                keypoint_package, keypoint_net, [torch.rand(1, 3, crop_size, crop_size, device=device)],
            ),
        }
        worst = max(verification.values())
        if worst > _VERIFY_TOLERANCE:
            raise RuntimeError(f"Exported networks differ from PyTorch by up to {worst:.2e}.")

    distortion = None if estimator.distortion_coeffs is None else np.asarray(estimator.distortion_coeffs, dtype=np.float64).reshape(-1)
    fold_radius = max_valid_radius(distortion)
    manifest = {
        "format_version": EXPORT_FORMAT_VERSION,
        "torch_version": torch.__version__,
        "device": device.type,
        "networks": {
            "roi_detector": {
                "package": roi_package.relative_to(out).as_posix(),
                "input": {"name": "image", "shape": [1, 3, "H", "W"], "dtype": "float32"},
                "output": {"name": "roi", "shape": [1, 4], "dtype": "float32"},
                "internal_input_size": list(roi_detector.input_size),
            },
            "keypoint_net": {
                "package": keypoint_package.relative_to(out).as_posix(),
                "input": {"name": "crop", "shape": [1, 3, crop_size, crop_size], "dtype": "float32"},
                "outputs": [
                    {"name": "heatmap_logits", "shape": [1, num_keypoints, heatmap_size, heatmap_size], "dtype": "float32"},
                    {"name": "rotation_6d", "shape": [1, 6], "dtype": "float32"},
                ],
            },
        },
        "keypoints": {
            "ids": [int(kp.keypoint_id) for kp in estimator.keypoints],
            "positions": np.asarray(estimator.object_points, dtype=np.float64).round(9).tolist(),
        },
        "camera": {
            "intrinsics": np.asarray(estimator.camera_intrinsics, dtype=np.float64).tolist(),
            "distortion_coeffs": None if distortion is None else distortion.tolist(),
            "max_valid_radius": None if not np.isfinite(fold_radius) else round(float(fold_radius), 6),
        },
        "crop_size": crop_size,
        "heatmap_stride": stride,
        "peak_threshold": float(estimator.peak_threshold),
        "pnp": {
            "min_inliers": pe._MIN_PNP_INLIERS,
            "reprojection_error_cells": pe._REPROJECTION_ERROR_CELLS,
            "min_reprojection_error_px": pe._MIN_REPROJECTION_ERROR_PX,
            "ransac_iterations": 100,
            "ransac_confidence": 0.99,
        },
        "resolve_symmetry": bool(estimator.resolve_symmetry),
        "symmetry_turns": [turn.astype(int).tolist() for turn in pe._SYMMETRY_TURNS],
        "max_rotation_disagreement_deg": estimator.max_rotation_disagreement,
        "min_keypoint_confidence": estimator.min_keypoint_confidence,
        "verification": verification,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    (out / README_NAME).write_text(_readme(manifest))
    return out


def _tf32_note(m: dict) -> str:
    if m["device"] != "cuda":
        return ""
    return """
On CUDA, LibTorch (like PyTorch) runs convolutions in TF32 by default, which moves network
outputs by about 1e-3 relative to full float32. Either is fine for pose accuracy; for results
closest to the full-precision reference (as in `verification`), turn it off once at startup:
`at::globalContext().setAllowTF32CuDNN(false);`
"""


def _readme(m: dict) -> str:
    roi, kp, pnp = m["networks"]["roi_detector"], m["networks"]["keypoint_net"], m["pnp"]
    s, k = m["crop_size"], len(m["keypoints"]["ids"])
    heatmap = kp["outputs"][0]["shape"][-1]
    radius = m["camera"]["max_valid_radius"]
    checks = []
    if m["max_rotation_disagreement_deg"] is not None:
        checks.append(f"   - rotation disagreement > {m['max_rotation_disagreement_deg']} deg -> \"rotation disagreement\"")
    if m["min_keypoint_confidence"] is not None:
        checks.append(f"   - keypoint confidence < {m['min_keypoint_confidence']} -> \"low keypoint confidence\"")
    checks = "\n".join(checks) if checks else "   - (no quality checks were calibrated for this run)"
    symmetry = (
        "Enabled. coarse = ray_rotation(t) @ R_allo. Of R @ T for each T in `symmetry_turns`\n"
        "   (identity first), keep the one with the largest trace((R @ T)^T @ coarse); t is unchanged."
        if m["resolve_symmetry"] else "Disabled for this run; skip."
    )
    return f"""# Exported pose estimator

Exported with PyTorch {m['torch_version']} for `{m['device']}`. All values below come from
`{MANIFEST_NAME}`; this file describes how to run the pipeline around the two networks.

## Files

- `{roi['package']}`: ROI detector forward pass (AOTInductor package).
- `{kp['package']}`: keypoint network forward pass (AOTInductor package).
- `{MANIFEST_NAME}`: keypoints, camera, and every setting used below.

## Running the networks in C++

Use LibTorch built for the same PyTorch version ({m['torch_version']}) and device type
({m['device']}). The packages are compiled code for that combination.

```cpp
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

torch::inductor::AOTIModelPackageLoader roi_detector("{roi['package']}");
torch::inductor::AOTIModelPackageLoader keypoint_net("{kp['package']}");
std::vector<at::Tensor> roi = roi_detector.run({{image}});   // roi[0]: [1, 4]
std::vector<at::Tensor> kp = keypoint_net.run({{crop}});     // kp[0]: heatmap logits, kp[1]: rotation_6d
```

Inputs are float32 NCHW tensors, RGB (OpenCV loads BGR: convert), values in [0, 1], on the
export device. ImageNet normalisation and the ROI detector's resize to
{roi['internal_input_size'][0]}x{roi['internal_input_size'][1]} happen inside the networks.
{_tf32_note(m)}
## Pipeline

Coordinates follow OpenCV: an integer is a pixel or heatmap-cell centre. `W`, `H` are the
camera image width and height; `K`, `D` are `camera.intrinsics` and `camera.distortion_coeffs`
(OpenCV order; null means no distortion). Every step can end with "no pose" and a reason.

1. **ROI.** Run the ROI detector on the whole image `[1, 3, H, W]`. Output `(cx, cy, w, h)`
   is the box centre and size as fractions of W and H: multiply x values by W, y by H.
   No pose ("no target in view") when w <= 1 px or h <= 1 px, or when the box clipped to the
   image is at most 1 px wide or high.

2. **Crop box.** `x1 = max(0, floor(cx - w/2))`, `y1 = max(0, floor(cy - h/2))`,
   `x2 = min(W, ceil(cx + w/2))`, `y2 = min(H, ceil(cy + h/2))`. If `x2 <= x1`:
   `x1 = min(x1, W - 1)`, `x2 = x1 + 1` (same for y). The box is `[x1, x2) x [y1, y2)`.

3. **Crop.** Resize `image[:, :, y1:y2, x1:x2]` to {s}x{s} with bilinear interpolation,
   `align_corners = false`, no antialiasing (LibTorch: `torch::nn::functional::interpolate`
   with those options matches training exactly).

4. **Keypoint network.** Input `[1, 3, {s}, {s}]`. Outputs heatmap logits
   `[1, {k}, {heatmap}, {heatmap}]` (one channel per keypoint, in `keypoints.ids` order) and
   `rotation_6d` `[1, 6]`.

5. **Peaks.** For each keypoint channel: `p = sigmoid(logits)`; `(iy, ix)` = position of the
   maximum, `confidence = p[iy, ix]`, set to 0 when below `peak_threshold`
   ({m['peak_threshold']}). Refine below one cell on `L = log(max(p, 1e-6))`, per axis: if
   `0 < ix < {heatmap} - 1` and `c = L[ix+1] - 2 L[ix] + L[ix-1] < 0`, then
   `dx = clamp(0.5 (L[ix-1] - L[ix+1]) / c, -0.5, 0.5)`, else `dx = 0` (same for y with
   neighbours along y). Python does this step in float32.

6. **Back to the image.** With `stride = heatmap_stride` ({m['heatmap_stride']}):
   `u = (ix + dx + 0.5) * stride * (x2 - x1) / {s} + x1 - 0.5`,
   `v = (iy + dy + 0.5) * stride * (y2 - y1) / {s} + y1 - 0.5`, in double precision, then
   rounded to float32 for PnP.

7. **PnP.** Use keypoints with `confidence >= peak_threshold`; at least {pnp['min_inliers']} needed,
   else no pose ("too few keypoints"). Object points are `keypoints.positions` (object
   frame). `cell_px = stride * max(x2 - x1, y2 - y1) / {s}`,
   `max_error = max({pnp['min_reprojection_error_px']}, {pnp['reprojection_error_cells']} * cell_px)` pixels.
   `cv::solvePnPRansac(object, image, K, D, rvec, tvec, false, {pnp['ransac_iterations']}, max_error, {pnp['ransac_confidence']}, inliers, cv::SOLVEPNP_SQPNP)`.
   Then recount inliers yourself: points whose reprojection (`cv::projectPoints`) is within
   `max_error` px. With 4-5 points OpenCV reports every point as an inlier. Need at least
   {pnp['min_inliers']} ("too few keypoints"). Refine with `cv::solvePnPRefineLM` on the inliers.
   Pose: `x_camera = R x_object + t`, `R = Rodrigues(rvec)`.

8. **Coarse rotation.** `R_allo = [a, b, a x b]` with `a = normalize(r6[0:3])`,
   `b = normalize(r6[3:6] - (a . r6[3:6]) a)` (columns). `ray_rotation(t)`: `r = t / |t|`,
   `n = (0, 0, 1) x r`; identity if `|n| < 1e-12`, else rotation about `n / |n|` by
   `atan2(|n|, r_z)`. `coarse = ray_rotation(t) @ R_allo`.

9. **Symmetry.** {symmetry}

10. **Checks**, in this order:
   - no keypoint projects inside the image -> "outside field of view". Only keypoints with
     camera z > 1e-6 and `hypot(x, y) / z <= max_valid_radius` ({'unlimited' if radius is None else radius}) count; project
     them with `cv::projectPoints(K, D)` and test `0 <= u < W`, `0 <= v < H`.
   - `rotation_disagreement = acos(clamp((trace(R^T coarse) - 1) / 2, -1, 1))` in degrees.
   - `keypoint_confidence` = mean confidence of the PnP inliers.
{checks}

The pose returned is `(R, t)` from step 7 (or 9). Reported alongside: number of inliers,
keypoint confidence and rotation disagreement.

## Checking a port against Python

Compare intermediate values with the Python `PoseEstimator` on the same images: ROI, crop
box, confidences and image points should agree to about 1e-4 px. Final poses can still
differ on a few percent of frames: when two inlier sets are about equally well supported,
RANSAC's choice flips on last-bit differences, and the exported networks already differ from
PyTorch by up to {m['verification']['keypoint_net_max_abs_difference'] if m['verification'] else 1e-6:.1e}.
Such poses differ by about the method's own error.
"""
