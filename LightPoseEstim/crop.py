"""
Crop geometry shared by training data and inference: ROI -> integer pixel box ->
resized crop -> heatmap cells. Coordinates use the OpenCV convention (an integer is
a pixel or cell centre), so targets and decoded peaks map back to the same place.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .roi import ROI, roi_to_bbox

HEATMAP_STRIDE = 4

Box = tuple[int, int, int, int]


def crop_box(roi: ROI, height: int, width: int) -> Box:
    """Integer pixel box (x1, y1, x2, y2), exclusive end, clamped to the image and at least 1 px."""
    bbox = roi_to_bbox(roi)
    x1 = int(max(0, np.floor(bbox.x1)))
    y1 = int(max(0, np.floor(bbox.y1)))
    x2 = int(min(width, np.ceil(bbox.x2)))
    y2 = int(min(height, np.ceil(bbox.y2)))
    if x2 <= x1:
        x1 = min(x1, width - 1)
        x2 = x1 + 1
    if y2 <= y1:
        y1 = min(y1, height - 1)
        y2 = y1 + 1
    return x1, y1, x2, y2


def crop_and_resize(image: torch.Tensor, box: Box, size: int) -> torch.Tensor:
    """Crop a [C, H, W] image to ``box`` and resize it to ``size`` x ``size``."""
    x1, y1, x2, y2 = box
    return F.interpolate(
        image[:, y1:y2, x1:x2].unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _scale(box: Box, out_size: tuple[int, int], stride: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    out_w, out_h = out_size
    return np.array([out_w / (x2 - x1), out_h / (y2 - y1)]) / stride


def image_to_crop(
    points: np.ndarray,
    box: Box,
    out_size: tuple[int, int],
    stride: int = 1,
) -> np.ndarray:
    """Full-image pixel coordinates -> coordinates in the resized crop, in units of ``stride`` pixels."""
    origin = np.array(box[:2], dtype=np.float64)
    return (np.asarray(points, dtype=np.float64) + 0.5 - origin) * _scale(box, out_size, stride) - 0.5


def crop_to_image(
    points: np.ndarray,
    box: Box,
    out_size: tuple[int, int],
    stride: int = 1,
) -> np.ndarray:
    """Inverse of :func:`image_to_crop`."""
    origin = np.array(box[:2], dtype=np.float64)
    return (np.asarray(points, dtype=np.float64) + 0.5) / _scale(box, out_size, stride) + origin - 0.5
