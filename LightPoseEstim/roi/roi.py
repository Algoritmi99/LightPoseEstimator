from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ROI:
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float


def roi_to_bbox(roi: ROI) -> BBox:
    return BBox(
        roi.cx - roi.w / 2,
        roi.cy - roi.h / 2,
        roi.cx + roi.w / 2,
        roi.cy + roi.h / 2,
    )


def bbox_to_roi(bbox: BBox) -> ROI:
    return ROI(
        bbox.x1 + (bbox.x2 - bbox.x1) / 2,
        bbox.y1 + (bbox.y2 - bbox.y1) / 2,
        bbox.x2 - bbox.x1,
        bbox.y2 - bbox.y1,
    )


def visualize_roi(
    image,
    roi,
    projected_points=None,
    normalized: bool = True,
    title: str = "ROI Visualization",
):
    """Draw ROI box and optional projected 2D points on an image."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    h, w = image.shape[:2]

    if hasattr(roi, "detach"):
        roi = roi.detach().cpu().numpy()

    cx, cy, rw, rh = roi
    if normalized:
        cx *= w
        cy *= h
        rw *= w
        rh *= h

    x1 = int(cx - rw / 2)
    y1 = int(cy - rh / 2)
    x2 = int(cx + rw / 2)
    y2 = int(cy + rh / 2)

    vis = image.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(vis, (int(cx), int(cy)), 4, (255, 0, 0), -1)

    if projected_points is not None:
        if hasattr(projected_points, "detach"):
            projected_points = projected_points.detach().cpu().numpy()
        projected_points = np.asarray(projected_points, dtype=np.float32)
        if projected_points.ndim == 2 and projected_points.shape[1] == 2:
            in_bounds = (
                (projected_points[:, 0] >= 0) & (projected_points[:, 0] < w) &
                (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h)
            )
            for px, py in projected_points[in_bounds]:
                cv2.circle(vis, (int(px), int(py)), 1, (255, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.imshow(vis)
    plt.title(title)
    plt.axis("off")
    plt.show()
