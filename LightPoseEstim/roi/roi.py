from dataclasses import dataclass


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
    return BBox(roi.cx - roi.w / 2, roi.cy - roi.h / 2, roi.cx + roi.w / 2, roi.cy + roi.h / 2)

def bbox_to_roi(bbox: BBox) -> ROI:
    return ROI(bbox.x1 + (bbox.x2 - bbox.x1) / 2, bbox.y1 + (bbox.y2 - bbox.y1) / 2, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
