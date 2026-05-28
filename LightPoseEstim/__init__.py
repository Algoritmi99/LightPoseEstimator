from .pose import Pose, Position, Quaternion
from .dataloader import DataLoader, ImagePoseDataset, ImageROIDataset
from .projection import project_mesh_vertices, project_points
from .keypoint_selector import MeshKeypointSelector, CanonicalKeypoint
from .roi import (
    BBox,
    ROI,
    ROIDetector,
    bbox_to_roi,
    denormalize_roi,
    normalize_roi,
    roi_to_bbox,
    visualize_roi,
)

# Backwards compatibility alias
project_mesh_points = project_mesh_vertices
