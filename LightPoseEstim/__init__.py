from .pose import Pose, Position, Quaternion
from .projection import project_mesh_vertices, project_points
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
from .dataloader import (
    DataLoader,
    EO1DataLoader,
    ImageKPHeatmapDataset,
    ImageKPHeatmapROIDataset,
    ImagePoseDataset,
    ImageROIDataset,
    open_dataset,
)
from .kp import KeypointHeatmapNet, KeypointHeatmapTrainer, heatmap_peaks
from .roi.roi_detector_trainer import ROIDetectorTrainer
from .pose_estimator import PoseEstimationResult, PoseEstimator
from .keypoint_selector import (
    CanonicalKeypoint,
    KeypointRunPaths,
    MeshKeypointSelector,
    load_canonical_keypoints,
    paths_from_keypoint_model,
    paths_from_keypoint_run_dir,
    resolve_keypoint_run_paths,
    save_canonical_keypoints,
)

# Backwards compatibility alias
project_mesh_points = project_mesh_vertices
