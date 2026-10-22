#ifndef LIGHTPOSEESTIMLIB_POSEESTIMATIONRESULT_H
#define LIGHTPOSEESTIMLIB_POSEESTIMATIONRESULT_H
#include <Eigen/Core>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include "Pose.h"


// Region of interest in pixels: box centre and size.
struct Roi {
    double cx = 0.0;
    double cy = 0.0;
    double w = 0.0;
    double h = 0.0;
};

struct PoseEstimationResult {
    std::optional<Pose> pose;                 // x_camera = R x_object + t
    Roi roi;
    std::vector<Eigen::Vector2d> imagePoints; // per keypoint, full-image pixels
    std::vector<double> confidence;           // per keypoint; 0 below the peak threshold
    int inliers = 0;
    // Mean heatmap peak of the PnP inliers, and degrees between the PnP rotation and the
    // keypoint net's coarse whole-crop rotation: both drop/grow when the evidence is poor.
    double keypointConfidence = std::numeric_limits<double>::quiet_NaN();
    double rotationDisagreement = std::numeric_limits<double>::quiet_NaN();
    // Why pose is empty: "no target in view", "too few keypoints", "outside field of view",
    // "rotation disagreement" or "low keypoint confidence"; empty when there is a pose.
    std::string rejected;
};


#endif //LIGHTPOSEESTIMLIB_POSEESTIMATIONRESULT_H
