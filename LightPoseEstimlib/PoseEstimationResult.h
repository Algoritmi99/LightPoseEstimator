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

// A pose for one of the object's turned twins, and how likely it is the true one.
struct TurnHypothesis {
    int turn;            // index into the exported turns (symmetric_keypoints.turns)
    double probability;
    Pose pose;
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
    // "rotation disagreement", "low keypoint confidence" or, from estimateTracked, "far from last
    // pose"; empty when there is a pose.
    std::string rejected;
    // Keypoints in symmetric sets: the turn of pose (the classifier's choice, or from
    // estimateTracked the twin continuing the last pose; -1 for other runs) and the classifier's
    // probability for it.
    int turn = -1;
    double turnConfidence = std::numeric_limits<double>::quiet_NaN();
    // Keypoints in symmetric sets: a pose for every turn PnP could solve, likeliest first.
    std::vector<TurnHypothesis> hypotheses;

    // Twins this view can't rule out, likeliest first: one for a clear result, more when the
    // view can't tell them apart (a near-symmetric object seen from its symmetric side).
    std::vector<TurnHypothesis> candidates(double minProbability = 0.2) const {
        std::vector<TurnHypothesis> out;
        for (const TurnHypothesis& h : hypotheses)
            if (h.probability >= minProbability) out.push_back(h);
        if (out.empty() && !hypotheses.empty()) out.push_back(hypotheses.front());
        return out;
    }

    bool ambiguous(double minProbability = 0.2) const {
        return pose.has_value() && candidates(minProbability).size() > 1;
    }
};


#endif //LIGHTPOSEESTIMLIB_POSEESTIMATIONRESULT_H
