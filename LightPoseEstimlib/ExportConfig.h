#ifndef LIGHTPOSEESTIMLIB_EXPORTCONFIG_H
#define LIGHTPOSEESTIMLIB_EXPORTCONFIG_H
#include <Eigen/Core>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>


struct PnpSettings {
    int minInliers;
    double reprojectionErrorCells;
    double minReprojectionErrorPx;
    int ransacIterations;
    double ransacConfidence;
};

// Everything in pose_estimator.json, written by PoseEstimator.export() in Python.
struct ExportConfig {
    static constexpr int supportedFormatVersion = 1;

    std::string torchVersion;                       // PyTorch version the packages were built with
    std::string device;                             // "cpu" or "cuda"
    std::filesystem::path roiDetectorPackage;       // absolute paths to the .pt2 packages
    std::filesystem::path keypointNetPackage;
    std::vector<int64_t> roiInternalInputSize;      // (width, height) the ROI detector resizes to
    std::vector<int> keypointIds;
    std::vector<Eigen::Vector3d> keypointPositions; // object frame, metres; row k = heatmap channel k
    Eigen::Matrix3d cameraIntrinsics;
    std::vector<double> distortionCoeffs;           // OpenCV order; empty = no distortion
    std::optional<double> maxValidRadius;           // limit of the lens model; empty = unlimited
    int cropSize = 0;
    int heatmapStride = 0;
    double peakThreshold = 0.0;
    PnpSettings pnp{};
    bool resolveSymmetry = false;
    std::vector<Eigen::Matrix3d> symmetryTurns;     // identity first
    std::optional<double> maxRotationDisagreementDeg; // empty = check off
    std::optional<double> minKeypointConfidence;      // empty = check off

    static ExportConfig load(const std::filesystem::path& exportDir);
};


#endif //LIGHTPOSEESTIMLIB_EXPORTCONFIG_H
