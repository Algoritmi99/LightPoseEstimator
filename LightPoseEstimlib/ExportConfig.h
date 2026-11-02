#ifndef LIGHTPOSEESTIMLIB_EXPORTCONFIG_H
#define LIGHTPOSEESTIMLIB_EXPORTCONFIG_H
#include <Eigen/Core>
#include <cstdint>
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

// Where the object differs from its turned twins (TurnDistinctness in Python): surface samples
// in the object frame, their normals, and per turn which samples land off the surface when turned.
struct TurnDistinctness {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<std::vector<uint8_t>> differs;  // [turn][sample]; row 0 (identity) all 0
    double low = 0.0;                           // distinct_range: share of the visible surface
    double high = 0.0;                          // that differs, from "can't tell" to "can"

    // Per turn, in [0, 1]: how well a view at this pose (object to camera) tells that turn's
    // twins apart; 1 for the identity.
    std::vector<double> informativeness(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) const;
};

// Keypoints in symmetric sets (train.py --symmetric-keypoints): the object looks almost the same
// after each turn, and heatmap j shows keypoint permutation[turn][j] under that turn's labelling.
struct SymmetricKeypoints {
    std::vector<Eigen::Matrix3d> turns;           // identity first
    std::vector<std::vector<int>> permutation;    // [turn][heatmap]
    std::vector<std::vector<int>> compose;        // [a][b]: index of turns[a] * turns[b]
    std::optional<TurnDistinctness> distinctness;
    double candidateMinProbability = 0.2;         // twins at least this likely are candidates
};

// Following the twins over a stream (tracking.TurnTracker in Python).
struct TrackingSettings {
    double temperature = 2.0;        // a frame's evidence counts 1/temperature
    double commitMargin = 4.0;       // log-odds lead that resolves the twin
    double beliefCap = 20.0;         // no twin's log belief falls further behind the likeliest's
    double gateDeg = 45.0;           // rotation change between frames that still continues a twin
    double maxGateDeg = 90.0;
    double missingEvidence = 0.25;   // for a twin the frame solved no pose for
    int maxMissed = 25;              // frames a twin other than the likeliest is kept without a pose
    std::optional<double> frameInterval;  // seconds between frames; empty: count the frames
    std::optional<double> resetAfter;     // seconds without a pose before starting over; empty: never
    double candidateMinProbability = 0.2;
};

// estimateTracked's limits on how far a frame's pose may move from the last pose.
struct ContinuitySettings {
    double maxRotationChangeDeg = 45.0;
    double maxTranslationChange = 0.15;  // share of the last pose's distance
};

// Everything in pose_estimator.json, written by PoseEstimator.export() in Python.
struct ExportConfig {
    static constexpr int supportedFormatVersion = 2;  // and 1, which has none of the fields marked v2

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
    std::optional<std::string> bayerPattern;        // v2: "BG", "GB", "RG" or "GR" for a raw sensor mosaic
    int cropSize = 0;
    int heatmapStride = 0;
    double peakThreshold = 0.0;
    PnpSettings pnp{};
    bool resolveSymmetry = false;
    std::vector<Eigen::Matrix3d> symmetryTurns;     // identity first
    std::optional<SymmetricKeypoints> symmetric;    // v2
    std::optional<double> maxRotationDisagreementDeg; // empty = check off
    std::optional<double> minKeypointConfidence;      // empty = check off
    ContinuitySettings continuity;                  // v2
    TrackingSettings tracking;                      // v2

    static ExportConfig load(const std::filesystem::path& exportDir);
};


#endif //LIGHTPOSEESTIMLIB_EXPORTCONFIG_H
