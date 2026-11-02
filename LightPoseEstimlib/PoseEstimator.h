//
// Created by algoritmi on 10/22/26.
//

#ifndef LIGHTPOSEESTIMLIB_POSEESTIMATOR_H
#define LIGHTPOSEESTIMLIB_POSEESTIMATOR_H
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <opencv2/core.hpp>
#include <filesystem>
#include <optional>
#include <string>
#include "ExportConfig.h"
#include "PoseEstimationResult.h"
#include "TurnTracker.h"


// Runs a pose estimator exported with PoseEstimator.export() in Python: the two networks
// from their AOTInductor packages, and the geometry around them (crop, peak decoding, PnP,
// twins of a near-symmetric object, checks, tracking) as in the Python PoseEstimator.
// Not thread-safe: use one instance per thread.
//
// In a navigation loop, call track() once per camera frame, in order:
//
//     PoseEstimator estimator(exportDir);
//     while (camera.read(frame)) {
//         const TrackedPose tracked = estimator.track(frame, timestamp);
//         if (const auto position = tracked.position()) { ... }  // metres, camera frame
//         if (const auto attitude = tracked.attitude()) { ... }  // once the twin is settled
//     }
class PoseEstimator {
public:
    explicit PoseEstimator(const std::filesystem::path& exportDir);

    // One image: 8-bit as cv::imread gives it (BGR), RGB with isBgr = false, grayscale, or, with a
    // Bayer pattern set, the camera's raw sensor mosaic (one channel).
    PoseEstimationResult estimate(const cv::Mat& image, bool isBgr = true);
    // float32 RGB in [0, 1], [3, H, W] or [1, 3, H, W], on any device (no Bayer conversion).
    PoseEstimationResult estimate(const at::Tensor& image);

    // The call for a navigation loop: every frame of one continuous stream, in order, frames
    // without the target included. Estimates the frame and follows the object's twins over the
    // stream (TurnTracker): position() whenever there is a pose, attitude() once the twin is
    // settled; frame->rejected says why a frame gave no pose. timestamp in seconds (optional).
    TrackedPose track(const cv::Mat& image, std::optional<double> timestamp = std::nullopt, bool isBgr = true);
    TrackedPose track(const at::Tensor& image, std::optional<double> timestamp = std::nullopt);
    // Start track() over, e.g. after the target was out of view for long; optionally with other
    // settings (the export's by default), which stay in effect.
    void resetTracking();
    void resetTracking(const TrackingSettings& settings);

    // One frame given the pose just before it (the last estimate, or a prediction of it such as
    // a navigation filter's): of the twins PnP solved, the one closest in rotation to lastPose,
    // with the checks applied to that twin, or no pose ("far from last pose") if it moved more
    // than the limits (the export's continuity settings by default). It keeps whichever twin
    // lastPose is: use track() unless lastPose comes from a source you trust. Without lastPose,
    // this is estimate().
    PoseEstimationResult estimateTracked(const cv::Mat& image, const std::optional<Pose>& lastPose, bool isBgr = true,
                                         const std::optional<ContinuitySettings>& limits = std::nullopt);
    PoseEstimationResult estimateTracked(const at::Tensor& image, const std::optional<Pose>& lastPose,
                                         const std::optional<ContinuitySettings>& limits = std::nullopt);

    // Use another camera than the exported one: its calibration (OpenCV distortion order) and,
    // if it delivers its raw sensor mosaic, its Bayer pattern ("BG", "GB", "RG" or "GR").
    // Starts track() over.
    void setCamera(const Eigen::Matrix3d& cameraIntrinsics, const std::vector<double>& distortionCoeffs,
                   const std::optional<std::string>& bayerPattern = std::nullopt);

    const ExportConfig& getConfig() const { return config; }

private:
    struct Box {
        int64_t x1, y1, x2, y2;  // [x1, x2) x [y1, y2), clamped to the image
    };

    ExportConfig config;
    c10::Device device;
    torch::inductor::AOTIModelPackageLoader roiDetector;
    torch::inductor::AOTIModelPackageLoader keyPointNet;
    // Pixel value -> float32 in [0, 1], computed on the CPU as torchvision's ToTensor does
    // (dividing on the GPU rounds 126 of the 256 values differently, by 1 ulp): on the device,
    // and as an OpenCV lookup table for converting images on the CPU in one pass.
    at::Tensor pixelToFloat;
    cv::Mat pixelLut;
    // float32 like the Python estimator, so PnP sees the same numbers.
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Point3f> objectPoints;
    // Keypoints in symmetric sets: per turn, the 3D keypoint each heatmap shows.
    std::vector<std::vector<cv::Point3f>> turnObjectPoints;
    double maxValidRadius;
    std::optional<int> bayerToGray;  // cv::COLOR_Bayer..2GRAY for a raw sensor mosaic
    TrackingSettings trackingSettings;
    std::optional<TurnTracker> tracker;

    at::Tensor toTensor(const cv::Mat& image, bool isBgr) const;
    PoseEstimationResult run(const at::Tensor& image, const std::optional<Pose>& lastPose, const ContinuitySettings& limits);
    static bool roiOutsideFrame(const Roi& roi, int64_t height, int64_t width);
    static Box cropBox(const Roi& roi, int64_t height, int64_t width);
    // RANSAC PnP on the confident keypoints (shown[k]: the 3D point heatmap k shows); fills
    // inlier flags over all keypoints.
    std::optional<Pose> solvePose(const std::vector<cv::Point2f>& imagePoints,
                                  const std::vector<float>& confidence,
                                  double maxError,
                                  const std::vector<cv::Point3f>& shown,
                                  std::vector<bool>& inliers) const;
    // Keypoints in symmetric sets: PnP under every turn, each weighted by the classifier. Fills
    // result's turn fields and returns the chosen twin's pose and inliers, turning allocentric
    // into its frame; with lastPose, the twin continuing it (empty and result.rejected set if it
    // moved too far).
    std::optional<Pose> solveTurns(const std::vector<cv::Point2f>& imagePoints,
                                   const std::vector<float>& confidence,
                                   double maxError,
                                   const at::Tensor& turnLogits,
                                   Eigen::Matrix3d& allocentric,
                                   PoseEstimationResult& result,
                                   const std::optional<Pose>& lastPose,
                                   const ContinuitySettings& limits,
                                   std::vector<bool>& inliers) const;
    Pose resolveSymmetry(const Pose& pose, const Eigen::Matrix3d& allocentric) const;
    bool poseOutsideFov(const Pose& pose, int64_t height, int64_t width) const;
};


#endif //LIGHTPOSEESTIMLIB_POSEESTIMATOR_H
