//
// Created by algoritmi on 10/22/26.
//

#ifndef LIGHTPOSEESTIMLIB_POSEESTIMATOR_H
#define LIGHTPOSEESTIMLIB_POSEESTIMATOR_H
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <opencv2/core.hpp>
#include <filesystem>
#include "ExportConfig.h"
#include "PoseEstimationResult.h"


// Runs a pose estimator exported with PoseEstimator.export() in Python: the two networks
// from their AOTInductor packages, and the geometry around them (crop, peak decoding, PnP,
// checks) as in the Python PoseEstimator. Not thread-safe: use one instance per thread.
class PoseEstimator {
public:
    explicit PoseEstimator(const std::filesystem::path& exportDir);

    // 8-bit image as cv::imread gives it (BGR), RGB with isBgr = false, or grayscale.
    PoseEstimationResult estimate(const cv::Mat& image, bool isBgr = true);
    // float32 RGB in [0, 1], [3, H, W] or [1, 3, H, W], on any device.
    PoseEstimationResult estimate(const at::Tensor& image);

    // Use another camera calibration than the exported one (OpenCV distortion order).
    void setCamera(const Eigen::Matrix3d& cameraIntrinsics, const std::vector<double>& distortionCoeffs);

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
    double maxValidRadius;

    static bool roiOutsideFrame(const Roi& roi, int64_t height, int64_t width);
    static Box cropBox(const Roi& roi, int64_t height, int64_t width);
    // RANSAC PnP on the confident keypoints; fills inlier flags over all keypoints.
    std::optional<Pose> solvePose(const std::vector<cv::Point2f>& imagePoints,
                                  const std::vector<float>& confidence,
                                  double maxError,
                                  std::vector<bool>& inliers) const;
    Pose resolveSymmetry(const Pose& pose, const Eigen::Matrix3d& allocentric) const;
    bool poseOutsideFov(const Pose& pose, int64_t height, int64_t width) const;
};


#endif //LIGHTPOSEESTIMLIB_POSEESTIMATOR_H
