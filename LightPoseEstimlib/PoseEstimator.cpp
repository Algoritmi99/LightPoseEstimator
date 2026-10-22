//
// Created by algoritmi on 10/22/26.
//

#include "PoseEstimator.h"
#include <torch/torch.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

using torch::indexing::Slice;
namespace F = torch::nn::functional;

namespace {
constexpr double minDepth = 1e-6;

ExportConfig loadChecked(const std::filesystem::path& exportDir) {
    ExportConfig config = ExportConfig::load(exportDir);
    // AOTInductor packages are compiled code for one PyTorch version and device type.
    const std::string built = config.torchVersion.substr(0, config.torchVersion.find('+'));
    const std::string running = std::to_string(TORCH_VERSION_MAJOR) + "." +
                                std::to_string(TORCH_VERSION_MINOR) + "." +
                                std::to_string(TORCH_VERSION_PATCH);
    if (built != running)
        throw std::runtime_error("The export was built with PyTorch " + config.torchVersion +
                                 " but this LibTorch is " + running + "; export again with " + running + ".");
    if (config.device == "cuda" && !torch::cuda::is_available())
        throw std::runtime_error("The export is for CUDA but no CUDA device is available; export for the CPU.");
    return config;
}

// Largest undistorted radius at which the radial distortion model is still monotonic; past it,
// projection folds points back into the image (max_valid_radius in Python).
double foldRadius(const std::vector<double>& coeffs) {
    if (coeffs.empty()) return std::numeric_limits<double>::infinity();
    double k[8] = {};
    std::copy_n(coeffs.begin(), std::min<size_t>(coeffs.size(), 8), k);
    constexpr int steps = 100001;
    constexpr double maxRadius = 10.0;
    double previous = 0.0;
    for (int i = 1; i < steps; ++i) {
        const double r = maxRadius * i / (steps - 1), r2 = r * r;
        const double distorted = r * (1 + k[0] * r2 + k[1] * r2 * r2 + k[4] * r2 * r2 * r2) /
                                 (1 + k[5] * r2 + k[6] * r2 * r2 + k[7] * r2 * r2 * r2);
        if (distorted - previous <= 0) return maxRadius * (i - 1) / (steps - 1);
        previous = distorted;
    }
    return std::numeric_limits<double>::infinity();
}

cv::Mat toCvFloat(const Eigen::Matrix3d& m) {
    cv::Mat out(3, 3, CV_32F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out.at<float>(r, c) = static_cast<float>(m(r, c));
    return out;
}

cv::Mat toDistCoeffs(const std::vector<double>& coeffs) {
    if (coeffs.empty()) return cv::Mat::zeros(5, 1, CV_32F);
    cv::Mat out(static_cast<int>(coeffs.size()), 1, CV_32F);
    for (size_t i = 0; i < coeffs.size(); ++i) out.at<float>(static_cast<int>(i)) = static_cast<float>(coeffs[i]);
    return out;
}

cv::Mat toCvVector(const Eigen::Vector3d& v) {
    return (cv::Mat_<double>(3, 1) << v.x(), v.y(), v.z());
}

Eigen::Matrix3d rodrigues(const Eigen::Vector3d& rvec) {
    cv::Mat rotation;
    cv::Rodrigues(toCvVector(rvec), rotation);
    Eigen::Matrix3d out;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out(r, c) = rotation.at<double>(r, c);
    return out;
}

// The Python Pose hands out its translation as float32, so the steps using it see it rounded.
Eigen::Vector3d float32Tvec(const Pose& pose) {
    return pose.getTvec().cast<float>().cast<double>();
}

// Smallest rotation turning the optical axis (+z) onto the ray to t.
Eigen::Matrix3d rayRotation(const Eigen::Vector3d& t) {
    const Eigen::Vector3d ray = t / t.norm();
    const Eigen::Vector3d axis = Eigen::Vector3d::UnitZ().cross(ray);
    const double sinAngle = axis.norm();
    if (sinAngle < 1e-12) return Eigen::Matrix3d::Identity();
    return rodrigues(axis / sinAngle * std::atan2(sinAngle, ray.z()));
}

double angleDeg(const Eigen::Matrix3d& relative) {
    return std::acos(std::clamp((relative.trace() - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / std::numbers::pi;
}

// heatmap_peaks in Python, op for op: peak cell and value per channel, refined below one cell
// by fitting a parabola to the log heatmap around the maximum.
std::pair<at::Tensor, at::Tensor> heatmapPeaks(const at::Tensor& heatmaps, double threshold) {
    const int64_t batch = heatmaps.size(0), channels = heatmaps.size(1);
    const int64_t height = heatmaps.size(2), width = heatmaps.size(3);
    const at::Tensor logFlat = heatmaps.clamp_min(1e-6).log().reshape({batch, channels, -1});
    auto [confidence, index] = heatmaps.reshape({batch, channels, -1}).max(-1);
    const at::Tensor y = index.div(width, "floor");
    const at::Tensor x = index.remainder(width);

    auto logValue = [&](int64_t dx, int64_t dy) {
        const at::Tensor xn = (x + dx).clamp(0, width - 1);
        const at::Tensor yn = (y + dy).clamp(0, height - 1);
        return logFlat.gather(-1, (yn * width + xn).unsqueeze(-1)).squeeze(-1);
    };
    const at::Tensor center = logValue(0, 0);
    auto refine = [&](const at::Tensor& minus, const at::Tensor& plus, const at::Tensor& interior) {
        const at::Tensor curvature = plus - 2.0 * center + minus;
        at::Tensor step = 0.5 * (minus - plus) / curvature;
        step = torch::where(interior & (curvature < 0), step, torch::zeros_like(step));
        return step.clamp(-0.5, 0.5);
    };
    const at::Tensor dx = refine(logValue(-1, 0), logValue(1, 0), (x > 0) & (x < width - 1));
    const at::Tensor dy = refine(logValue(0, -1), logValue(0, 1), (y > 0) & (y < height - 1));
    const at::Tensor coordinates = torch::stack({x.to(torch::kFloat32) + dx, y.to(torch::kFloat32) + dy}, -1);
    confidence = confidence * (confidence >= threshold).to(torch::kFloat32);
    return {coordinates, confidence};
}

// rotation_from_6d in Python: first two matrix columns, made orthonormal by Gram-Schmidt.
at::Tensor rotationFrom6d(const at::Tensor& x) {
    const at::Tensor a = x.index({Slice(), Slice(0, 3)});
    const at::Tensor b = x.index({Slice(), Slice(3, 6)});
    const at::Tensor first = F::normalize(a, F::NormalizeFuncOptions().dim(1));
    const at::Tensor second = F::normalize(b - (first * b).sum(1, true) * first, F::NormalizeFuncOptions().dim(1));
    return torch::stack({first, second, torch::linalg_cross(first, second)}, 2);
}
}

PoseEstimator::PoseEstimator(const std::filesystem::path& exportDir)
    : config(loadChecked(exportDir)),
      device(config.device == "cuda" ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU)),
      roiDetector(config.roiDetectorPackage.string()),
      keyPointNet(config.keypointNetPackage.string()),
      pixelToFloat(torch::arange(256, torch::kUInt8).to(torch::kFloat32).div(255).to(device)),
      maxValidRadius(std::numeric_limits<double>::infinity()) {
    for (const Eigen::Vector3d& p : config.keypointPositions)
        objectPoints.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()));
    setCamera(config.cameraIntrinsics, config.distortionCoeffs);
}

void PoseEstimator::setCamera(const Eigen::Matrix3d& cameraIntrinsics, const std::vector<double>& distortionCoeffs) {
    config.cameraIntrinsics = cameraIntrinsics;
    config.distortionCoeffs = distortionCoeffs;
    cameraMatrix = toCvFloat(cameraIntrinsics);
    distCoeffs = toDistCoeffs(distortionCoeffs);
    maxValidRadius = foldRadius(distortionCoeffs);
    config.maxValidRadius = std::isfinite(maxValidRadius) ? std::optional(maxValidRadius) : std::nullopt;
}

PoseEstimationResult PoseEstimator::estimate(const cv::Mat& image, bool isBgr) {
    if (image.empty()) throw std::invalid_argument("Empty image.");
    if (image.depth() != CV_8U) throw std::invalid_argument("Expected an 8-bit image.");
    cv::Mat rgb;
    switch (image.channels()) {
        case 1: cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB); break;
        case 3: if (isBgr) cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB); else rgb = image; break;
        case 4: cv::cvtColor(image, rgb, isBgr ? cv::COLOR_BGRA2RGB : cv::COLOR_RGBA2RGB); break;
        default: throw std::invalid_argument("Expected an image with 1, 3 or 4 channels.");
    }
    if (!rgb.isContinuous()) rgb = rgb.clone();
    // As torchvision's ToTensor: uint8 HWC -> float32 CHW / 255. On the GPU the values are looked
    // up instead, so only the 8-bit image is copied and the result matches the CPU division.
    const at::Tensor bytes = torch::from_blob(rgb.data, {rgb.rows, rgb.cols, 3}, torch::kUInt8);
    if (device.is_cpu())
        return estimate(bytes.permute({2, 0, 1}).to(torch::kFloat32).div(255));
    const at::Tensor tensor = pixelToFloat.index_select(0, bytes.to(device).flatten().to(torch::kInt32))
                                  .view({rgb.rows, rgb.cols, 3}).permute({2, 0, 1});
    return estimate(tensor);
}

PoseEstimationResult PoseEstimator::estimate(const at::Tensor& input) {
    torch::NoGradGuard noGrad;
    const at::Tensor image = (input.dim() == 3 ? input.unsqueeze(0) : input).to(device, torch::kFloat32).contiguous();
    TORCH_CHECK(image.dim() == 4 && image.size(0) == 1 && image.size(1) == 3,
                "Expected an image tensor of shape [3, H, W] or [1, 3, H, W].");
    const int64_t height = image.size(2), width = image.size(3);
    const size_t numKeypoints = objectPoints.size();

    PoseEstimationResult result;
    result.imagePoints.assign(numKeypoints, Eigen::Vector2d::Zero());
    result.confidence.assign(numKeypoints, 0.0);

    // 1. ROI, scaled to pixels in float32 as the Python estimator does.
    const at::Tensor roi = roiDetector.run({image})[0].to(torch::kCPU, torch::kFloat32).contiguous();
    const float* r = roi.data_ptr<float>();
    const auto w = static_cast<float>(width), h = static_cast<float>(height);
    result.roi = {r[0] * w, r[1] * h, r[2] * w, r[3] * h};
    if (roiOutsideFrame(result.roi, height, width)) {
        result.rejected = "no target in view";
        return result;
    }

    // 2-4. Crop and keypoint network.
    const Box box = cropBox(result.roi, height, width);
    const int64_t size = config.cropSize;
    const at::Tensor crop = F::interpolate(
        image.index({Slice(), Slice(), Slice(box.y1, box.y2), Slice(box.x1, box.x2)}),
        F::InterpolateFuncOptions().size(std::vector<int64_t>{size, size}).mode(torch::kBilinear).align_corners(false)
    ).contiguous();
    const std::vector<at::Tensor> outputs = keyPointNet.run({crop});

    // 5. Peaks and the coarse rotation, with the same tensor operations as in Python.
    auto [cells, peaks] = heatmapPeaks(outputs[0].sigmoid(), config.peakThreshold);
    cells = cells[0].to(torch::kCPU).contiguous();
    peaks = peaks[0].to(torch::kCPU).contiguous();
    const at::Tensor coarseRotation = rotationFrom6d(outputs[1])[0].to(torch::kCPU, torch::kDouble).contiguous();
    const auto rotationAccessor = coarseRotation.accessor<double, 2>();
    Eigen::Matrix3d allocentric;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            allocentric(i, j) = rotationAccessor[i][j];

    // 6. Back to full-image pixels, as crop_to_image in Python (double, then float32 for PnP).
    const float* cell = cells.data_ptr<float>();
    const float* peak = peaks.data_ptr<float>();
    const double scaleX = static_cast<double>(size) / static_cast<double>(box.x2 - box.x1) / config.heatmapStride;
    const double scaleY = static_cast<double>(size) / static_cast<double>(box.y2 - box.y1) / config.heatmapStride;
    std::vector<cv::Point2f> imagePoints(numKeypoints);
    std::vector<float> confidence(numKeypoints);
    for (size_t k = 0; k < numKeypoints; ++k) {
        imagePoints[k] = {
            static_cast<float>((static_cast<double>(cell[2 * k]) + 0.5) / scaleX + static_cast<double>(box.x1) - 0.5),
            static_cast<float>((static_cast<double>(cell[2 * k + 1]) + 0.5) / scaleY + static_cast<double>(box.y1) - 0.5),
        };
        confidence[k] = peak[k];
        result.imagePoints[k] = {imagePoints[k].x, imagePoints[k].y};
        result.confidence[k] = peak[k];
    }

    // 7. PnP; keypoints reprojecting more than this many heatmap cells away are outliers.
    const double cellPx = static_cast<double>(config.heatmapStride) *
                          static_cast<double>(std::max(box.x2 - box.x1, box.y2 - box.y1)) / static_cast<double>(size);
    const double maxError = std::max(config.pnp.minReprojectionErrorPx, config.pnp.reprojectionErrorCells * cellPx);
    std::vector<bool> inliers;
    std::optional<Pose> pose = solvePose(imagePoints, confidence, maxError, inliers);
    if (!pose) {
        result.rejected = "too few keypoints";
        return result;
    }

    // 8-9. Symmetry and the evidence behind the pose.
    if (config.resolveSymmetry) pose = resolveSymmetry(*pose, allocentric);
    double confidenceSum = 0.0;
    for (size_t k = 0; k < numKeypoints; ++k) {
        if (!inliers[k]) continue;
        ++result.inliers;
        confidenceSum += confidence[k];
    }
    // numpy's mean of the float32 confidences is float32.
    result.keypointConfidence = static_cast<float>(confidenceSum / result.inliers);
    const Eigen::Matrix3d coarse = rayRotation(float32Tvec(*pose)) * allocentric;
    result.rotationDisagreement = angleDeg(pose->getRotationMatrix().transpose() * coarse);

    // 10. Checks.
    if (poseOutsideFov(*pose, height, width))
        result.rejected = "outside field of view";
    else if (config.maxRotationDisagreementDeg && result.rotationDisagreement > *config.maxRotationDisagreementDeg)
        result.rejected = "rotation disagreement";
    else if (config.minKeypointConfidence && result.keypointConfidence < *config.minKeypointConfidence)
        result.rejected = "low keypoint confidence";
    else
        result.pose = pose;
    return result;
}

bool PoseEstimator::roiOutsideFrame(const Roi& roi, int64_t height, int64_t width) {
    if (roi.w <= 1.0 || roi.h <= 1.0) return true;
    const double x1 = std::max(roi.cx - roi.w / 2, 0.0);
    const double y1 = std::max(roi.cy - roi.h / 2, 0.0);
    const double x2 = std::min(roi.cx + roi.w / 2, static_cast<double>(width));
    const double y2 = std::min(roi.cy + roi.h / 2, static_cast<double>(height));
    return x2 - x1 <= 1.0 || y2 - y1 <= 1.0;
}

PoseEstimator::Box PoseEstimator::cropBox(const Roi& roi, int64_t height, int64_t width) {
    Box box{
        std::max<int64_t>(0, static_cast<int64_t>(std::floor(roi.cx - roi.w / 2))),
        std::max<int64_t>(0, static_cast<int64_t>(std::floor(roi.cy - roi.h / 2))),
        std::min<int64_t>(width, static_cast<int64_t>(std::ceil(roi.cx + roi.w / 2))),
        std::min<int64_t>(height, static_cast<int64_t>(std::ceil(roi.cy + roi.h / 2))),
    };
    if (box.x2 <= box.x1) {
        box.x1 = std::min(box.x1, width - 1);
        box.x2 = box.x1 + 1;
    }
    if (box.y2 <= box.y1) {
        box.y1 = std::min(box.y1, height - 1);
        box.y2 = box.y1 + 1;
    }
    return box;
}

std::optional<Pose> PoseEstimator::solvePose(const std::vector<cv::Point2f>& imagePoints,
                                             const std::vector<float>& confidence,
                                             double maxError,
                                             std::vector<bool>& inliers) const {
    const auto threshold = static_cast<float>(config.peakThreshold);
    std::vector<size_t> valid;
    for (size_t k = 0; k < confidence.size(); ++k)
        if (confidence[k] >= threshold) valid.push_back(k);
    if (static_cast<int>(valid.size()) < config.pnp.minInliers) return std::nullopt;

    std::vector<cv::Point3f> object;
    std::vector<cv::Point2f> image;
    for (const size_t k : valid) {
        object.push_back(objectPoints[k]);
        image.push_back(imagePoints[k]);
    }
    cv::Mat rvec, tvec;
    if (!cv::solvePnPRansac(object, image, cameraMatrix, distCoeffs, rvec, tvec, false,
                            config.pnp.ransacIterations, static_cast<float>(maxError),
                            config.pnp.ransacConfidence, cv::noArray(), cv::SOLVEPNP_SQPNP))
        return std::nullopt;

    // With 4 or 5 points OpenCV solves directly and reports every point as an inlier, so count
    // inliers from the reprojection error here.
    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(object, rvec, tvec, cameraMatrix, distCoeffs, reprojected);
    std::vector<cv::Point3f> inlierObject;
    std::vector<cv::Point2f> inlierImage;
    inliers.assign(confidence.size(), false);
    for (size_t i = 0; i < object.size(); ++i) {
        const float dx = reprojected[i].x - image[i].x, dy = reprojected[i].y - image[i].y;
        if (std::sqrt(dx * dx + dy * dy) > static_cast<float>(maxError)) continue;
        inlierObject.push_back(object[i]);
        inlierImage.push_back(image[i]);
        inliers[valid[i]] = true;
    }
    if (static_cast<int>(inlierObject.size()) < config.pnp.minInliers) return std::nullopt;
    cv::solvePnPRefineLM(inlierObject, inlierImage, cameraMatrix, distCoeffs, rvec, tvec);

    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    Eigen::Matrix3d r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r(i, j) = rotation.at<double>(i, j);
    return Pose::fromRotationMatrix(r, {tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)});
}

Pose PoseEstimator::resolveSymmetry(const Pose& pose, const Eigen::Matrix3d& allocentric) const {
    // Keypoints on a near-symmetric object can land on the wrong side, turning the PnP pose
    // 180 degrees about an object axis; keep the turn closest to the network's coarse rotation.
    const Eigen::Vector3d t = float32Tvec(pose);
    const Eigen::Matrix3d coarse = rayRotation(t) * allocentric;
    const Eigen::Matrix3d rotation = pose.getRotationMatrix();
    size_t best = 0;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < config.symmetryTurns.size(); ++i) {
        const double score = ((rotation * config.symmetryTurns[i]).transpose() * coarse).trace();
        if (score > bestScore) {
            best = i;
            bestScore = score;
        }
    }
    if (best == 0) return pose;
    return Pose::fromRotationMatrix(rotation * config.symmetryTurns[best], t);
}

bool PoseEstimator::poseOutsideFov(const Pose& pose, int64_t height, int64_t width) const {
    const Eigen::Vector3d rvec = pose.getRvec();
    const Eigen::Matrix3d rotation = rodrigues(rvec);
    const Eigen::Vector3d t = float32Tvec(pose);
    std::vector<cv::Point3f> projectable;
    for (const cv::Point3f& p : objectPoints) {
        const Eigen::Vector3d camera = rotation * Eigen::Vector3d(p.x, p.y, p.z) + t;
        if (camera.z() <= minDepth || std::hypot(camera.x(), camera.y()) / camera.z() > maxValidRadius) continue;
        projectable.push_back(p);
    }
    if (projectable.empty()) return true;
    std::vector<cv::Point2f> projected;
    cv::projectPoints(projectable, toCvVector(rvec), toCvVector(t), cameraMatrix, distCoeffs, projected);
    for (const cv::Point2f& q : projected)
        if (q.x >= 0 && q.x < static_cast<float>(width) && q.y >= 0 && q.y < static_cast<float>(height)) return false;
    return true;
}
