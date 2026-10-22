//
// Created by algoritmi on 10/22/26.
//

#include <torch/torch.h>
#include <ATen/Context.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include "PoseEstimator.h"

// Runs an exported pose estimator on images and prints one JSON object per image.
// --full-float32 turns off TF32 on CUDA, for results closest to the Python reference.
// --threads N limits LibTorch's CPU threads (set OMP_NUM_THREADS=N too: CPU packages run
// their kernels on the system OpenMP runtime).
int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    bool fullFloat32 = false;
    int threads = 0;
    while (!args.empty() && args.front().starts_with("--")) {
        if (args.front() == "--full-float32") {
            fullFloat32 = true;
        } else if (args.front() == "--threads" && args.size() > 1) {
            threads = std::stoi(args[1]);
            args.erase(args.begin());
        } else {
            break;
        }
        args.erase(args.begin());
    }
    if (args.size() < 2) {
        std::cerr << "Usage: main [--full-float32] [--threads N] <export_dir> <image>...\n";
        return 2;
    }
    if (fullFloat32) {
        at::globalContext().setAllowTF32CuDNN(false);
        at::globalContext().setAllowTF32CuBLAS(false);
    }
    if (threads > 0) {
        at::set_num_threads(threads);
        cv::setNumThreads(threads);
    }
    std::cerr << "torch " << TORCH_VERSION
              << ", CUDA available: " << torch::cuda::is_available() << "\n";

    PoseEstimator estimator(args[0]);
    std::cerr << "Loaded " << args[0] << " for " << estimator.getConfig().device << "\n";
    for (size_t i = 1; i < args.size(); ++i) {
        const cv::Mat image = cv::imread(args[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Cannot read " << args[i] << "\n";
            continue;
        }
        const auto start = std::chrono::steady_clock::now();
        const PoseEstimationResult result = estimator.estimate(image);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        nlohmann::json out = {
            {"image", args[i]},
            {"ms", ms},
            {"roi", {result.roi.cx, result.roi.cy, result.roi.w, result.roi.h}},
            {"rejected", result.rejected},
            {"inliers", result.inliers},
            {"keypoint_confidence", result.keypointConfidence},
            {"rotation_disagreement", result.rotationDisagreement},
            {"confidence", result.confidence},
        };
        nlohmann::json points = nlohmann::json::array();
        for (const Eigen::Vector2d& p : result.imagePoints) points.push_back({p.x(), p.y()});
        out["image_points"] = points;
        if (result.pose) {
            const Pose& pose = *result.pose;
            out["translation"] = {pose.position.x, pose.position.y, pose.position.z};
            out["quaternion"] = {pose.quaternion.qx, pose.quaternion.qy, pose.quaternion.qz, pose.quaternion.qw};
        }
        std::cout << out.dump() << std::endl;
    }
    return 0;
}
