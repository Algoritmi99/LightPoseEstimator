//
// Created by algoritmi on 10/22/26.
//

#include <torch/torch.h>
#include <ATen/Context.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include "PoseEstimator.h"

namespace {
nlohmann::json poseJson(const Pose& pose) {
    return {
        {"translation", {pose.position.x, pose.position.y, pose.position.z}},
        {"quaternion", {pose.quaternion.qx, pose.quaternion.qy, pose.quaternion.qz, pose.quaternion.qw}},
    };
}

nlohmann::json hypothesesJson(const std::vector<TurnHypothesis>& hypotheses) {
    nlohmann::json out = nlohmann::json::array();
    for (const TurnHypothesis& h : hypotheses) {
        nlohmann::json item = poseJson(h.pose);
        item["turn"] = h.turn;
        item["probability"] = h.probability;
        out.push_back(item);
    }
    return out;
}

nlohmann::json resultJson(const PoseEstimationResult& result, double candidateMinProbability) {
    nlohmann::json out = {
        {"roi", {result.roi.cx, result.roi.cy, result.roi.w, result.roi.h}},
        {"rejected", result.rejected},
        {"inliers", result.inliers},
        {"keypoint_confidence", result.keypointConfidence},
        {"rotation_disagreement", result.rotationDisagreement},
        {"confidence", result.confidence},
        {"turn", result.turn},
        {"turn_confidence", result.turnConfidence},
        {"ambiguous", result.ambiguous(candidateMinProbability)},
        {"hypotheses", hypothesesJson(result.hypotheses)},
    };
    nlohmann::json points = nlohmann::json::array();
    for (const Eigen::Vector2d& p : result.imagePoints) points.push_back({p.x(), p.y()});
    out["image_points"] = points;
    if (result.pose) out.update(poseJson(*result.pose));
    return out;
}

// camera_intrinsics.json as in the datasets: {"K": 3x3, "D": [...]}.
std::pair<Eigen::Matrix3d, std::vector<double>> readCamera(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open " + path);
    const nlohmann::json camera = nlohmann::json::parse(file);
    Eigen::Matrix3d k;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            k(r, c) = camera.at("K").at(r).at(c).get<double>();
    std::vector<double> d;
    for (const auto& v : camera.at("D").flatten()) d.push_back(v.get<double>());
    return {k, d};
}
}

// Runs an exported pose estimator on images and prints one JSON object per image.
//   --track             the images are one stream, in order: also print the tracked pose
//   --list FILE         read "image [timestamp]" lines from FILE instead of the arguments
//   --camera FILE       camera_intrinsics.json (K, D) to use instead of the exported camera
//   --bayer PATTERN     the images are raw sensor mosaics (BG, GB, RG or GR)
//   --full-float32      turn off TF32 on CUDA, for results closest to the Python reference
//   --threads N         limit LibTorch's CPU threads (set OMP_NUM_THREADS=N too: CPU packages
//                       run their kernels on the system OpenMP runtime)
int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    bool fullFloat32 = false, track = false;
    int threads = 0;
    std::string listFile, cameraFile;
    std::optional<std::string> bayer;
    while (!args.empty() && args.front().starts_with("--")) {
        const std::string option = args.front();
        const bool hasValue = args.size() > 1;
        if (option == "--full-float32") {
            fullFloat32 = true;
        } else if (option == "--track") {
            track = true;
        } else if (option == "--threads" && hasValue) {
            threads = std::stoi(args[1]);
            args.erase(args.begin());
        } else if (option == "--list" && hasValue) {
            listFile = args[1];
            args.erase(args.begin());
        } else if (option == "--camera" && hasValue) {
            cameraFile = args[1];
            args.erase(args.begin());
        } else if (option == "--bayer" && hasValue) {
            bayer = args[1];
            args.erase(args.begin());
        } else {
            break;
        }
        args.erase(args.begin());
    }
    std::vector<std::pair<std::string, std::optional<double>>> frames;
    for (size_t i = 1; i < args.size(); ++i) frames.emplace_back(args[i], std::nullopt);
    if (!listFile.empty()) {
        std::ifstream list(listFile);
        std::string line;
        while (std::getline(list, line)) {
            std::istringstream fields(line);
            std::string path;
            double timestamp;
            if (!(fields >> path)) continue;
            frames.emplace_back(path, fields >> timestamp ? std::optional(timestamp) : std::nullopt);
        }
    }
    if (args.empty() || frames.empty()) {
        std::cerr << "Usage: main [--track] [--list FILE] [--camera FILE] [--bayer PATTERN] [--full-float32] "
                     "[--threads N] <export_dir> [<image>...]\n";
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
    std::cerr << "torch " << TORCH_VERSION << ", CUDA available: " << torch::cuda::is_available() << "\n";

    PoseEstimator estimator(args[0]);
    if (!cameraFile.empty() || bayer) {
        auto [k, d] = cameraFile.empty()
            ? std::pair(estimator.getConfig().cameraIntrinsics, estimator.getConfig().distortionCoeffs)
            : readCamera(cameraFile);
        estimator.setCamera(k, d, bayer ? bayer : estimator.getConfig().bayerPattern);
    }
    const ExportConfig& config = estimator.getConfig();
    const double candidateMin = config.symmetric ? config.symmetric->candidateMinProbability : 0.2;
    std::cerr << "Loaded " << args[0] << " for " << config.device
              << (config.symmetric ? ", keypoints in symmetric sets" : "")
              << (config.bayerPattern ? ", raw " + *config.bayerPattern + " mosaics" : "") << "\n";

    for (const auto& [path, timestamp] : frames) {
        // A raw mosaic has one channel; anything else as BGR.
        const cv::Mat image = cv::imread(path, config.bayerPattern ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Cannot read " << path << "\n";
            continue;
        }
        const auto start = std::chrono::steady_clock::now();
        nlohmann::json out;
        if (track) {
            const TrackedPose tracked = estimator.track(image, timestamp);
            out = resultJson(*tracked.frame, candidateMin);
            nlohmann::json t = {
                {"probability", tracked.probability},
                {"resolved", tracked.resolved},
                {"measured", tracked.measured},
                {"candidates", hypothesesJson(tracked.candidates)},
                {"position", nullptr},
                {"attitude", nullptr},
            };
            if (tracked.pose) t.update(poseJson(*tracked.pose));
            if (const auto position = tracked.position()) t["position"] = {position->x(), position->y(), position->z()};
            if (const auto attitude = tracked.attitude()) t["attitude"] = {attitude->qx, attitude->qy, attitude->qz, attitude->qw};
            out["tracked"] = t;
        } else {
            out = resultJson(estimator.estimate(image), candidateMin);
        }
        out["ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        out["image"] = path;
        std::cout << out.dump() << std::endl;
    }
    return 0;
}
