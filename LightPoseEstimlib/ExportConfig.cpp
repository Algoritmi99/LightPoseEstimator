#include "ExportConfig.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
std::optional<double> optionalDouble(const nlohmann::json& value) {
    if (value.is_null()) return std::nullopt;
    return value.get<double>();
}

Eigen::Matrix3d matrix3(const nlohmann::json& rows) {
    Eigen::Matrix3d m;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            m(r, c) = rows.at(r).at(c).get<double>();
    return m;
}
}

ExportConfig ExportConfig::load(const std::filesystem::path& exportDir) {
    const std::filesystem::path manifestPath = exportDir / "pose_estimator.json";
    std::ifstream file(manifestPath);
    if (!file) throw std::runtime_error("Cannot open " + manifestPath.string());
    const nlohmann::json m = nlohmann::json::parse(file);

    const int formatVersion = m.at("format_version").get<int>();
    if (formatVersion != supportedFormatVersion)
        throw std::runtime_error("Unsupported export format version " + std::to_string(formatVersion));

    ExportConfig config;
    config.torchVersion = m.at("torch_version").get<std::string>();
    config.device = m.at("device").get<std::string>();
    const auto& networks = m.at("networks");
    config.roiDetectorPackage = exportDir / networks.at("roi_detector").at("package").get<std::string>();
    config.keypointNetPackage = exportDir / networks.at("keypoint_net").at("package").get<std::string>();
    config.roiInternalInputSize = networks.at("roi_detector").at("internal_input_size").get<std::vector<int64_t>>();

    const auto& keypoints = m.at("keypoints");
    config.keypointIds = keypoints.at("ids").get<std::vector<int>>();
    for (const auto& p : keypoints.at("positions"))
        config.keypointPositions.emplace_back(p.at(0).get<double>(), p.at(1).get<double>(), p.at(2).get<double>());

    const auto& camera = m.at("camera");
    config.cameraIntrinsics = matrix3(camera.at("intrinsics"));
    if (!camera.at("distortion_coeffs").is_null())
        config.distortionCoeffs = camera.at("distortion_coeffs").get<std::vector<double>>();
    config.maxValidRadius = optionalDouble(camera.at("max_valid_radius"));

    config.cropSize = m.at("crop_size").get<int>();
    config.heatmapStride = m.at("heatmap_stride").get<int>();
    config.peakThreshold = m.at("peak_threshold").get<double>();

    const auto& pnp = m.at("pnp");
    config.pnp = {
        pnp.at("min_inliers").get<int>(),
        pnp.at("reprojection_error_cells").get<double>(),
        pnp.at("min_reprojection_error_px").get<double>(),
        pnp.at("ransac_iterations").get<int>(),
        pnp.at("ransac_confidence").get<double>(),
    };

    config.resolveSymmetry = m.at("resolve_symmetry").get<bool>();
    for (const auto& turn : m.at("symmetry_turns"))
        config.symmetryTurns.push_back(matrix3(turn));
    config.maxRotationDisagreementDeg = optionalDouble(m.at("max_rotation_disagreement_deg"));
    config.minKeypointConfidence = optionalDouble(m.at("min_keypoint_confidence"));
    return config;
}
