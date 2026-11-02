#include "ExportConfig.h"
#include <Eigen/Geometry>
#include <algorithm>
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

nlohmann::json parseFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open " + path.string());
    return nlohmann::json::parse(file);
}

std::vector<Eigen::Vector3d> vectors3(const nlohmann::json& rows) {
    std::vector<Eigen::Vector3d> out;
    out.reserve(rows.size());
    for (const auto& p : rows)
        out.emplace_back(p.at(0).get<double>(), p.at(1).get<double>(), p.at(2).get<double>());
    return out;
}

TurnDistinctness loadDistinctness(const std::filesystem::path& path, size_t turns) {
    const nlohmann::json d = parseFile(path);
    TurnDistinctness out;
    out.points = vectors3(d.at("points"));
    out.normals = vectors3(d.at("normals"));
    for (const auto& row : d.at("differs")) {
        std::vector<uint8_t> differs;
        differs.reserve(row.size());
        for (const auto& v : row) differs.push_back(v.get<int>() != 0);
        out.differs.push_back(std::move(differs));
    }
    const auto range = d.at("distinct_range").get<std::vector<double>>();
    out.low = range.at(0);
    out.high = range.at(1);
    if (out.normals.size() != out.points.size() || out.differs.size() != turns ||
        std::any_of(out.differs.begin(), out.differs.end(), [&](const auto& row) { return row.size() != out.points.size(); }))
        throw std::runtime_error("Inconsistent sizes in " + path.string());
    return out;
}

SymmetricKeypoints loadSymmetric(const nlohmann::json& s, const std::filesystem::path& exportDir, size_t keypoints) {
    SymmetricKeypoints out;
    for (const auto& turn : s.at("turns")) out.turns.push_back(matrix3(turn));
    out.permutation = s.at("permutation").get<std::vector<std::vector<int>>>();
    out.compose = s.at("compose").get<std::vector<std::vector<int>>>();
    out.candidateMinProbability = s.at("candidate_min_probability").get<double>();
    const size_t turns = out.turns.size();
    auto valid = [&](const std::vector<std::vector<int>>& table, size_t width, size_t limit) {
        return table.size() == turns && std::all_of(table.begin(), table.end(), [&](const auto& row) {
            return row.size() == width && std::all_of(row.begin(), row.end(), [&](int v) { return v >= 0 && static_cast<size_t>(v) < limit; });
        });
    };
    if (turns == 0 || !valid(out.permutation, keypoints, keypoints) || !valid(out.compose, turns, turns))
        throw std::runtime_error("Inconsistent symmetric_keypoints in pose_estimator.json.");
    if (!s.at("distinctness").is_null())
        out.distinctness = loadDistinctness(exportDir / s.at("distinctness").get<std::string>(), turns);
    return out;
}
}

std::vector<double> TurnDistinctness::informativeness(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) const {
    // How much of the visible surface (by projected area: normals facing the camera) differs
    // under each turn, mapped from distinct_range onto [0, 1] by a smoothstep.
    std::vector<double> facing(points.size());
    double facingSum = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d q = rotation * points[i] + translation;
        facing[i] = std::max(0.0, (rotation * normals[i]).dot(-q / q.norm()));
        facingSum += facing[i];
    }
    std::vector<double> info(differs.size());
    for (size_t g = 0; g < differs.size(); ++g) {
        double share = 0.0;
        for (size_t i = 0; i < points.size(); ++i)
            if (differs[g][i]) share += facing[i];
        share /= std::max(facingSum, 1e-12);
        const double x = std::clamp((share - low) / (high - low), 0.0, 1.0);
        info[g] = x * x * (3.0 - 2.0 * x);
    }
    if (!info.empty()) info[0] = 1.0;
    return info;
}

ExportConfig ExportConfig::load(const std::filesystem::path& exportDir) {
    const nlohmann::json m = parseFile(exportDir / "pose_estimator.json");

    const int formatVersion = m.at("format_version").get<int>();
    if (formatVersion < 1 || formatVersion > supportedFormatVersion)
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
    config.keypointPositions = vectors3(keypoints.at("positions"));

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

    if (formatVersion >= 2) {
        if (!camera.at("bayer_pattern").is_null())
            config.bayerPattern = camera.at("bayer_pattern").get<std::string>();
        if (!m.at("symmetric_keypoints").is_null())
            config.symmetric = loadSymmetric(m.at("symmetric_keypoints"), exportDir, config.keypointPositions.size());
        const auto& continuity = m.at("continuity");
        config.continuity = {
            continuity.at("max_rotation_change_deg").get<double>(),
            continuity.at("max_translation_change").get<double>(),
        };
        const auto& t = m.at("tracking");
        config.tracking.temperature = t.at("temperature").get<double>();
        config.tracking.commitMargin = t.at("commit_margin").get<double>();
        config.tracking.beliefCap = t.at("belief_cap").get<double>();
        config.tracking.gateDeg = t.at("gate_deg").get<double>();
        config.tracking.maxGateDeg = t.at("max_gate_deg").get<double>();
        config.tracking.missingEvidence = t.at("missing_evidence").get<double>();
        config.tracking.maxMissed = t.at("max_missed").get<int>();
        config.tracking.frameInterval = optionalDouble(t.at("frame_interval"));
        config.tracking.resetAfter = optionalDouble(t.at("reset_after"));
        config.tracking.candidateMinProbability = t.at("candidate_min_probability").get<double>();
    }
    return config;
}
