#include "TurnTracker.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>
#include <numeric>

namespace {
constexpr double outOfGateCost = 1e6;
constexpr size_t maxExhaustive = 9;  // larger problems fall back to greedy matching

// Degrees between two poses' rotations.
double angleDeg(const Pose& a, const Pose& b) {
    const Eigen::Quaterniond qa(a.quaternion.qw, a.quaternion.qx, a.quaternion.qy, a.quaternion.qz);
    const Eigen::Quaterniond qb(b.quaternion.qw, b.quaternion.qx, b.quaternion.qy, b.quaternion.qz);
    return qa.normalized().angularDistance(qb.normalized()) * 180.0 / std::numbers::pi;
}

// For each twin, the hypothesis continuing it, or -1: the minimum-cost assignment of rotation
// angles, angles beyond the gate costing outOfGateCost (as SciPy's linear_sum_assignment in
// Python), keeping the pairs within the gate. Exhaustive: there are a handful of each.
std::vector<int> associate(const std::vector<Pose>& twins, const std::vector<TurnHypothesis>& hypotheses, double gate) {
    const size_t n = twins.size(), m = hypotheses.size();
    std::vector<int> match(n, -1);
    if (n == 0 || m == 0) return match;
    std::vector<std::vector<double>> angle(n, std::vector<double>(m));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            angle[i][j] = angleDeg(twins[i], hypotheses[j].pose);
    auto cost = [&](size_t i, size_t j) { return angle[i][j] <= gate ? angle[i][j] : outOfGateCost; };

    if (std::max(n, m) > maxExhaustive) {
        // Greedy, closest pairs first.
        std::vector<std::pair<size_t, size_t>> pairs;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                if (angle[i][j] <= gate) pairs.emplace_back(i, j);
        std::stable_sort(pairs.begin(), pairs.end(), [&](const auto& a, const auto& b) { return angle[a.first][a.second] < angle[b.first][b.second]; });
        std::vector<bool> used(m, false);
        for (const auto& [i, j] : pairs)
            if (match[i] < 0 && !used[j]) {
                match[i] = static_cast<int>(j);
                used[j] = true;
            }
        return match;
    }

    // Every twin gets a hypothesis when there are at least as many hypotheses, else every
    // hypothesis a twin; search those assignments for the lowest total cost.
    const bool byTwin = n <= m;
    const size_t a = byTwin ? n : m, b = byTwin ? m : n;
    auto pairCost = [&](size_t i, size_t j) { return byTwin ? cost(i, j) : cost(j, i); };
    std::vector<size_t> current(a), best(a);
    std::vector<bool> used(b, false);
    double bestTotal = std::numeric_limits<double>::infinity();
    std::function<void(size_t, double)> search = [&](size_t i, double total) {
        if (total >= bestTotal) return;
        if (i == a) {
            bestTotal = total;
            best = current;
            return;
        }
        for (size_t j = 0; j < b; ++j) {
            if (used[j]) continue;
            used[j] = true;
            current[i] = j;
            search(i + 1, total + pairCost(i, j));
            used[j] = false;
        }
    };
    search(0, 0.0);
    for (size_t i = 0; i < a; ++i) {
        const size_t twin = byTwin ? i : best[i], hypothesis = byTwin ? best[i] : i;
        if (angle[twin][hypothesis] <= gate) match[twin] = static_cast<int>(hypothesis);
    }
    return match;
}
}

std::optional<Eigen::Vector3d> TrackedPose::position() const {
    if (!pose) return std::nullopt;
    const Pose& source = (!measured && frame && frame->pose) ? *frame->pose : *pose;
    return source.getTvec();
}

std::optional<Quaternion> TrackedPose::attitude() const {
    if (!resolved || !pose) return std::nullopt;
    return pose->quaternion;
}

TurnTracker::TurnTracker(const TrackingSettings& settings) : settings(settings) {}

void TurnTracker::reset() {
    twins.clear();
    lastTime.reset();
    skipped = 0;
}

double TurnTracker::evidence(double probability) const {
    return std::log(std::max(probability, 1e-9)) / settings.temperature;
}

double TurnTracker::gate(std::optional<double> timestamp) const {
    // How far a twin may have turned since the last pose: wider after frames without one (like
    // a random walk), up to maxGateDeg.
    double frames = skipped + 1;
    if (timestamp && lastTime && settings.frameInterval && *settings.frameInterval > 0)
        frames = std::max((*timestamp - *lastTime) / *settings.frameInterval, 1.0);
    return std::min(settings.gateDeg * std::sqrt(frames), settings.maxGateDeg);
}

TrackedPose TurnTracker::update(const PoseEstimationResult& result, std::optional<double> timestamp) {
    if (!result.pose || result.hypotheses.empty()) {
        ++skipped;
        return {};
    }
    if (settings.resetAfter && timestamp && lastTime && *timestamp - *lastTime > *settings.resetAfter) reset();
    const std::vector<TurnHypothesis>& hypotheses = result.hypotheses;
    if (twins.empty()) {
        for (const TurnHypothesis& h : hypotheses)
            twins.push_back({nextKey++, h.pose, h.turn, evidence(h.probability), 0});
    } else {
        const double gateDeg = gate(timestamp);
        std::vector<Pose> poses;
        for (const Twin& twin : twins) poses.push_back(twin.pose);
        const std::vector<int> match = associate(poses, hypotheses, gateDeg);
        std::vector<bool> continued(hypotheses.size(), false);
        for (size_t i = 0; i < twins.size(); ++i) {
            Twin& twin = twins[i];
            if (match[i] < 0) {
                ++twin.missed;
                twin.belief += evidence(settings.missingEvidence);
            } else {
                const TurnHypothesis& h = hypotheses[match[i]];
                twin.pose = h.pose;
                twin.turn = h.turn;
                twin.missed = 0;
                twin.belief += evidence(h.probability);
                continued[match[i]] = true;
            }
        }
        // A hypothesis far from every twin so far starts a new one, as unlikely as allowed.
        for (size_t j = 0; j < hypotheses.size(); ++j) {
            if (continued[j]) continue;
            const bool far = std::all_of(twins.begin(), twins.end(), [&](const Twin& twin) { return angleDeg(twin.pose, hypotheses[j].pose) > gateDeg; });
            if (far) twins.push_back({nextKey++, hypotheses[j].pose, hypotheses[j].turn, -std::numeric_limits<double>::infinity(), 0});
        }
    }
    double top = -std::numeric_limits<double>::infinity();
    for (const Twin& twin : twins) top = std::max(top, twin.belief);
    for (Twin& twin : twins) twin.belief = std::max(twin.belief - top, -settings.beliefCap);
    std::erase_if(twins, [&](const Twin& twin) { return twin.belief != 0.0 && twin.missed > settings.maxMissed; });
    lastTime = timestamp;
    skipped = 0;
    return tracked();
}

TrackedPose TurnTracker::tracked() const {
    std::vector<size_t> order(twins.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return twins[a].belief > twins[b].belief; });
    const double top = twins[order[0]].belief;
    double sum = 0.0;
    for (const Twin& twin : twins) sum += std::exp(twin.belief - top);
    const double logTotal = top + std::log(sum);

    TrackedPose out;
    std::vector<TurnHypothesis> hypotheses;
    for (const size_t k : order)
        hypotheses.push_back({twins[k].turn, std::exp(twins[k].belief - logTotal), twins[k].pose});
    for (const TurnHypothesis& h : hypotheses)
        if (h.probability >= settings.candidateMinProbability) out.candidates.push_back(h);
    if (out.candidates.empty()) out.candidates.push_back(hypotheses.front());
    // A lone twin is not resolved: nothing has been ruled out.
    const double margin = order.size() > 1 ? twins[order[0]].belief - twins[order[1]].belief : 0.0;
    const Twin& best = twins[order[0]];
    out.pose = best.pose;
    out.probability = hypotheses.front().probability;
    out.resolved = margin >= settings.commitMargin;
    out.measured = best.missed == 0;
    return out;
}
