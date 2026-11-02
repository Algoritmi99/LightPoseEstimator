#ifndef LIGHTPOSEESTIMLIB_TURNTRACKER_H
#define LIGHTPOSEESTIMLIB_TURNTRACKER_H
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>
#include "ExportConfig.h"
#include "PoseEstimationResult.h"


// The stream's view of one frame: the likeliest twin, and every twin not yet ruled out.
struct TrackedPose {
    std::optional<Pose> pose;  // the likeliest twin's pose; empty when there is none yet
    double probability = std::numeric_limits<double>::quiet_NaN();
    std::vector<TurnHypothesis> candidates;  // every twin still in play, likeliest first
    bool resolved = false;     // the twin is settled: pose can be used whole
    bool measured = true;      // false: this frame solved no pose for that twin, whose pose is its last one
    std::optional<PoseEstimationResult> frame;  // this frame's own estimate (from PoseEstimator::track)

    // The target's position (camera frame, metres) whenever there is a pose, settled or not: the
    // twins differ by a turn about the object's centre, so they share it. From this frame's own
    // estimate when it solved no pose for the reported twin.
    std::optional<Eigen::Vector3d> position() const;
    // The target's rotation (object to camera frame) once the twin is settled; empty before.
    std::optional<Quaternion> attitude() const;
};

// Resolving a near-symmetric object's turned twins over an image stream (tracking.TurnTracker in
// Python). Each twin is followed as its own identity: a frame's hypotheses continue the twins
// they are close to in rotation, and each twin collects the estimator's evidence for it. A twin
// the frame solved no pose for keeps its last pose and gets neutral evidence, so one frame where
// PnP misses the true twin can't flip the stream onto another. Neighbouring frames see nearly
// the same view, so their evidence is discounted, and no twin falls more than the belief cap
// behind the likeliest, so steady contrary evidence can still overturn a wrong lead.
class TurnTracker {
public:
    explicit TurnTracker(const TrackingSettings& settings = {});

    // Each frame's estimate, in order, including frames without a pose (they widen the gate).
    TrackedPose update(const PoseEstimationResult& result, std::optional<double> timestamp = std::nullopt);
    void reset();
    const TrackingSettings& getSettings() const { return settings; }

private:
    struct Twin {
        int64_t key;
        Pose pose;
        int turn;
        double belief;  // log, relative to the likeliest twin's (so <= 0)
        int missed;     // consecutive frames that solved no pose for it
    };

    TrackingSettings settings;
    std::vector<Twin> twins;
    std::optional<double> lastTime;
    int skipped = 0;    // frames without a pose since the last one with
    int64_t nextKey = 0;

    double evidence(double probability) const;
    double gate(std::optional<double> timestamp) const;
    TrackedPose tracked() const;
};


#endif //LIGHTPOSEESTIMLIB_TURNTRACKER_H
