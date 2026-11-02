"""
Resolving a near-symmetric object's turned twins over an image stream.

From one image, views that show none of the parts that differ between twins can't tell them
apart (for the Astrobee: its front and back, seen edge-on), and the estimator then reports each
twin as a candidate. Across a stream the true pose moves a little from frame to frame while a
twin is 180 degrees away, so each twin can be followed as its own identity: a frame's
hypotheses (``PoseEstimationResult.hypotheses``) continue the twins they are close to, and each
twin collects the estimator's evidence for it. A frame that does show the differing parts
settles the turn for the frames around it.

A twin the frame solved no pose for keeps its last pose and gets neutral evidence, rather than
being handed over to a hypothesis 180 degrees away: one frame where PnP misses the true twin
can't flip the stream onto the wrong one. Consecutive frames see nearly the same view, so their
evidence is discounted (``temperature``), and no twin falls more than ``belief_cap`` behind the
likeliest, so steady contrary evidence can still overturn a wrong lead.

``TurnTracker`` filters online (past frames only), as on the payload; ``smooth_turns`` also uses
later frames, to evaluate recordings.
"""
import dataclasses
import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .pose import Pose
from .pose_estimator import CANDIDATE_MIN_PROBABILITY, PoseEstimationResult, TurnHypothesis

DEFAULT_TEMPERATURE = 2.0       # a frame's evidence counts 1/temperature: neighbours see nearly the same view
DEFAULT_COMMIT_MARGIN = 4.0     # log-odds lead (about 55:1) that resolves the turn
DEFAULT_BELIEF_CAP = 20.0       # no twin's log belief falls further behind the likeliest's
DEFAULT_GATE_DEG = 45.0         # rotation change between frames that still continues a twin
MAX_GATE_DEG = 90.0             # twins are 180 degrees apart, so never more than this
MISSING_EVIDENCE = 0.25         # for a twin the frame solved no pose for: as if one of four turns
MAX_MISSED = 25                 # frames a twin other than the likeliest is kept without a pose


@dataclass
class TrackedPose:
    """The stream's view of one frame: the likeliest twin, and every twin not yet ruled out."""
    pose: Pose | None
    probability: float = float("nan")
    candidates: list[TurnHypothesis] = field(default_factory=list)
    resolved: bool = False
    # False when this frame solved no pose for the likeliest twin: ``pose`` is then its last one.
    measured: bool = True


@dataclass
class _Twin:
    """One twin followed through the stream."""
    key: int
    pose: Pose
    turn: int
    belief: float = 0.0         # log, relative to the likeliest twin's (so <= 0)
    missed: int = 0             # consecutive frames that solved no pose for it


def _rotation(pose: Pose) -> Rotation:
    q = pose.quaternion
    return Rotation.from_quat([q.qx, q.qy, q.qz, q.qw])


def _angle(a: Pose, b: Pose) -> float:
    return float(np.degrees((_rotation(a) * _rotation(b).inv()).magnitude()))


def _hypotheses(result: PoseEstimationResult | None) -> list[TurnHypothesis]:
    if result is None or result.pose is None:
        return []
    return result.hypotheses


def _associate(twins: list[_Twin], hypotheses: list[TurnHypothesis], gate_deg: float) -> list[int | None]:
    """For each twin, the hypothesis that continues it (closest rotations, within the gate), or None."""
    if not twins or not hypotheses:
        return [None] * len(twins)
    angles = np.array([[_angle(t.pose, h.pose) for h in hypotheses] for t in twins])
    rows, cols = linear_sum_assignment(np.where(angles <= gate_deg, angles, 1e6))
    match: list[int | None] = [None] * len(twins)
    for r, c in zip(rows, cols):
        if angles[r, c] <= gate_deg:
            match[r] = int(c)
    return match


def _tracked(twins: list[_Twin], log_belief: np.ndarray, commit_margin: float) -> TrackedPose:
    order = np.argsort(-log_belief)
    probability = np.exp(log_belief - np.logaddexp.reduce(log_belief))
    hypotheses = [TurnHypothesis(twins[k].turn, float(probability[k]), twins[k].pose) for k in order]
    # A lone twin is not resolved: nothing has been ruled out.
    margin = log_belief[order[0]] - log_belief[order[1]] if len(order) > 1 else 0.0
    candidates = [h for h in hypotheses if h.probability >= CANDIDATE_MIN_PROBABILITY] or hypotheses[:1]
    best = twins[order[0]]
    return TrackedPose(best.pose, hypotheses[0].probability, candidates, bool(margin >= commit_margin), best.missed == 0)


class TurnTracker:
    """
    Online: feed each frame's estimate (``update``) in order, including frames without a pose;
    the result says which twin the stream so far supports and whether it is ``resolved``. Until
    it is, every twin still in play is a candidate. A twin may turn further the more frames
    passed without a pose (counted from the calls, or from time stamps with ``frame_interval``),
    and after ``reset_after`` seconds without a pose the tracker starts over.
    """

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        commit_margin: float = DEFAULT_COMMIT_MARGIN,
        belief_cap: float = DEFAULT_BELIEF_CAP,
        gate_deg: float = DEFAULT_GATE_DEG,
        frame_interval: float | None = None,
        reset_after: float | None = None,
    ):
        self.temperature = temperature
        self.commit_margin = commit_margin
        self.belief_cap = belief_cap
        self.gate_deg = gate_deg
        self.frame_interval = frame_interval
        self.reset_after = reset_after
        self._keys = itertools.count()
        self.reset()

    def reset(self) -> None:
        self._twins: list[_Twin] = []
        self._time: float | None = None
        self._skipped = 0                   # frames without a pose since the last one with

    def update(self, result: PoseEstimationResult | None, timestamp: float | None = None) -> TrackedPose:
        if self._observe(result, timestamp) is None:
            return TrackedPose(None)
        return _tracked(self._twins, np.array([t.belief for t in self._twins]), self.commit_margin)

    def _evidence(self, probability: float) -> float:
        return float(np.log(max(probability, 1e-9))) / self.temperature

    def _gate(self, timestamp: float | None) -> float:
        """How far a twin may have turned since the last pose: wider after frames without one
        (like a random walk), up to MAX_GATE_DEG."""
        frames = self._skipped + 1
        if timestamp is not None and self._time is not None and self.frame_interval:
            frames = max((timestamp - self._time) / self.frame_interval, 1.0)
        return min(self.gate_deg * np.sqrt(frames), MAX_GATE_DEG)

    def _observe(self, result: PoseEstimationResult | None, timestamp: float | None) -> dict[int, float] | None:
        """Take one frame: None if it has no pose, else the evidence each twin got from it."""
        hypotheses = _hypotheses(result)
        if not hypotheses:
            self._skipped += 1
            return None
        if self.reset_after is not None and timestamp is not None and self._time is not None and timestamp - self._time > self.reset_after:
            self.reset()
        gains: dict[int, float] = {}
        if not self._twins:
            for h in hypotheses:
                twin = _Twin(next(self._keys), h.pose, h.turn, self._evidence(h.probability))
                gains[twin.key] = twin.belief
                self._twins.append(twin)
        else:
            gate = self._gate(timestamp)
            match = _associate(self._twins, hypotheses, gate)
            for twin, j in zip(self._twins, match):
                if j is None:
                    twin.missed += 1
                    gains[twin.key] = self._evidence(MISSING_EVIDENCE)
                else:
                    twin.pose, twin.turn, twin.missed = hypotheses[j].pose, hypotheses[j].turn, 0
                    gains[twin.key] = self._evidence(hypotheses[j].probability)
                twin.belief += gains[twin.key]
            # A hypothesis far from every twin so far starts a new one, as unlikely as allowed.
            continued = {j for j in match if j is not None}
            for j, h in enumerate(hypotheses):
                if j not in continued and all(_angle(t.pose, h.pose) > gate for t in self._twins):
                    twin = _Twin(next(self._keys), h.pose, h.turn, -np.inf)
                    gains[twin.key] = self._evidence(h.probability)
                    self._twins.append(twin)
        top = max(t.belief for t in self._twins)
        for twin in self._twins:
            twin.belief = max(twin.belief - top, -self.belief_cap)
        self._twins = [t for t in self._twins if t.belief == 0.0 or t.missed <= MAX_MISSED]
        self._time, self._skipped = timestamp, 0
        return gains


def smooth_turns(
    results: list[PoseEstimationResult | None],
    timestamps: list[float] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    commit_margin: float = DEFAULT_COMMIT_MARGIN,
    belief_cap: float = DEFAULT_BELIEF_CAP,
    gate_deg: float = DEFAULT_GATE_DEG,
    frame_interval: float | None = None,
) -> list[TrackedPose]:
    """Offline: the twins followed as ``TurnTracker`` does, each frame's weighed by the evidence of
    the frames before it and, run backwards along the same twins, of the frames after it. Frames
    without a pose get an empty TrackedPose."""
    tracker = TurnTracker(temperature, commit_margin, belief_cap, gate_deg, frame_interval)
    steps = []
    for k, result in enumerate(results):
        gains = tracker._observe(result, None if timestamps is None else timestamps[k])
        if gains is not None:
            steps.append((k, [dataclasses.replace(t) for t in tracker._twins], gains))
    out = [TrackedPose(None) for _ in results]
    later: dict[int, float] = {}            # log evidence of the frames after this one, per twin
    for k, twins, gains in reversed(steps):
        unknown = -belief_cap if later else 0.0
        out[k] = _tracked(twins, np.array([t.belief + later.get(t.key, unknown) for t in twins]), commit_margin)
        later = {t.key: later.get(t.key, unknown) + gains[t.key] for t in twins}
        top = max(later.values())
        later = {key: max(value - top, -belief_cap) for key, value in later.items()}
    return out
