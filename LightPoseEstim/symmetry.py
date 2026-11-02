"""
Near-symmetries of the object, for keypoints chosen in symmetric sets.

A box-like object such as the Astrobee looks almost the same after a 180-degree turn about
any of its axes; only details (markers, a window, a screen) tell the turns apart. Asking the
keypoint heatmaps to tell them apart makes them memorise context, and on unseen recordings
they then label keypoints with their mirrored twins. So keypoints come in sets that the turns
map onto each other (keypoint j of set o is set o's base point turned by turn e_j, snapped onto
the surface), and the heatmaps may use any of the four labellings the turns induce. A separate
classifier picks the turn that maps the labelling used to the true one, and PnP pairs each
heatmap with the 3D keypoint it then shows.
"""
from dataclasses import dataclass, field

import numpy as np

# Identity and 180-degree turns about the object x, y and z axes, in that order (the order of
# pose_estimator._SYMMETRY_TURNS). They form a group: each is its own inverse, and the product
# of any two is the third.
TURNS = np.stack([np.diag(signs).astype(np.float64) for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))])

# Labelling ambiguity tolerated near the boundary between two canonical labellings (degrees).
DEFAULT_TOLERANCE_DEG = 15.0
# A view tells a turn's twins apart when this share of the visible surface (by projected area)
# differs under the turn: not at all below the first value, fully above the second. Measured on
# the Astrobee's vertical turn: below 2.5% the turn classifier was wrong in 70% of unseen frames,
# at 2.5-3% in half, above 5% in 2-4%.
DEFAULT_DISTINCT_RANGE = (0.025, 0.05)


def compose_table(turns: np.ndarray) -> np.ndarray:
    """compose[a, b] is the index of turns[a] @ turns[b]."""
    count = len(turns)
    table = np.empty((count, count), dtype=np.int64)
    for a in range(count):
        for b in range(count):
            product = turns[a] @ turns[b]
            table[a, b] = int(np.argmin([np.abs(product - t).sum() for t in turns]))
    return table


def _angles_to_reference(allocentric: np.ndarray, reference: np.ndarray, turns: np.ndarray) -> np.ndarray:
    """
    Degrees between each ``allocentric @ turns[g]`` and ``reference``: [..., G] for
    allocentric [..., 3, 3]. For the diagonal turns only the diagonal of reference^T @ R counts.
    """
    diagonal = np.einsum("ki,...ki->...i", reference, allocentric)          # diag(reference^T @ R)
    signs = np.stack([np.diag(t) for t in turns])                           # [G, 3]
    traces = diagonal @ signs.T                                             # [..., G]
    return np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))


@dataclass
class KeypointSymmetry:
    """
    ``permutation[g, j]``: under labelling g, heatmap j shows keypoint permutation[g, j]; the
    pose PnP then finds is the true one turned by turns[g] (object frame, right-multiplied).

    ``reference`` fixes the canonical labelling for training: the turn whose allocentric
    rotation ``R_allo @ turns[g]`` is closest to it. Within ``tolerance_deg`` of the closest,
    any turn is allowed, so frames near a boundary between labellings get no forced choice.
    """

    turns: np.ndarray
    permutation: np.ndarray
    reference: np.ndarray = field(default_factory=lambda: np.eye(3))
    tolerance_deg: float = DEFAULT_TOLERANCE_DEG
    # Where the object's twins differ (TurnDistinctness), for soft turn targets in training.
    distinctness: "TurnDistinctness | None" = None

    @classmethod
    def from_keypoints(cls, keypoints, turns: np.ndarray = TURNS, **kwargs) -> "KeypointSymmetry | None":
        """From keypoints with ``orbit`` and ``turn`` set (as select_symmetric_keypoints gives); None otherwise."""
        if not keypoints or any(getattr(kp, "orbit", None) is None for kp in keypoints):
            return None
        index = {(kp.orbit, kp.turn): j for j, kp in enumerate(keypoints)}
        compose = compose_table(turns)
        permutation = np.array([
            [index[(kp.orbit, int(compose[g, kp.turn]))] for kp in keypoints]
            for g in range(len(turns))
        ], dtype=np.int64)
        return cls(turns=np.asarray(turns, np.float64), permutation=permutation, **kwargs)

    def allowed_turns(self, allocentric: np.ndarray) -> np.ndarray:
        """Boolean [G]: labellings a training frame with this allocentric rotation may use."""
        angles = _angles_to_reference(np.asarray(allocentric, np.float64), self.reference, self.turns)
        return angles <= angles.min() + self.tolerance_deg

    def canonical_turn(self, allocentric: np.ndarray) -> int:
        return int(np.argmin(_angles_to_reference(np.asarray(allocentric, np.float64), self.reference, self.turns)))


@dataclass
class TurnDistinctness:
    """
    Where the object differs from its turned copies: surface samples (object frame) and, per
    turn, which of them land off the surface when turned. ``informativeness`` says how well a
    view can tell each turn's twins apart: 0 when (almost) nothing that differs faces the
    camera, so the twins look alike, 1 when enough of it does.
    """

    points: np.ndarray            # [P, 3]
    normals: np.ndarray           # [P, 3]
    differs: np.ndarray           # [G, P] bool; row 0 (identity) is all False
    distinct_range: tuple[float, float] = DEFAULT_DISTINCT_RANGE

    @classmethod
    def from_mesh(cls, mesh, turns: np.ndarray = TURNS, samples: int = 6000, tolerance: float = 0.004, seed: int = 0):
        import trimesh

        points, faces = trimesh.sample.sample_surface(mesh, samples, seed=seed)
        normals = mesh.face_normals[faces]
        differs = np.zeros((len(turns), len(points)), dtype=bool)
        for g in range(1, len(turns)):
            _, distance, _ = trimesh.proximity.closest_point(mesh, points @ turns[g].T)
            differs[g] = distance > tolerance
        return cls(np.asarray(points, np.float64), np.asarray(normals, np.float64), differs)

    def save(self, path) -> None:
        np.savez(path, points=self.points, normals=self.normals, differs=self.differs, distinct_range=np.asarray(self.distinct_range))

    @classmethod
    def load(cls, path) -> "TurnDistinctness":
        with np.load(path) as data:
            return cls(data["points"], data["normals"], data["differs"], tuple(float(v) for v in data["distinct_range"]))

    def informativeness(self, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """[G] in [0, 1] for an object pose (rotation [3, 3], translation [3]); identity is 1.
        Nearly the same for a pose and its turned twins: what differs under a turn is itself
        (up to sampling) symmetric under it."""
        rotation = np.asarray(rotation, np.float64)
        points = self.points @ rotation.T + np.asarray(translation, np.float64).reshape(1, 3)
        normals = self.normals @ rotation.T
        facing = np.clip(np.einsum("ij,ij->i", normals, -points / np.linalg.norm(points, axis=1, keepdims=True)), 0.0, None)
        share = (self.differs * facing).sum(axis=1) / max(facing.sum(), 1e-12)
        low, high = self.distinct_range
        x = np.clip((share - low) / (high - low), 0.0, 1.0)
        info = x * x * (3.0 - 2.0 * x)
        info[0] = 1.0
        return info


def twin_balance_weights(
    allocentric: np.ndarray,
    symmetry: "KeypointSymmetry",
    radius_deg: float = 20.0,
    low: float = 0.5,
    high: float = 5.0,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn-loss weights for training crops (allocentric rotations [N, 3, 3]) that keep the turn
    classifier from learning which twin a view usually is. Twins share a canonical view
    (R_allo @ turns[canonical turn]) and differ in their canonical turn; among the crops whose
    canonical view lies within ``radius_deg``, ``share`` is the fraction with the same turn and
    ``present`` the number of turns with at least 5% of them. Each crop gets
    1 / (max(present, 2) share), clipped to [low, high] and normalised to mean 1, so around every
    view the twins present weigh the same in total: the rare twin of a common view is up-weighted
    (up to ``high``), and a view only ever seen as one twin, which can only teach the shortcut,
    weighs half. With ``sample_weights`` (how often training draws each crop), shares count drawn
    crops, so the balance holds for what the classifier sees. Returns the weights and the shares.
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    allocentric = np.asarray(allocentric, np.float64)
    drawn = np.ones(len(allocentric)) if sample_weights is None else np.asarray(sample_weights, np.float64)
    turn = np.array([symmetry.canonical_turn(a) for a in allocentric])
    quaternions = Rotation.from_matrix(allocentric @ symmetry.turns[turn]).as_quat()
    quaternions *= np.where(quaternions[:, 3:4] < 0, -1.0, 1.0)
    chord = 2.0 * np.sin(np.radians(radius_deg) / 4.0)         # |q1 - q2| at that rotation angle
    counts = np.zeros((len(allocentric), len(symmetry.turns)))
    for g in range(len(symmetry.turns)):
        # Counted per sampling weight (there are only a few), each count times its weight.
        for value in np.unique(drawn[turn == g]):
            members = quaternions[(turn == g) & (drawn == value)]
            # q and -q are the same rotation: search both.
            tree = cKDTree(np.concatenate([members, -members]))
            counts[:, g] += value * tree.query_ball_point(quaternions, chord, return_length=True)
    total = np.maximum(counts.sum(axis=1), 1e-12)
    share = counts[np.arange(len(turn)), turn] / total
    present = np.maximum((counts / total[:, None] >= 0.05).sum(axis=1), 2)
    weights = np.clip(1.0 / (present * np.maximum(share, 1e-6)), low, high)
    return weights / np.average(weights, weights=drawn), share


def choose_reference(
    allocentric: np.ndarray,
    turns: np.ndarray = TURNS,
    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
    candidates: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Reference rotation that puts the boundaries between canonical labellings where the
    training frames are sparse: of ``candidates`` random rotations, the one leaving the fewest
    frames (allocentric [N, 3, 3]) with two labellings within ``tolerance_deg``. Returns it and
    that share of frames.
    """
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    if len(allocentric) > 6000:
        allocentric = allocentric[rng.choice(len(allocentric), 6000, replace=False)]
    references = np.concatenate([np.eye(3)[None], Rotation.random(candidates - 1, random_state=seed).as_matrix()])
    best, best_share = references[0], 1.0
    for start in range(0, len(references), 256):
        block = references[start:start + 256]                               # [C, 3, 3]
        diagonal = np.einsum("cki,nki->cni", block, allocentric)            # [C, N, 3]
        signs = np.stack([np.diag(t) for t in turns])
        angles = np.degrees(np.arccos(np.clip((diagonal @ signs.T - 1.0) / 2.0, -1.0, 1.0)))
        two = np.sort(angles, axis=2)[..., :2]
        share = ((two[..., 1] - two[..., 0]) < tolerance_deg).mean(axis=1)
        k = int(np.argmin(share))
        if share[k] < best_share:
            best, best_share = block[k], float(share[k])
    return best, best_share
