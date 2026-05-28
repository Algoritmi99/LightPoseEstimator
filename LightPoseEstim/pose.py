import numpy as np
from scipy.spatial.transform import Rotation
import cv2


class Position:
    x: float
    y: float
    z: float

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_tvec(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32).reshape(3, 1)

    def __repr__(self):
        return f"Position(x={self.x}, y={self.y}, z={self.z})"

class Quaternion:
    qx: float
    qy: float
    qz: float
    qw: float

    def __init__(self, qx, qy, qz, qw):
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw

    def to_rotation_matrix(self):
        return Rotation.from_quat([
            self.qx,
            self.qy,
            self.qz,
            self.qw,
        ]).as_matrix()

    def to_rvec(self):
        return cv2.Rodrigues(self.to_rotation_matrix())[0]

    def __repr__(self):
        return f"Quaternion(qx={self.qx}, qy={self.qy}, qz={self.qz}, qw={self.qw})"


class Pose:
    position: Position
    quaternion: Quaternion

    def __init__(self, x, y, z, qx, qy, qz, qw):
        self.position = Position(x, y, z)
        self.quaternion = Quaternion(qx, qy, qz, qw)

    def get_tvec(self):
        return self.position.to_tvec()

    def get_rvec(self):
        return self.quaternion.to_rvec()

    def __repr__(self):
        return f"Pose(position={self.position}, quaternion={self.quaternion})"
