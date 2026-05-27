class Position:
    x: float
    y: float
    z: float

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

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

    def __repr__(self):
        return f"Quaternion(qx={self.qx}, qy={self.qy}, qz={self.qz}, qw={self.qw})"


class Pose:
    position: Position
    quaternion: Quaternion

    def __init__(self, x, y, z, qx, qy, qz, qw):
        self.position = Position(x, y, z)
        self.quaternion = Quaternion(qx, qy, qz, qw)

    def __repr__(self):
        return f"Pose(position={self.position}, quaternion={self.quaternion})"
