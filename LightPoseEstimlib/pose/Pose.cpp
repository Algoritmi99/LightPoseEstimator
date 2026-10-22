//
// Created by algoritmi on 10/22/26.
//

#include "Pose.h"
#include "Position.h"
#include "Quaternion.h"

Pose::Pose(const Position& position, const Quaternion& quaternion)
    : position(position),
      quaternion(quaternion) {}

Pose::Pose(double x, double y, double z, double qx, double qy, double qz, double qw)
    : position(x, y, z),
      quaternion(qx, qy, qz, qw) {}

Pose Pose::fromRotationMatrix(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) {
    const Eigen::Quaterniond q(rotation);
    return {translation.x(), translation.y(), translation.z(), q.x(), q.y(), q.z(), q.w()};
}

Eigen::Vector3d Pose::getTvec() const {
    return this->position.toTvec();
}

Eigen::Vector3d Pose::getRvec() const {
    return this->quaternion.toRvec();
}

Eigen::Matrix3d Pose::getRotationMatrix() const {
    return this->quaternion.toRotationMatrix();
}

std::ostream& operator<<(std::ostream& os, const Pose& m) {
    return os <<"Pose(position=" << m.position << ", quaternion=" << m.quaternion << ")";
}
