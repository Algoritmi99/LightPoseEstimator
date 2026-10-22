//
// Created by algoritmi on 10/22/26.
//

#include "Quaternion.h"
#include <ostream>

Quaternion::Quaternion(double qx, double qy, double qz, double qw) {
    this->qx = qx;
    this->qy = qy;
    this->qz = qz;
    this->qw = qw;
}

Eigen::Matrix3d Quaternion::toRotationMatrix() const {
    return Eigen::Quaterniond(qw, qx, qy, qz)
               .normalized()
               .toRotationMatrix();
}

Eigen::Vector3d Quaternion::toRvec() const {
    const Eigen::AngleAxisd aa(Eigen::Quaterniond(qw, qx, qy, qz).normalized());
    return aa.angle() * aa.axis();
}

std::ostream& operator<<(std::ostream& os, const Quaternion& m) {
    return os << "Quaternion(" << m.qx << ", " << m.qy << ", " << m.qz << ", " << m.qw << ")";
}
