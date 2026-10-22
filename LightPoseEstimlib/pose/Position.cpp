//
// Created by algoritmi on 10/22/26.
//

#include "Position.h"
#include <ostream>

Position::Position(double x, double y, double z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

Eigen::Vector3d Position::toTvec() const {
    return {static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
}

std::ostream& operator<<(std::ostream& os, const Position& m) {
    return os << "Position(" << m.x << ", " << m.y << ", " << m.z << ")";
}