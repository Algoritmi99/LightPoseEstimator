//
// Created by algoritmi on 10/22/26.
//

#ifndef LIGHTPOSEESTIMLIB_POSE_H
#define LIGHTPOSEESTIMLIB_POSE_H
#include "Position.h"
#include "Quaternion.h"


class Pose {
public:
    Position position;
    Quaternion quaternion;

    Pose(const Position& position, const Quaternion& quaternion);
    Pose(double x, double y, double z, double qx, double qy, double qz, double qw);
    static Pose fromRotationMatrix(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation);

    Eigen::Vector3d getTvec() const;
    Eigen::Vector3d getRvec() const;
    Eigen::Matrix3d getRotationMatrix() const;

    friend std::ostream& operator<<(std::ostream& os, const Pose& m);
};


#endif //LIGHTPOSEESTIMLIB_POSE_H
