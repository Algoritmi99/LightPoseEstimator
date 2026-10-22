//
// Created by algoritmi on 10/22/26.
//

#ifndef LIGHTPOSEESTIMLIB_QUATERNION_H
#define LIGHTPOSEESTIMLIB_QUATERNION_H
#include <Eigen/Geometry>


class Quaternion {
public:
    double qx, qy, qz, qw;

    Quaternion(double qx, double qy, double qz, double qw);
    Eigen::Matrix3d toRotationMatrix() const;
    Eigen::Vector3d toRvec() const;
    friend std::ostream& operator<<(std::ostream& os, const Quaternion& m);
};


#endif //LIGHTPOSEESTIMLIB_QUATERNION_H
