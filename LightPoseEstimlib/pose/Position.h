//
// Created by algoritmi on 10/22/26.
//

#ifndef LIGHTPOSEESTIMLIB_POSITION_H
#define LIGHTPOSEESTIMLIB_POSITION_H
#include <iosfwd>
#include <Eigen/Geometry>


class Position {
public:
    double x;
    double y;
    double z;

    Position(double x, double y, double z);
    [[nodiscard]] Eigen::Vector3d toTvec() const;
    friend std::ostream& operator<<(std::ostream& os, const Position& m);
};


#endif //LIGHTPOSEESTIMLIB_POSITION_H
