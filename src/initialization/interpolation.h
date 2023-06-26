
#ifndef INTERPOLATION_H_
#define INTERPOLATION_H_

// STL
#include <iostream>
#include <vector>
//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "so3.h"

class BSpline_Cubic { // 三次B样条插值
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    public:
    BSpline_Cubic() { }
    BSpline_Cubic(int frames);

    bool Set_ControlPoints(std::vector<Eigen::Vector3d> Points);

    // pj应该 ∈ [5, nframes - 1]
    // 所以取t的时候应该保证  t ∈ [2*dt, 1 - 3*dt],dt为节点表间隔
    Eigen::Vector3d Get_Interpolation(double t);
    Eigen::Vector3d Get_Interpolation_FirstOrder(double t);
    Eigen::Vector3d Get_Interpolation_SecondOrder(double t);

    public:
    int frames;
    double dt;
    Eigen::MatrixXd A;
    Eigen::MatrixXd Z;// ControlPoints = 6 * A(-1) * Z
    Eigen::VectorXd Knots;
    Eigen::MatrixXd ControlPoints;
};

std::vector<Eigen::Matrix3d> Rot_Interpolation(Eigen::Matrix3d R0, Eigen::Matrix3d R10,
                                               std::vector<Eigen::Vector3d> w, const double dt);


#endif
