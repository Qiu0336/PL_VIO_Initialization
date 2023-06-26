// Adapted from ORB-SLAM3
// by David Zuñiga-Noël [dzuniga@uma.es]

/**
* This file *was* part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef IMU_PREINTEGRATION_H
#define IMU_PREINTEGRATION_H

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include "so3.h"

namespace Eigen {
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
} // namespace Eigen

namespace IMU {

const double rate = 200.;
const double dt = 1./rate;
const double ng = 1.7e-4;
const double na = 2.e-3;

const double GRAVITY_MAGNITUDE = 9.81;
const Eigen::Vector3d GRAVITY_VECTOR(0.0, 0.0, -GRAVITY_MAGNITUDE);
static Eigen::Matrix6d Sigma = Eigen::Matrix6d::Identity(); // discrete

class Measurement
{
    public:
    Measurement() { }
    Measurement(const double w_x, const double w_y, const double w_z,
              const double a_x, const double a_y, const double a_z,
              const double dt): w(w_x, w_y, w_z), a(a_x, a_y, a_z), dt(dt) { }

    Measurement(const Eigen::Vector3d& a, const Eigen::Vector3d& w, const double dt)
    : w(w), a(a), dt(dt) { }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
    Eigen::Vector3d w;
    Eigen::Vector3d a;
    double dt;
};

//Preintegration of IMU Measurements
class Preintegrated
{
    public:
    Preintegrated(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba);
    ~Preintegrated() {}
    void IntegrateNewMeasurement(const Eigen::Vector3d& w, const Eigen::Vector3d& a, const double dt);

    void IntegrateNewMeasurement_Mid(const Eigen::Vector3d& w1, const Eigen::Vector3d& a1,
                                     const Eigen::Vector3d& w2, const Eigen::Vector3d& a2,
                                     const double dt);
    void IntegrateNewMeasurement_Pro(const Eigen::Vector3d& w1, const Eigen::Vector3d& a1,
                                     const Eigen::Vector3d& w2, const Eigen::Vector3d& a2,
                                     const double dt);

    void SetOriginalGyroBias(const Eigen::Vector3d& bg);// 自己加的
    void SetOriginalAccBias(const Eigen::Vector3d& ba);// 自己加的

    void SetNewGyroBias(const Eigen::Vector3d& bg);
    void SetNewAccBias(const Eigen::Vector3d& ba);
    Eigen::Vector3d GetGyroDeltaBias() const;
    Eigen::Vector3d GetGyroDeltaBias(const Eigen::Vector3d& bg) const;
    Eigen::Vector3d GetGyroOriginalBias() const;
    Eigen::Vector3d GetGyroUpdatedBias() const;
    Eigen::Vector3d GetAccDeltaBias() const;
    Eigen::Vector3d GetAccDeltaBias(const Eigen::Vector3d& ba) const;
    Eigen::Vector3d GetAccOriginalBias() const;
    Eigen::Vector3d GetAccUpdatedBias() const;
    Eigen::Matrix3d GetDeltaRotation(const Eigen::Vector3d& bg) const;
    Eigen::Vector3d GetDeltaVelocity(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) const;
    Eigen::Vector3d GetDeltaPosition(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) const;
    Eigen::Matrix3d GetUpdatedDeltaRotation() const;
    Eigen::Vector3d GetUpdatedDeltaVelocity() const;
    Eigen::Vector3d GetUpdatedDeltaPosition() const;
    Eigen::Matrix3d GetOriginalDeltaRotation() const;
    Eigen::Vector3d GetOriginalDeltaVelocity() const;
    Eigen::Vector3d GetOriginalDeltaPosition() const;
  
    Preintegrated() {}
    void Initialize(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
    double dT;
    Eigen::Matrix9d C;

    Eigen::Matrix6d Nga; //, NgaWalk;

    // Values for the original bias (when integration was computed)
    Eigen::Vector6d b;// bias，包括gyro和acc
    Eigen::Matrix3d dR;
    Eigen::Vector3d dV, dP;
    Eigen::Matrix3d JRg, JVg, JVa, JPg, JPa;

    //Eigen::Vector3d avgA, avgW;

    private:
    // Updated bias
    Eigen::Vector6d bu;
    // Dif between original and updated bias
    // This is used to compute the updated values of the preintegration
    Eigen::Vector6d db;


};

} // namespace IMU

#endif // IMU_PREINTEGRATION_H
