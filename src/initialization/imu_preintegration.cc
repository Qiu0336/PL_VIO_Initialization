/**
* This file is part of ORB-SLAM3
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

#include "imu_preintegration.h"

namespace IMU {

Preintegrated::Preintegrated(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba)
{
    //NgaWalk = Preintegrated::SigmaW;
    Initialize(bg, ba);
}

void Preintegrated::IntegrateNewMeasurement(const Eigen::Vector3d& w,
                                            const Eigen::Vector3d& a,
                                            const double dt)
{
    // Position is updated first, as it depends on previously computed velocity and rotation.
    // Velocity is updated secondly, as it depends on previously computed rotation.
    // Rotation is the last to be updated.

    //Matrices to compute covariance

    Eigen::Matrix9d A;
    A.setIdentity();

    Eigen::Matrix<double, 9, 6> B;
    B.setZero();

    Eigen::Vector3d acc = a - b.tail<3>();// 测量值-ba
    //Eigen::Vector3d accW(angVel.x()-b.bwx, angVel.y()-b.bwy, angVel.z()-b.bwz);

    //avgA = (dT*avgA + dR*acc*dt)/(dT+dt);
    //avgW = (dT*avgW + accW*dt)/(dT+dt);

    // Update delta position dP and velocity dV (rely on no-updated delta rotation)
    // 计算预积分△p和△v
    dP += dV*dt + 0.5*dR*acc*dt*dt;
    dV += dR*acc*dt;

    // Compute velocity and position parts of matrices A and B (rely on non-updated delta rotation)
    Eigen::Matrix3d Wacc = Skew(acc);

    A.block<3, 3>(3, 0) = -dR*dt*Wacc;
    A.block<3, 3>(6, 0) = -0.5*dR*dt*dt*Wacc;
    A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity()*dt;
    B.block<3, 3>(3, 3) = dR*dt;
    B.block<3, 3>(6, 3) = 0.5*dR*dt*dt;

    // Update position and velocity jacobians wrt bias correction预积分里面的bias一阶展开更新
    JPa = JPa + JVa*dt -0.5*dR*dt*dt;
    JPg = JPg + JVg*dt -0.5*dR*dt*dt*Wacc*JRg;
    JVa = JVa - dR*dt;
    JVg = JVg - dR*dt*Wacc*JRg;

    // Update delta rotation 计算预积分△R

    const Eigen::Vector3d dwt = dt*(w - b.head<3>());// 轴角法表示的旋转
    const Eigen::Matrix3d deltaR = ExpSO3(dwt);// R = EXP(w^)
    const Eigen::Matrix3d rightJ = RightJacobianSO3(dwt);// R = EXP(w^)

    dR *= deltaR;// 更新量用乘法，转了w*t的角度

    // Compute rotation parts of matrices A and B
    A.block<3, 3>(0, 0) = deltaR.transpose();
    B.block<3, 3>(0, 0) = rightJ*dt;

    // Update covariance 迭代计算预积分噪声项协方差
    C = A*C*A.transpose() + B*Sigma*B.transpose();
    //C.block<6, 6>(9, 9) += NgaWalk;

    // Update rotation jacobian wrt bias correction预积分里面的旋转项的bias一阶展开更新
    JRg = deltaR.transpose()*JRg - rightJ*dt;// 这里看似和论文不一致，其实有点变换技巧

    // Total integrated time
    dT += dt;

}


void Preintegrated::IntegrateNewMeasurement_Mid(const Eigen::Vector3d& w1, const Eigen::Vector3d& a1,
                                            const Eigen::Vector3d& w2, const Eigen::Vector3d& a2,
                                            const double dt)
{

    Eigen::Matrix9d A;
    A.setIdentity();

    Eigen::Matrix<double, 9, 6> B;
    B.setZero();

    Eigen::Vector3d acc1 = a1 - b.tail<3>();// 测量值-ba
    Eigen::Vector3d acc2 = a2 - b.tail<3>();// 测量值-ba
    Eigen::Vector3d acc = 0.5*(acc1 + acc2);// 测量值-ba
    //Eigen::Vector3d accW(angVel.x()-b.bwx, angVel.y()-b.bwy, angVel.z()-b.bwz);

    //avgA = (dT*avgA + dR*acc*dt)/(dT+dt);
    //avgW = (dT*avgW + accW*dt)/(dT+dt);

    // Update delta position dP and velocity dV (rely on no-updated delta rotation)
    // 计算预积分△p和△v



    // Compute velocity and position parts of matrices A and B (rely on non-updated delta rotation)
    Eigen::Matrix3d Wacc = Skew(acc);

    A.block<3, 3>(3, 0) = -dR*dt*Wacc;
    A.block<3, 3>(6, 0) = -0.5*dR*dt*dt*Wacc;
    A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity()*dt;
    B.block<3, 3>(3, 3) = dR*dt;
    B.block<3, 3>(6, 3) = 0.5*dR*dt*dt;

    // Update position and velocity jacobians wrt bias correction预积分里面的bias一阶展开更新
    JPa = JPa + JVa*dt -0.5*dR*dt*dt;
    JPg = JPg + JVg*dt -0.5*dR*dt*dt*Wacc*JRg;
    JVa = JVa - dR*dt;
    JVg = JVg - dR*dt*Wacc*JRg;

    // Update delta rotation 计算预积分△R

    const Eigen::Vector3d dwt = dt*(0.5*(w1 + w2) - b.head<3>());// 轴角法表示的旋转
    const Eigen::Matrix3d deltaR = ExpSO3(dwt);// R = EXP(w^)

    const Eigen::Matrix3d dR1 = dR;
    dR *= deltaR;// 更新量用乘法，转了w*t的角度
    const Eigen::Matrix3d dR2 = dR;

    dP += dV*dt + 0.5*0.5*(dR1*acc1 + dR2*acc2)*dt*dt;
    dV += 0.5*(dR1*acc1 + dR2*acc2)*dt;
    const Eigen::Matrix3d rightJ = RightJacobianSO3(dwt);// R = EXP(w^)

    // Compute rotation parts of matrices A and B
    A.block<3, 3>(0, 0) = deltaR.transpose();
    B.block<3, 3>(0, 0) = rightJ*dt;

    // Update covariance 迭代计算预积分噪声项协方差
    C = A*C*A.transpose() + B*Sigma*B.transpose();
    //C.block<6, 6>(9, 9) += NgaWalk;

    // Update rotation jacobian wrt bias correction预积分里面的旋转项的bias一阶展开更新
    JRg = deltaR.transpose()*JRg - rightJ*dt;// 这里看似和论文不一致，其实有点变换技巧

    // Total integrated time
    dT += dt;

}

void Preintegrated::IntegrateNewMeasurement_Pro(const Eigen::Vector3d& w1, const Eigen::Vector3d& a1,
                                                const Eigen::Vector3d& w2, const Eigen::Vector3d& a2,
                                                const double dt)
{
    const Eigen::Vector3d w = 0.5*(w1 + w2) - b.head<3>();
//    const Eigen::Vector3d w = w1 - b.head<3>();
    const Eigen::Matrix3d dR1 = dR;
    const Eigen::Vector3d dwt = dt*w;// 轴角法表示的旋转
    const Eigen::Matrix3d deltaR = ExpSO3(dwt);// R = EXP(w^)
    dR *= deltaR;// 更新量用乘法，转了w*t的角度
    const Eigen::Matrix3d dR2 = dR;

    double w_norm = w.norm();
    double w_norm2 = w_norm*w_norm;
    double w_norm3 = w_norm2*w_norm;
    double w_norm4 = w_norm3*w_norm;
    double wt = w_norm*dt;
    double wt2 = wt*wt;
    double sinwt = sin(wt);
    double coswt = cos(wt);
    double dt2 = dt*dt;
    double dt3 = dt2*dt;
    double dt4 = dt3*dt;
    Eigen::Matrix3d Skew_w = Skew(w);
    Eigen::Matrix3d Skew_w2 = Skew(w)*Skew(w);

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d A;
    Eigen::Matrix3d B;
    Eigen::Matrix3d C;

    if(w_norm < 10e-4)
    {
        std::cout << "///// " << std::endl;
        A = I*dt + 0.5*dt2*Skew_w + dt3*Skew_w2 / 6.0f;
        B = 0.5*I*dt2 + dt3*Skew_w / 3.0f + 0.125*dt4*Skew_w2;
        C = I*dt3 / 3.0f + 0.25*dt4*Skew_w + 0.1*dt4*dt*Skew_w2;
    }
    else
    {
        A = I*dt + Skew_w*(1 - coswt)/w_norm2 + Skew_w2*(wt - sinwt)/w_norm3;
        B = 0.5*I*dt2 + Skew_w*(sinwt - wt*coswt)/w_norm3 + Skew_w2*(0.5*wt2 + 1 - wt*sinwt - coswt)/w_norm4;
        C = I*dt3/3.0f + Skew_w*(2*coswt + 2*wt*sinwt - wt2*coswt - 2)/w_norm4 + Skew_w2*(wt2*wt/3.0f + 2*sinwt - wt2*sinwt - 2*wt*coswt)/w_norm4*w_norm;
    }


    Eigen::Vector3d ba = b.tail<3>();

    dP += dV*dt + dR1*(A*(a1 - ba)*dt + B*(a2 - 2*a1 + ba) + C*(a1 - a2)/dt);
    dV += dR1*(A*(a1 - ba) + B*(a2 - a1)/dt);

}


void Preintegrated::SetOriginalGyroBias(const Eigen::Vector3d& bg) {
    b.head<3>() = bg;
}

void Preintegrated::SetOriginalAccBias(const Eigen::Vector3d& ba) {
    b.tail<3>() = ba;
}

void Preintegrated::SetNewGyroBias(const Eigen::Vector3d& bg) {
    bu.head<3>() = bg;
    db.head<3>() = bg - b.head<3>();
}

void Preintegrated::SetNewAccBias(const Eigen::Vector3d& ba) {
    bu.tail<3>() = ba;
    db.tail<3>() = ba - b.tail<3>();
}

Eigen::Vector3d Preintegrated::GetGyroDeltaBias() const {
    return db.head<3>();
}

Eigen::Vector3d Preintegrated::GetGyroDeltaBias(const Eigen::Vector3d& bg) const {
    return bg-b.head<3>();
}

Eigen::Vector3d Preintegrated::GetGyroOriginalBias() const {
    return b.head<3>();
}

Eigen::Vector3d Preintegrated::GetGyroUpdatedBias() const {
    return bu.head<3>();
}

Eigen::Vector3d Preintegrated::GetAccDeltaBias() const {
    return db.tail<3>();
}

Eigen::Vector3d Preintegrated::GetAccDeltaBias(const Eigen::Vector3d& ba) const {
    return ba-b.tail<3>();
}

Eigen::Vector3d Preintegrated::GetAccOriginalBias() const {
    return b.tail<3>();
}

Eigen::Vector3d Preintegrated::GetAccUpdatedBias() const {
    return bu.tail<3>();
}

Eigen::Matrix3d Preintegrated::GetDeltaRotation(const Eigen::Vector3d& bg) const {
    return dR*ExpSO3(JRg*(bg-b.head<3>()));
}

Eigen::Vector3d Preintegrated::GetDeltaVelocity(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) const {
    return dV + JVg*(bg-b.head<3>()) + JVa*(ba-b.tail<3>());
}

Eigen::Vector3d Preintegrated::GetDeltaPosition(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) const {
    return dP + JPg*(bg-b.head<3>()) + JPa*(ba-b.tail<3>());
}

Eigen::Matrix3d Preintegrated::GetUpdatedDeltaRotation() const {
    return dR*ExpSO3(JRg*db.head<3>());
}

Eigen::Vector3d Preintegrated::GetUpdatedDeltaVelocity() const {
    return dV + JVg*db.head<3>() + JVa*db.tail<3>();
}

Eigen::Vector3d Preintegrated::GetUpdatedDeltaPosition() const {
    return dP + JPg*db.head<3>() + JPa*db.tail<3>();
}

Eigen::Matrix3d Preintegrated::GetOriginalDeltaRotation() const {
    return dR;
}

Eigen::Vector3d Preintegrated::GetOriginalDeltaVelocity() const {
    return dV;
}

Eigen::Vector3d Preintegrated::GetOriginalDeltaPosition() const {
    return dP;
}

void Preintegrated::Initialize(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
    dT = 0.0;
    C.setZero();

    b.head<3>() = bg;
    b.tail<3>() = ba;
    dR.setIdentity();
    dV.setZero();
    dP.setZero();
    JRg.setZero();
    JVg.setZero();
    JVa.setZero();
    JPg.setZero();
    JPa.setZero();
//    avgA.setZero();
//    avgW.setZero();

    bu = b;
    db.setZero();
}

} // namespace IMU
