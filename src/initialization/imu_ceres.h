
#ifndef IMU_CERES_H
#define IMU_CERES_H

#include <cmath>
#include <memory>

#include <ceres/local_parameterization.h>
#include <ceres/sized_cost_function.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "initialization.h"
#include "imu_preintegration.h"
#include "so3.h"


class ScaleParameterization : public ceres::LocalParameterization {
    public:
    virtual ~ScaleParameterization() {}
// Implements x_plus_delta = x*exp(delta)，参数s的更新方式
    bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override
    {
        x_plus_delta[0] = x[0]*std::exp(delta[0]);
        return true;
    }
// 新参数s相对于老参数的雅克比，用一阶泰勒展开（delta_s=0处）
    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        jacobian[0] = x[0];// s*exp(delta_s)=s+s*delta_s
        return true;
    }
    int GlobalSize() const override { return 1; }// 参数的实际维度
    int LocalSize() const override { return 1; }// 正切空间上的维度
};


class GravityParameterization : public ceres::LocalParameterization {
    public:
    virtual ~GravityParameterization() {}
    bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Matrix3d> R(x);
        Eigen::Matrix3d delta_R = ExpSO3(delta[0], delta[1], 0.0);
        Eigen::Map<Eigen::Matrix3d> result(x_plus_delta);
        result = R*delta_R;
        return true;
    }

// 新参数Rwg相对于老参数的雅克比,这里是Rwg对δα和δβ的一阶导数，用一阶泰勒展开（δg=0处）
// 将R=REXP(δα,δβ,0)用罗德里格斯公式展开取δφ趋于0，然后在取一阶导数
    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Eigen::Map<const Eigen::Matrix3d> R(x);
        Eigen::Map<Eigen::Matrix<double, 9, 2, Eigen::RowMajor>> J(jacobian);
        J.block<3, 1>(0, 0).setZero();
        J.block<3, 1>(3, 0) =  R.block<3, 1>(0, 2);
        J.block<3, 1>(6, 0) = -R.block<3, 1>(0, 1);
        J.block<3, 1>(0, 1) = -R.block<3, 1>(0, 2);
        J.block<3, 1>(3, 1).setZero();
        J.block<3, 1>(6, 1) =  R.block<3, 1>(0, 0);
        return true;
    }
    int GlobalSize() const override { return 9; }// 参数的实际维度
    int LocalSize() const override { return 2; }// 正切空间上的维度，这里Rwg实则2维
};


// proposed方法中优化gyro bias残差项的定义，第一个3为残差维度，第二个3为待优化参数维度
class GyroscopeBiasCostFunction : public ceres::SizedCostFunction<3, 3> {
    public:
    GyroscopeBiasCostFunction(std::shared_ptr<const IMU::Preintegrated> pInt, const Eigen::Matrix3d& Ri, const Eigen::Matrix3d& Rj)
    : pInt(pInt), Ri(Ri), Rj(Rj)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(pInt->C.block<3, 3>(0, 0));
        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~GyroscopeBiasCostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);

        // 得到优化gyro的残差项
        const Eigen::Matrix3d eR = pInt->GetDeltaRotation(bg).transpose()*Ri.transpose()*Rj;
        const Eigen::Vector3d er = LogSO3(eR);

        Eigen::Map<Eigen::Vector3d> e(residuals);
        e = er;
        e = SqrtInformation*e;

        // 雅克比解析式
        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                // wrt gyro bias
                const Eigen::Vector3d dbg = pInt->GetGyroDeltaBias(bg);
                const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);

                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = -invJr*eR.transpose()*RightJacobianSO3(pInt->JRg*dbg)*pInt->JRg;
                J = SqrtInformation*J;
            }
        }
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    std::shared_ptr<const IMU::Preintegrated> pInt;
    const Eigen::Matrix3d Ri, Rj;
    Eigen::Matrix3d SqrtInformation;
};

// iterative方法中的不带bias先验的残差项定义
// velocity1, velocity2, bias_g, bias_a, Rwg, scale
// SizedCostFunction中，9为残差维度，333391为后面待优化参数维度
class InertialCostFunction : public ceres::SizedCostFunction<9, 3, 3, 3, 3, 9, 1> {
    public:
    InertialCostFunction(std::shared_ptr<const IMU::Preintegrated> pInt,
                         const Eigen::Matrix3d &R1, const Eigen::Vector3d &p1,
                         const Eigen::Matrix3d &R2, const Eigen::Vector3d &p2,
                         const Eigen::Isometry3d &Tcb = Eigen::Isometry3d::Identity())
        : pInt(pInt), dt(pInt->dT), R1_(R1), R2_(R2), p1_(p1), p2_(p2), Tcb(Tcb)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix9d> solver(pInt->C);// 预积分的协方差
        SqrtInformation = solver.operatorInverseSqrt();// 信息矩阵的平方根，即协方差矩阵逆的平方根
    }
    virtual ~InertialCostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override
    {
        Eigen::Map<const Eigen::Vector3d> v1(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> v2(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> bg(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> ba(parameters[3]);
        Eigen::Map<const Eigen::Matrix3d> Rwg(parameters[4]);
        const double s = parameters[5][0];

        const Eigen::Vector3d g = Rwg*IMU::GRAVITY_VECTOR;

        const Eigen::Matrix3d dR = pInt->GetDeltaRotation(bg);
        const Eigen::Vector3d dV = pInt->GetDeltaVelocity(bg, ba);
        const Eigen::Vector3d dP = pInt->GetDeltaPosition(bg, ba);

        const Eigen::Matrix3d R1 = R1_*Tcb.linear();
        const Eigen::Matrix3d R2 = R2_*Tcb.linear();

        const Eigen::Matrix3d eR = dR.transpose()*R1.transpose()*R2;
        const Eigen::Vector3d er = LogSO3(eR);// 旋转项的残差

        // IMU残差项定义，维度为9
        Eigen::Map<Eigen::Vector9d> e(residuals);
        e.head<3>()     = er;
        e.segment<3>(3) = R1.transpose()*(s*(v2 - v1) - g*dt) - dV;
        e.tail<3>()     = R1.transpose()*(s*(p2_ - p1_ - v1*dt) - 0.5*g*dt*dt + (R2_-R1_)*Tcb.translation()) - dP;
        // 这里第三项后面加入R1.transpose()*(R2_-R1_)*Tcb.translation()外参限制，即等于tbc2-tbc1，
        // 应该是用平移外参限制，在这里这一项应该没什么用
    
        e = SqrtInformation*e;// 乘上信息矩阵权重，ceres中自动平方

        if (jacobians != nullptr)
        {// 这里求解雅克比
            const Eigen::Vector3d dbg = pInt->GetGyroDeltaBias(bg);
            const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);
            if (jacobians[0] != nullptr) {// 残差对v1的雅克比
                // wrt velocity1
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[0]);
                J.block<3, 3>(0, 0).setZero();// r△R
                J.block<3, 3>(3, 0) = -s*R1.transpose();// r△v
                J.block<3, 3>(6, 0) = -s*dt*R1.transpose();// r△p
                J = SqrtInformation*J;
            }
            if (jacobians[1] != nullptr) {// 残差对v2的雅克比
                // wrt velocity2
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[1]);
                J.block<3, 3>(0, 0).setZero();
                J.block<3, 3>(3, 0) = s*R1.transpose();
                J.block<3, 3>(6, 0).setZero();
                J = SqrtInformation*J;
            }
            if (jacobians[2] != nullptr) {// 残差对bg的雅克比
                // wrt gyro bias
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[2]);
                J.block<3, 3>(0, 0) = -invJr*eR.transpose()*RightJacobianSO3(pInt->JRg*dbg)*pInt->JRg;
                J.block<3, 3>(3, 0) = -pInt->JVg;
                J.block<3, 3>(6, 0) = -pInt->JPg;
                J = SqrtInformation*J;
            }
            if (jacobians[3] != nullptr) {// 残差对ba的雅克比
                // wrt acc bias
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobians[3]);
                J.block<3, 3>(0, 0).setZero();
                J.block<3, 3>(3, 0) = -pInt->JVa;
                J.block<3, 3>(6, 0) = -pInt->JPa;
                J = SqrtInformation*J;
            }
            if (jacobians[4] != nullptr) {// 残差对Rwg的雅克比//论文中给出，和论文中有点不同？？？？？？？
                // wrt Rwg
                Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> J(jacobians[4]);
                J.setZero();
                J.block<3, 3>(3, 6) = dt*IMU::GRAVITY_MAGNITUDE*R1.transpose();
                J.block<3, 3>(6, 6) = 0.5*dt*dt*IMU::GRAVITY_MAGNITUDE*R1.transpose();
                J = SqrtInformation*J;
            }
            if (jacobians[5] != nullptr) {// 残差对s的雅克比//论文中给出，和论文中有点不同？？？？？？？
                // wrt scale
                Eigen::Map<Eigen::Vector9d> J(jacobians[5]);
                J.block<3, 1>(0, 0).setZero();
                J.block<3, 1>(3, 0) = R1.transpose()*(v2 - v1);
                J.block<3, 1>(6, 0) = R1.transpose()*(p2_ - p1_ - v1*dt);
                J = SqrtInformation*J;
            }
        }

    return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    std::shared_ptr<const IMU::Preintegrated> pInt;
    const double dt;

    const Eigen::Matrix3d R1_, R2_;
    const Eigen::Vector3d p1_, p2_;
    const Eigen::Isometry3d Tcb;

    Eigen::Matrix9d SqrtInformation;
};

class InertialCostFunction_BSpline : public ceres::SizedCostFunction<3, 1, 3, 9> {
 public:
  InertialCostFunction_BSpline(const Eigen::Matrix<double, 3, 7> &Mk,
                               const Eigen::Vector3d &Pik): Mk(Mk), Pik(Pik)
  {
//    SqrtInformation = solver.operatorInverseSqrt();// 信息矩阵的平方根，即协方差矩阵逆的平方根
    SqrtInformation = Eigen::Matrix3d::Identity();
  }
  virtual ~InertialCostFunction_BSpline() { }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {

    const Eigen::Vector3d Ps  = Mk.col(0);
    const Eigen::Matrix3d Pg  = Mk.block<3, 3>(0, 1);
    const Eigen::Matrix3d Pba = Mk.block<3, 3>(0, 4);

    const double s = parameters[0][0];
    Eigen::Map<const Eigen::Vector3d> ba(parameters[1]);
    Eigen::Map<const Eigen::Matrix3d> Rwg(parameters[2]);

    const Eigen::Vector3d g = Rwg*IMU::GRAVITY_VECTOR;

    // 残差项定义，维度为3

    Eigen::Map<Eigen::Vector3d> e(residuals);
    e = Ps*s + Pba*ba + Pg*g - Pik;
    e = SqrtInformation*e;// 乘上信息矩阵权重，ceres中自动平方

    if (jacobians != nullptr) {// 这里求解雅克比
      if (jacobians[0] != nullptr) {
        // wrt scale
        Eigen::Map<Eigen::Vector3d> J(jacobians[0]);
        J = Ps;
        J = SqrtInformation*J;
      }
      if (jacobians[1] != nullptr) {// 残差对ba的雅克比
        // wrt acc bias
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[1]);
        J = Pba;
        J = SqrtInformation*J;
      }
      if (jacobians[2] != nullptr) {// 残差对Rwg的雅克比
        // wrt Rwg
        Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> J(jacobians[2]);
        J.setZero();
        J.block<3, 3>(0, 6) = - Pg*IMU::GRAVITY_MAGNITUDE;
        J = SqrtInformation*J;
      }
    }

    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:

  const Eigen::Matrix<double, 3, 7> Mk;
  const Eigen::Vector3d Pik;

  Eigen::Matrix3d SqrtInformation;
};


// bias
class BiasPriorCostFunction : public ceres::SizedCostFunction<3, 3> {
    public:
    BiasPriorCostFunction(const double variance, const Eigen::Vector3d &mean = Eigen::Vector3d::Zero())
    : weight(std::sqrt(variance)), mean(mean) { }
    virtual ~BiasPriorCostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bias(parameters[0]);

        Eigen::Map<Eigen::Vector3d> error(residuals);
        error = weight*(mean - bias);

        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = -weight*Eigen::Matrix3d::Identity();
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    const double weight;
    const Eigen::Vector3d mean;
};


class Point2LineCostFunction : public ceres::SizedCostFunction<3, 3, 1, 1, 1> {
    public:
    Point2LineCostFunction(const Eigen::Vector3d &u11, const Eigen::Vector3d &u12,
                           const Eigen::Vector3d &u21, const Eigen::Vector3d &u22,
                           const IMU::Preintegrated &pInt,
                           const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : u11(u11), u12(u12), u21(u21), u22(u22), pInt(pInt), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~Point2LineCostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        const double lamda12 = parameters[1][0];
        const double lamda21 = parameters[2][0];
        const double lamda22 = parameters[3][0];

        const Eigen::Matrix3d Rbc = Tbc.rotation();
        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);

        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        e = Rbc*(u11 - lamda12*u12) - DR*Rbc*(lamda21*u21 - lamda22*u22);

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Vector3d Error2 = - Rbc*(lamda21*u21 - lamda22*u22);
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = - DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg;
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[1]);
                J = - Rbc*u12;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[2]);
                J = - DR*Rbc*u21;
            }
            if (jacobians[3] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[3]);
                J = DR*Rbc*u22;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u11;
    const Eigen::Vector3d u12;
    const Eigen::Vector3d u21;
    const Eigen::Vector3d u22;
    const IMU::Preintegrated pInt;
    const Eigen::Isometry3d Tbc;
};



class CloseformCostFunction : public ceres::SizedCostFunction<3, 3, 3, 3, 1, 1> {
    public:
    CloseformCostFunction(const Eigen::Vector3d &u1, const Eigen::Vector3d &u2,
                          const IMU::Preintegrated &pInt, const double &delt,
                          const Eigen::Vector3d &gtba,
                          const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : u1(u1), u2(u2), pInt(pInt), delt(delt), gtba(gtba), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~CloseformCostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
//        Eigen::Map<const Eigen::Matrix3d> Rwg(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> g0(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> v0(parameters[2]);
        const double lamda1 = parameters[3][0];
        const double lamda2 = parameters[4][0];

        const Eigen::Vector3d Pbc = Tbc.translation();
        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
//        const Eigen::Vector3d DP = pInt.GetDeltaPosition(bg, gtba);
        const Eigen::Vector3d DP = pInt.GetDeltaPosition(bg, Eigen::Vector3d::Zero());
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
//        const Eigen::Vector3d g0 = Rwg*IMU::GRAVITY_VECTOR;

        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        e = - lamda1*u1 + DR*lamda2*u2 + v0*delt + 0.5*g0*delt*delt + DP + (DR - I)*Pbc;

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Vector3d Error2 = lamda2*u2 + Pbc;
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = - DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg + pInt.JPg;
            }
//            if (jacobians[1] != nullptr)
//            {
//                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> J(jacobians[1]);
//                J.setZero();
//                J.block<3, 3>(0, 6) = - 0.5*delt*delt*IMU::GRAVITY_MAGNITUDE*I;
//            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = 0.5*delt*delt*I;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[2]);
                J = delt*I;
            }
            if (jacobians[3] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[3]);
                J = - u1;
            }
            if (jacobians[4] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[4]);
                J = DR*u2;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u1;
    const Eigen::Vector3d u2;
    const IMU::Preintegrated pInt;
    const double delt;
    const Eigen::Vector3d gtba;
    const Eigen::Isometry3d Tbc;
};

// 3*Init::FrameNum*Init::PointNum  不拆开，合在一起，只优化bg，X作为参数传入
class CloseformCostFunction2 : public ceres::SizedCostFunction<600, 3> {
    public:
    CloseformCostFunction2(const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                          std::vector<IMU::Preintegrated> &pInts, const Eigen::VectorXd &X,
                          const double dt, const Eigen::Vector3d &gtba,
                          const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : AlignedPts(AlignedPts), pInts(pInts), X(X), dt(dt), gtba(gtba), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~CloseformCostFunction2() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);

        const Eigen::Matrix3d Rbc = Tbc.rotation();
        const Eigen::Vector3d Pbc = Tbc.translation();
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        const int PNum = 20;
        const int FrmNum = 10 + 1;
        Eigen::MatrixXd A(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
        Eigen::VectorXd S(3*(FrmNum - 1)*PNum);
        A.setZero();
        S.setZero();

        for(int i = 0; i < PNum; i++)
        {
            for(int j = 1; j < FrmNum; j++)
            {
                Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(bg);
                Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(bg, gtba);
                double delt = double(j)*dt;
                int colindex = 3*(i*(FrmNum - 1) + j - 1);
                A.block(colindex, 0, 3, 3) = 0.5*delt*delt*I;
                A.block(colindex, 3, 3, 3) = delt*I;
                A.block(colindex, i*FrmNum + 6, 3, 1) = - Rbc*AlignedPts[0][i];
                A.block(colindex, i*FrmNum + j + 6, 3, 1) = DR*Rbc*AlignedPts[j][i];
                S.segment(colindex, 3) = - DP - (DR - I)*Pbc;
            }
        }

        Eigen::Map<Eigen::Matrix<double, 3*(FrmNum - 1)*PNum, 1>> e(residuals);

        e = A*X - S;

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 3*(FrmNum - 1)*PNum, 3, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                for(int i = 0; i < PNum; i++)
                {
                    for(int j = 1; j < FrmNum; j++)
                    {
                        int colindex = 3*(i*(FrmNum - 1) + j - 1);
                        const Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(bg);
                        const Eigen::Vector3d dbg = pInts[j - 1].GetGyroDeltaBias(bg);
                        Eigen::Vector3d Error2 = Rbc*X[i*FrmNum + j + 6]*AlignedPts[j][i] + Pbc;
                        J.block(colindex, 0, 3, 3) = - DR*Skew(Error2)*RightJacobianSO3(pInts[j - 1].JRg*dbg)*pInts[j - 1].JRg + pInts[j - 1].JPg;
                    }
                }
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const std::vector<std::vector<Eigen::Vector3d>> AlignedPts;
    std::vector<IMU::Preintegrated> pInts;
    const Eigen::VectorXd X;
    const double dt;
    const Eigen::Vector3d gtba;
    const Eigen::Isometry3d Tbc;
};


// 3*Init::FrameNum*Init::PointNum   (Init::FrameNum + 1)*Init::PointNum
// 不拆开，合在一起优化所有参数
class CloseformCostFunction3 : public ceres::SizedCostFunction<600, 3, 3, 3, 210> {
    public:
    CloseformCostFunction3(const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                          std::vector<IMU::Preintegrated> &pInts,
                          const double dt, const Eigen::Vector3d &gtba,
                          const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : AlignedPts(AlignedPts), pInts(pInts), dt(dt), gtba(gtba), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~CloseformCostFunction3() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {

        const Eigen::Matrix3d Rbc = Tbc.rotation();
        const Eigen::Vector3d Pbc = Tbc.translation();
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        const int PNum = 10;
        const int FrmNum = 20 + 1;

        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
//        Eigen::Map<const Eigen::Matrix3d> Rwg(parameters[1]);
//        const Eigen::Vector3d g0 = Rwg*IMU::GRAVITY_VECTOR;
        Eigen::Map<const Eigen::Vector3d> g0(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> v0(parameters[2]);
        Eigen::Map<const Eigen::Matrix<double, PNum*FrmNum, 1>> lamda(parameters[3]);

        Eigen::Map<Eigen::Matrix<double, 3*(FrmNum - 1)*PNum, 1>> e(residuals);
        for(int i = 0; i < PNum; i++)
        {
            for(int j = 1; j < FrmNum; j++)
            {
                Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(bg);
                Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(bg, gtba);
                double delt = double(j)*dt;
                int colindex = 3*(i*(FrmNum - 1) + j - 1);
                e.segment(colindex, 3) = - lamda[i*FrmNum]*Rbc*AlignedPts[0][i]
                        + lamda[i*FrmNum + j]*DR*Rbc*AlignedPts[j][i]
                        + (DR - I)*Pbc + v0*delt + 0.5*g0*delt*delt + DP;
            }
        }

        if (jacobians != nullptr)
        {
//            if (jacobians[0] != nullptr)
//            {
                Eigen::Map<Eigen::Matrix<double, 3*(FrmNum - 1)*PNum, 3, Eigen::RowMajor>> J0(jacobians[0]);
                Eigen::Map<Eigen::Matrix<double, 3*(FrmNum - 1)*PNum, 3, Eigen::RowMajor>> J1(jacobians[1]);
//                Eigen::Map<Eigen::Matrix<double, 3*(FrmNum - 1)*PNum, 9, Eigen::RowMajor>> J1(jacobians[1]);
                Eigen::Map<Eigen::Matrix<double, 3*(FrmNum - 1)*PNum, 3, Eigen::RowMajor>> J2(jacobians[2]);
                Eigen::Map<Eigen::Matrix<double, 3*(FrmNum - 1)*PNum, FrmNum*PNum, Eigen::RowMajor>> J3(jacobians[3]);
                J0.setZero();
                J1.setZero();
                J2.setZero();
                J3.setZero();


                for(int i = 0; i < PNum; i++)
                {
                    for(int j = 1; j < FrmNum; j++)
                    {
                        double delt = double(j)*dt;
                        int colindex = 3*(i*(FrmNum - 1) + j - 1);
                        const Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(bg);
                        const Eigen::Vector3d dbg = pInts[j - 1].GetGyroDeltaBias(bg);
                        Eigen::Vector3d Error2 = lamda[i*FrmNum + j]*Rbc*AlignedPts[j][i] + Pbc;
                        J0.block(colindex, 0, 3, 3) = - DR*Skew(Error2)*RightJacobianSO3(pInts[j - 1].JRg*dbg)*pInts[j - 1].JRg + pInts[j - 1].JPg;
                        J1.block(colindex, 0, 3, 3) = 0.5*delt*delt*I;
//                        J1.block(colindex, 6, 3, 3) = - 0.5*delt*delt*IMU::GRAVITY_MAGNITUDE*I;
                        J2.block(colindex, 0, 3, 3) = delt*I;
                        J3.block(colindex, i*FrmNum, 3, 1) = - Rbc*AlignedPts[0][i];
                        J3.block(colindex, i*FrmNum + j, 3, 1) = DR*Rbc*AlignedPts[j][i];
                    }
                }
//            }
        }

        /*
        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 3*(FrmNum - 1)*PNum, 3, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                for(int i = 0; i < PNum; i++)
                {
                    for(int j = 1; j < FrmNum; j++)
                    {
                        int colindex = 3*(i*(FrmNum - 1) + j - 1);
                        const Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(bg);
                        const Eigen::Vector3d dbg = pInts[j - 1].GetGyroDeltaBias(bg);
                        Eigen::Vector3d Error2 = lamda[i*FrmNum + j]*Rbc*AlignedPts[j][i] + Pbc;
                        J.block(colindex, 0, 3, 3) = - DR*Skew(Error2)*RightJacobianSO3(pInts[j - 1].JRg*dbg)*pInts[j - 1].JRg + pInts[j - 1].JPg;
                    }
                }
            }
        }
        */
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const std::vector<std::vector<Eigen::Vector3d>> AlignedPts;
    std::vector<IMU::Preintegrated> pInts;
    const double dt;
    const Eigen::Vector3d gtba;
    const Eigen::Isometry3d Tbc;
};



class CloseformCostFunction_Onlybg : public ceres::SizedCostFunction<3, 3> {
    public:
    CloseformCostFunction_Onlybg(const Eigen::Vector3d &u1, const Eigen::Vector3d &u2,
                                 const IMU::Preintegrated &pInt, const double &delt,
                                 const Eigen::Vector3d &g, const Eigen::Vector3d &v0,
                                 const Eigen::Vector3d &gtba,
                                 const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : u1(u1), u2(u2), pInt(pInt), delt(delt), g(g), v0(v0), gtba(gtba), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~CloseformCostFunction_Onlybg() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);

        const Eigen::Matrix3d Rbc = Tbc.rotation();
        const Eigen::Vector3d Pbc = Tbc.translation();
        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
        const Eigen::Vector3d DP = pInt.GetDeltaPosition(bg, gtba);
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        e = - Rbc*u1 + DR*Rbc*u2 + v0*delt + 0.5*g*delt*delt + DP + (DR - I)*Pbc;

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Vector3d Error2 = Rbc*u2 + Pbc;
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = - DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg + pInt.JPg;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u1;
    const Eigen::Vector3d u2;
    const IMU::Preintegrated pInt;
    const double delt;
    const Eigen::Vector3d g;
    const Eigen::Vector3d v0;
    const Eigen::Vector3d gtba;
    const Eigen::Isometry3d Tbc;
};



class CloseformCostFunction4 : public ceres::SizedCostFunction<3, 3, 3, 3, 1, 1, 2, 2> {
    public:
    CloseformCostFunction4(const Eigen::Vector3d &u1, const Eigen::Vector3d &u2,
                          const IMU::Preintegrated &pInt, const double &delt,
                          const Eigen::Vector3d &gtba,
                          const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : u1(u1), u2(u2), pInt(pInt), delt(delt), gtba(gtba), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~CloseformCostFunction4() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
//        Eigen::Map<const Eigen::Matrix3d> Rwg(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> g0(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> v0(parameters[2]);
        const double lamda1 = parameters[3][0];
        const double lamda2 = parameters[4][0];
        Eigen::Map<const Eigen::Vector2d> duv1(parameters[5]);
        Eigen::Map<const Eigen::Vector2d> duv2(parameters[6]);

        const Eigen::Matrix3d Rbc = Tbc.rotation();
        const Eigen::Vector3d Pbc = Tbc.translation();
        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
        const Eigen::Vector3d DP = pInt.GetDeltaPosition(bg, gtba);
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
//        const Eigen::Vector3d g0 = Rwg*IMU::GRAVITY_VECTOR;

        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);

        Eigen::Vector3d _duv1, _duv2;
        _duv1.setZero();
        _duv2.setZero();
        _duv1.head(2) = duv1;
        _duv2.head(2) = duv2;
//        e = - Rbc*lamda1*u1 + DR*Rbc*lamda2*u2 + v0*delt + 0.5*g0*delt*delt + DP + (DR - I)*Pbc;
        e = - Rbc*lamda1*(u1 + _duv1) + DR*Rbc*lamda2*(u2 + _duv2) + v0*delt + 0.5*g0*delt*delt + DP + (DR - I)*Pbc;

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Vector3d Error2 = Rbc*lamda2*(u2 + _duv2) + Pbc;
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = - DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg + pInt.JPg;
            }
//            if (jacobians[1] != nullptr)
//            {
//                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> J(jacobians[1]);
//                J.setZero();
//                J.block<3, 3>(0, 6) = - 0.5*delt*delt*IMU::GRAVITY_MAGNITUDE*I;
//            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = 0.5*delt*delt*I;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[2]);
                J = delt*I;
            }
            if (jacobians[3] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[3]);
                J = - Rbc*(u1 + _duv1);
            }
            if (jacobians[4] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[4]);
                J = DR*Rbc*(u2 + _duv2);
            }
            if (jacobians[5] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> J(jacobians[5]);
                J = - lamda1*Rbc.block(0, 0, 3, 2);
            }
            if (jacobians[6] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> J(jacobians[6]);
                J = (lamda2*DR*Rbc).block(0, 0, 3, 2);
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u1;
    const Eigen::Vector3d u2;
    const IMU::Preintegrated pInt;
    const double delt;
    const Eigen::Vector3d gtba;
    const Eigen::Isometry3d Tbc;
};

class DuvCostFunction : public ceres::SizedCostFunction<2, 2> {
    public:
    DuvCostFunction(const double variance)
    : weight(std::sqrt(variance)) { }
    virtual ~DuvCostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector2d> duv(parameters[0]);

        Eigen::Map<Eigen::Vector2d> error(residuals);
        error = weight*duv;

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> J(jacobians[0]);
                J = weight*Eigen::Matrix2d::Identity();
            }
        }

        return true;
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
    const double weight;
};



class LineCostFunctionNew : public ceres::SizedCostFunction<3, 3, 1, 1, 1> {
    public:
    LineCostFunctionNew(const Eigen::Vector3d &p1, const Eigen::Vector3d &n1,
                        const Eigen::Vector3d &p2, const Eigen::Vector3d &n2,
                        const IMU::Preintegrated &pInt,
                        const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : p1(p1), n1(n1), p2(p2), n2(n2), pInt(pInt), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~LineCostFunctionNew() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        const double lamda1 = parameters[1][0];
        const double lamda2 = parameters[2][0];
        const double lamda3 = parameters[3][0];

        Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
        Eigen::Matrix3d Rbc = Tbc.rotation();

        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        e = Rbc*(p1 + lamda1*n1) - DR*Rbc*(lamda2*p2 + lamda3*n2);

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Vector3d Error2 = - Rbc*(lamda2*p2 + lamda3*n2);
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = - DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg;
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[1]);
                J = Rbc*n1;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[2]);
                J = - DR*Rbc*p2;
            }
            if (jacobians[3] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[3]);
                J = - DR*Rbc*n2;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d p1;
    const Eigen::Vector3d n1;
    const Eigen::Vector3d p2;
    const Eigen::Vector3d n2;
    const IMU::Preintegrated pInt;
    const Eigen::Isometry3d Tbc;
};



class LineCostFunction : public ceres::SizedCostFunction<3, 3, 1, 1> {
    public:
    LineCostFunction(const Eigen::Vector3d &p1, const Eigen::Vector3d &n1,
                     const Eigen::Vector3d &p2, const Eigen::Vector3d &n2,
                     const IMU::Preintegrated &pInt,
                     const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : p1(p1), n1(n1), p2(p2), n2(n2), pInt(pInt), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~LineCostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        const double theta1 = parameters[1][0];
        const double theta_n = parameters[2][0];

        Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
        Eigen::Matrix3d Rbc = Tbc.rotation();
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        Eigen::Vector3d Error1 = Rbc*(cos(theta1)*I + sin(theta1)*Skew(n1))*p1;
        Eigen::Vector3d Error2 = Rbc*(cos(theta_n)*I + sin(theta_n)*Skew(n2))*p2;
        e = Error1 - DR*Error2;

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg;
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[1]);
                J = Rbc*( - sin(theta1)*I + cos(theta1)*Skew(n1))*p1;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[2]);
                J = - DR*Rbc*( - sin(theta_n)*I + cos(theta_n)*Skew(n2))*p2;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d p1;
    const Eigen::Vector3d n1;
    const Eigen::Vector3d p2;
    const Eigen::Vector3d n2;
    const IMU::Preintegrated pInt;
    const Eigen::Isometry3d Tbc;
};

class LineCostFunction2 : public ceres::SizedCostFunction<3, 3, 1, 1, 1> {
    public:
    LineCostFunction2(const Eigen::Vector3d &p1, const Eigen::Vector3d &n1,
                     const Eigen::Vector3d &p2, const Eigen::Vector3d &n2,
                     const IMU::Preintegrated &pInt,
                     const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : p1(p1), n1(n1), p2(p2), n2(n2), pInt(pInt), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~LineCostFunction2() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        const double lamda0 = parameters[1][0];
        const double lamda1 = parameters[2][0];
        const double lamda2 = parameters[3][0];

        Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
        Eigen::Matrix3d Rbc = Tbc.rotation();
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        e = Rbc*(I + lamda0*Skew(n1))*p1 - DR*Rbc*(lamda1*I + lamda2*Skew(n2))*p2;

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Vector3d Error2 = Rbc*(lamda1*I + lamda2*Skew(n2))*p2;
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg;
//                J = DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg);
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[1]);
                J = Rbc*Skew(n1)*p1;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[2]);
                J = - DR*Rbc*p2;
            }
            if (jacobians[3] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[3]);
                J = - DR*Rbc*Skew(n2)*p2;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d p1;
    const Eigen::Vector3d n1;
    const Eigen::Vector3d p2;
    const Eigen::Vector3d n2;
    const IMU::Preintegrated pInt;
    const Eigen::Isometry3d Tbc;
};

class LineCostFunction3 : public ceres::SizedCostFunction<3, 3, 1, 1, 1, 1> {
    public:
    LineCostFunction3(const Eigen::Vector3d &u11, const Eigen::Vector3d &u12,
                     const Eigen::Vector3d &u21, const Eigen::Vector3d &u22,
                     const IMU::Preintegrated &pInt,
                     const Eigen::Matrix3d &Rbc = Eigen::Matrix3d::Identity())
        : u11(u11), u12(u12), u21(u21), u22(u22), pInt(pInt), Rbc(Rbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~LineCostFunction3() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        const double lamda11 = parameters[1][0];
        const double lamda12 = parameters[2][0];
        const double lamda21 = parameters[3][0];
        const double lamda22 = parameters[4][0];

        Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        e = Rbc*(lamda11*u11 - lamda12*u12) - DR*Rbc*(lamda21*u21 - lamda22*u22);

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Vector3d Error2 = Rbc*(lamda21*u21 - lamda22*u22);
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg;
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[1]);
                J = Rbc*u11;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[2]);
                J = - Rbc*u12;
            }
            if (jacobians[3] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[3]);
                J = - DR*Rbc*u21;
            }
            if (jacobians[4] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[4]);
                J = DR*Rbc*u22;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u11;
    const Eigen::Vector3d u12;
    const Eigen::Vector3d u21;
    const Eigen::Vector3d u22;
    const IMU::Preintegrated pInt;
    const Eigen::Matrix3d Rbc;
};

class LineCostFunction4 : public ceres::SizedCostFunction<3, 3, 1, 1, 1> {
    public:
    LineCostFunction4(const Eigen::Vector3d &u11, const Eigen::Vector3d &u12,
                     const Eigen::Vector3d &u21, const Eigen::Vector3d &u22,
                     const IMU::Preintegrated &pInt,
                     const Eigen::Matrix3d &Rbc = Eigen::Matrix3d::Identity())
        : u11(u11), u12(u12), u21(u21), u22(u22), pInt(pInt), Rbc(Rbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~LineCostFunction4() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        const double lamda12 = parameters[1][0];
        const double lamda21 = parameters[2][0];
        const double lamda22 = parameters[3][0];

        Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        e = Rbc*(u11 - lamda12*u12) - DR*Rbc*(lamda21*u21 - lamda22*u22);

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Vector3d Error2 = Rbc*(lamda21*u21 - lamda22*u22);
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg;
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[1]);
                J = - Rbc*u12;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[2]);
                J = - DR*Rbc*u21;
            }
            if (jacobians[3] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[3]);
                J = DR*Rbc*u22;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u11;
    const Eigen::Vector3d u12;
    const Eigen::Vector3d u21;
    const Eigen::Vector3d u22;
    const IMU::Preintegrated pInt;
    const Eigen::Matrix3d Rbc;
};


class LineCostFunction5 : public ceres::SizedCostFunction<3, 3> {
    public:
    LineCostFunction5(const Eigen::Vector3d &u11, const Eigen::Vector3d &u12,
                     const Eigen::Vector3d &u21, const Eigen::Vector3d &u22,
                     const IMU::Preintegrated &pInt,
                     const Eigen::Matrix3d &Rbc = Eigen::Matrix3d::Identity())
        : u11(u11), u12(u12), u21(u21), u22(u22), pInt(pInt), Rbc(Rbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~LineCostFunction5() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);

        Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        e = Rbc*(u11 - u12) - DR*Rbc*(u21 - u22);

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Vector3d Error2 = Rbc*(u21 - u22);
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u11;
    const Eigen::Vector3d u12;
    const Eigen::Vector3d u21;
    const Eigen::Vector3d u22;
    const IMU::Preintegrated pInt;
    const Eigen::Matrix3d Rbc;
};



class ReprojectionCostFunction : public ceres::SizedCostFunction<2, 3, 3, 1>
{
    public:
    ReprojectionCostFunction(const Eigen::Vector3d &u1, const Eigen::Vector3d &u2,
                             const IMU::Preintegrated &pInt,
                             const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : u1(u1), u2(u2), pInt(pInt), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~ReprojectionCostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t12(parameters[1]);
        const double lamda = parameters[2][0];

        const Eigen::Matrix3d Rbc = Tbc.rotation();
        const Eigen::Vector3d tbc = Tbc.translation();
        const Eigen::Matrix3d Rcb = Rbc.transpose();
        const Eigen::Vector3d tcb = - Rcb*tbc;
        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);

        Eigen::Vector3d fc;
        fc = 1.0f/lamda*Rcb*DR*Rbc*u2 + Rcb*DR*tbc + Rcb*t12 + tcb;
//        fc = 1.0f/lamda*Rcb*DR*Rbc*u2 + Rcb*t12;

        Eigen::Map<Eigen::Matrix<double, 2, 1>> e(residuals);
        e[0] = fc[0]/fc[2] - u1[0]/u1[2];
        e[1] = fc[1]/fc[2] - u1[1]/u1[2];

        if (jacobians != nullptr)
        {
            Eigen::Matrix<double, 2, 3> Jrfc;
            Jrfc << 1.0f/fc[2], 0, - fc[0]/(fc[2]*fc[2]),
                    0, 1.0f/fc[2], - fc[1]/(fc[2]*fc[2]);
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                const Eigen::Vector3d Error2 = Rbc*u2/lamda + tbc;
//                const Eigen::Vector3d Error2 = Rbc*u2/lamda;
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = - Jrfc*Rcb*DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg;
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = Jrfc*Rcb;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[2]);
                J = - Jrfc*Rcb*DR*Rbc*u2/(lamda*lamda);
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u1;
    const Eigen::Vector3d u2;
    const IMU::Preintegrated pInt;
    const Eigen::Isometry3d Tbc;
};


class ReprojectionCostFunctionR21 : public ceres::SizedCostFunction<2, 3, 3, 1>
{
    public:
    ReprojectionCostFunctionR21(const Eigen::Vector3d &u1, const Eigen::Vector3d &u2,
                             const IMU::Preintegrated &pInt, const double &delt,
                             const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : u1(u1), u2(u2), pInt(pInt), delt(delt), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~ReprojectionCostFunctionR21() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t21(parameters[1]);
        const double lamda = parameters[2][0];

        const Eigen::Matrix3d Rbc = Tbc.rotation();
        const Eigen::Vector3d tbc = Tbc.translation();
        const Eigen::Matrix3d Rcb = Rbc.transpose();
        const Eigen::Vector3d tcb = - Rcb*tbc;
        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
//        const Eigen::Vector3d DP = pInt.GetDeltaPosition(bg, Eigen::Vector3d::Zero());
        const Eigen::Matrix3d DRT = DR.transpose();

        Eigen::Vector3d fc;
//        const Eigen::Vector3d Error2 = Rbc*u1/lamda + tbc - v0*delt - 0.5*g0*delt*delt;
//        fc = Rcb*DRT*(Error2 - DP) + tcb;
//        fc = Rcb*DRT*(Rbc*u1/lamda + tbc - v0*delt - 0.5*g0*delt*delt) + tcb;

        fc = Rcb*DRT*Rbc*u1/lamda + Rcb*DRT*tbc + Rcb*t21 + tcb;

        Eigen::Map<Eigen::Matrix<double, 2, 1>> e(residuals);
        e[0] = fc[0]/fc[2] - u2[0]/u2[2];
        e[1] = fc[1]/fc[2] - u2[1]/u2[2];

        if (jacobians != nullptr)
        {
            Eigen::Matrix<double, 2, 3> Jrfc;
            Jrfc << 1.0f/fc[2], 0, - fc[0]/(fc[2]*fc[2]),
                    0, 1.0f/fc[2], - fc[1]/(fc[2]*fc[2]);
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                const Eigen::Matrix3d JRTg = - DR*pInt.JRg;
                const Eigen::Vector3d Error2 = Rbc*u1/lamda + tbc;
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = - Jrfc*Rcb*DRT*Skew(Error2)*RightJacobianSO3(JRTg*dbg)*JRTg;
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = Jrfc*Rcb;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[2]);
                J = - Jrfc*Rcb*DRT*Rbc*u1/(lamda*lamda);
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u1;
    const Eigen::Vector3d u2;
    const IMU::Preintegrated pInt;
    const double delt;
    const Eigen::Isometry3d Tbc;
};



class ReprojectionCostFunction2 : public ceres::SizedCostFunction<2, 3, 3, 3, 1>
{
    public:
    ReprojectionCostFunction2(const Eigen::Vector3d &u1, const Eigen::Vector3d &u2,
                              const IMU::Preintegrated &pInt, const Eigen::Vector3d &gtba,
                              double &dt,
                              const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : u1(u1), u2(u2), pInt(pInt), gtba(gtba), dt(dt), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~ReprojectionCostFunction2() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> v0(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> g0(parameters[2]);
        const double lamda = parameters[3][0];

        const Eigen::Matrix3d Rbc = Tbc.rotation();
        const Eigen::Vector3d tbc = Tbc.translation();
        const Eigen::Matrix3d Rcb = Rbc.transpose();
        const Eigen::Vector3d tcb = - Rcb*tbc;
        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
        const Eigen::Vector3d DV = pInt.GetDeltaVelocity(bg, Eigen::Vector3d::Zero());
        const Eigen::Vector3d DP = pInt.GetDeltaPosition(bg, Eigen::Vector3d::Zero());

        Eigen::Vector3d fc;
        fc = Rcb*DR*Rbc*u2/lamda + Rcb*DR*tbc + Rcb*(DR*v0*dt - 0.5*DR*g0*dt*dt + DP - DV*dt) + tcb;

        Eigen::Map<Eigen::Matrix<double, 2, 1>> e(residuals);
        e[0] = fc[0]/fc[2] - u1[0]/u1[2];
        e[1] = fc[1]/fc[2] - u1[1]/u1[2];
        if (jacobians != nullptr)
        {
            Eigen::Matrix<double, 2, 3> Jrfc;
            Jrfc << 1.0f/fc[2], 0, - fc[0]/(fc[2]*fc[2]),
                    0, 1.0f/fc[2], - fc[1]/(fc[2]*fc[2]);
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                const Eigen::Vector3d Error2 = Rbc*u2/lamda + tbc + v0*dt - 0.5*g0*dt*dt;
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = Jrfc*( - Rcb*DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg + Rcb*pInt.JPg - Rcb*pInt.JVg*dt);
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = Jrfc*Rcb*DR*dt;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[2]);
                J = - Jrfc*Rcb*0.5*DR*dt*dt;
            }
            if (jacobians[3] != nullptr)
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[3]);
                J = - Jrfc*Rcb*DR*Rbc*u2/(lamda*lamda);
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u1;
    const Eigen::Vector3d u2;
    const IMU::Preintegrated pInt;
    const Eigen::Vector3d gtba;
    const double dt;
    const Eigen::Isometry3d Tbc;
};





class ReprojectionCostFunctionLines : public ceres::SizedCostFunction<2, 3, 3, 1, 1>
{
    public:
    ReprojectionCostFunctionLines(const Eigen::Vector3d &start1, const Eigen::Vector3d &end1,
                                  const Eigen::Vector3d &start2, const Eigen::Vector3d &end2,
                                  const IMU::Preintegrated &pInt,
                                  const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : start1(start1), end1(end1), start2(start2), end2(end2), pInt(pInt), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~ReprojectionCostFunctionLines() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t12(parameters[1]);
        const double lamda1 = parameters[2][0];
        const double lamda2 = parameters[3][0];

        const Eigen::Matrix3d Rbc = Tbc.rotation();
        const Eigen::Vector3d tbc = Tbc.translation();
        const Eigen::Matrix3d Rcb = Rbc.transpose();
        const Eigen::Vector3d tcb = - Rcb*tbc;
        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);

        const Eigen::Vector3d n2 = Skew(start2)*end2/(lamda1*lamda2);
        const Eigen::Vector3d d2 = end2/lamda2 - start2/lamda1;

        Eigen::Vector3d n1;
        n1 = Rcb*DR*Rbc*n2 + Rcb*(Skew(t12)*DR + DR*Skew(tbc) - Skew(tbc)*DR)*Rbc*d2;

        const double dot1 = start1.dot(n1);
        const double dot2 = end1.dot(n1);
        const double norm1_2 = 1.0f / n1.head(2).norm();
        const double norm3_2 = 1.0f / (n1.head(2).dot(n1.head(2))*n1.head(2).norm());

        Eigen::Map<Eigen::Matrix<double, 2, 1>> e(residuals);
        e[0] = dot1*norm1_2;
        e[1] = dot2*norm1_2;

        if (jacobians != nullptr)
        {
            Eigen::Matrix<double, 2, 3> Jrn1;
            Jrn1 << start1[0]*norm1_2 - n1[0]*dot1*norm3_2,
                    start1[1]*norm1_2 - n1[1]*dot1*norm3_2,
                    norm1_2,
                    end1[0]*norm1_2 - n1[0]*dot2*norm3_2,
                    end1[1]*norm1_2 - n1[1]*dot2*norm3_2,
                    norm1_2;
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                const Eigen::Vector3d Error2 = Rbc*n2 + Skew(tbc)*Rbc*d2;
                const Eigen::Vector3d Error3 = Rbc*d2;
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix3d Jn1bg =
                - Rcb*DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg
                - Rcb*(Skew(t12) - Skew(tbc))*DR*Skew(Error3)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg;
                J = Jrn1*Jn1bg;
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                Eigen::Matrix3d Jn1t12 = - Rcb*Skew(DR*Rbc*d2);
                J = Jrn1*Jn1t12;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[2]);
                Eigen::Vector3d Jn1lamda1 = - Rcb*DR*Rbc*n2/lamda1
                + Rcb*(Skew(t12)*DR + DR*Skew(tbc) - Skew(tbc)*DR)*Rbc*start2/(lamda1*lamda1);
                J = Jrn1*Jn1lamda1;
            }
            if (jacobians[3] != nullptr)
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[3]);
                Eigen::Vector3d Jn1lamda2 = - Rcb*DR*Rbc*n2/lamda2
                - Rcb*(Skew(t12)*DR + DR*Skew(tbc) - Skew(tbc)*DR)*Rbc*end2/(lamda2*lamda2);
                J = Jrn1*Jn1lamda2;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d start1;
    const Eigen::Vector3d end1;
    const Eigen::Vector3d start2;
    const Eigen::Vector3d end2;
    const IMU::Preintegrated pInt;
    const Eigen::Isometry3d Tbc;
};









class Reprojection_3D_CostFunction : public ceres::SizedCostFunction<2, 3, 3, 3>
{
    public:
    Reprojection_3D_CostFunction(const Eigen::Vector3d &u1,
                             const IMU::Preintegrated &pInt,
                             const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : u1(u1), pInt(pInt), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~Reprojection_3D_CostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t12(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> pt(parameters[2]);

        const Eigen::Matrix3d Rbc = Tbc.rotation();
        const Eigen::Vector3d tbc = Tbc.translation();
        const Eigen::Matrix3d Rcb = Rbc.transpose();
        const Eigen::Vector3d tcb = - Rcb*tbc;
        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);

        Eigen::Vector3d fc;
        fc = Rcb*DR*Rbc*pt + Rcb*DR*tbc + Rcb*t12 + tcb;

        Eigen::Map<Eigen::Matrix<double, 2, 1>> e(residuals);
        e[0] = fc[0]/fc[2] - u1[0]/u1[2];
        e[1] = fc[1]/fc[2] - u1[1]/u1[2];

        if (jacobians != nullptr)
        {
            Eigen::Matrix<double, 2, 3> Jrfc;
            Jrfc << 1.0f/fc[2], 0, - fc[0]/(fc[2]*fc[2]),
                    0, 1.0f/fc[2], - fc[1]/(fc[2]*fc[2]);
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                const Eigen::Vector3d Error2 = Rbc*pt + tbc;
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = - Jrfc*Rcb*DR*Skew(Error2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg;
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = Jrfc*Rcb;
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[2]);
                J = Jrfc*Rcb*DR*Rbc;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u1;
    const IMU::Preintegrated pInt;
    const Eigen::Isometry3d Tbc;
};

class Reprojection_3D_Self_CostFunction : public ceres::SizedCostFunction<2, 3>
{
    public:
    Reprojection_3D_Self_CostFunction(const Eigen::Vector3d &u2): u2(u2)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~Reprojection_3D_Self_CostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> pt(parameters[0]);

        Eigen::Map<Eigen::Matrix<double, 2, 1>> e(residuals);
        e[0] = pt[0]/pt[2] - u2[0];
        e[1] = pt[1]/pt[2] - u2[1];

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
                J << 1.0f/pt[2], 0, - pt[0]/(pt[2]*pt[2]),
                     0, 1.0f/pt[2], - pt[1]/(pt[2]*pt[2]);
            }
        }
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d u2;
    const Eigen::Isometry3d Tbc;
};



class CloseformCostFunctionLinesBeta : public ceres::SizedCostFunction<3, 3, 1>
{
    public:
    CloseformCostFunctionLinesBeta(const Eigen::Vector3d &p1, const Eigen::Vector3d &q1,
                          const IMU::Preintegrated &pInt, const Eigen::Matrix3d &M)
        : p1(p1), q1(q1), pInt(pInt), M(M) // 这里Rbc已经之前乘在了p1 q1上就不需要再乘了
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~CloseformCostFunctionLinesBeta() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        const double beta = parameters[1][0];

        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
        const Eigen::Matrix3d DRT = DR.transpose();

        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        e = M*DRT*(beta*q1 - p1);

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                const Eigen::Vector3d Error2 = beta*q1 - p1;
                const Eigen::Matrix3d JRTg = - DR*pInt.JRg;
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = - M*DRT*Skew(Error2)*RightJacobianSO3(JRTg*dbg)*JRTg;
            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[1]);
                J = M*DRT*q1;
            }
        }
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d p1;
    const Eigen::Vector3d q1;
    const IMU::Preintegrated pInt;
    const Eigen::Matrix3d M;
};



class CloseformCostFunctionLines : public ceres::SizedCostFunction<3, 3, 3, 3, 1, 1, 1>
{
    public:
    CloseformCostFunctionLines(const Eigen::Vector3d &n1, const Eigen::Vector3d &n2,
                               const Eigen::Vector3d &p1, const Eigen::Vector3d &q1,
                               const IMU::Preintegrated &pInt, const double &delt,
                               const Eigen::Vector3d &gtba,
                               const Eigen::Isometry3d &Tbc = Eigen::Isometry3d::Identity())
        : n1(n1), n2(n2), p1(p1), q1(q1), pInt(pInt), delt(delt), gtba(gtba), Tbc(Tbc)
    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Eigen::Matrix3d::Identity());
//        SqrtInformation = solver.operatorInverseSqrt();
    }
    virtual ~CloseformCostFunctionLines() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
//        Eigen::Map<const Eigen::Matrix3d> Rwg(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> g0(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> v0(parameters[2]);
        const double beta = parameters[3][0];
        const double lamda1 = parameters[4][0];
        const double lamda2 = parameters[5][0];

        const Eigen::Vector3d Pbc = Tbc.translation();
        const Eigen::Matrix3d DR = pInt.GetDeltaRotation(bg);
//        const Eigen::Vector3d DP = pInt.GetDeltaPosition(bg, gtba);
        const Eigen::Vector3d DP = pInt.GetDeltaPosition(bg, Eigen::Vector3d::Zero());
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
//        const Eigen::Vector3d g0 = Rwg*IMU::GRAVITY_VECTOR;

        const Eigen::Vector3d d = p1 + beta*q1;
        const Eigen::Vector3d err = ((DR - I)*Pbc + v0*delt + 0.5*g0*delt*delt + DP);

        Eigen::Map<Eigen::Matrix<double, 3, 1>> e(residuals);
        e = lamda1*n1 - lamda2*DR*n2 + Skew(d)*err;

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = pInt.GetGyroDeltaBias(bg);
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = lamda2*DR*Skew(n2)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg
                        + Skew(d)*(- DR*Skew(Pbc)*RightJacobianSO3(pInt.JRg*dbg)*pInt.JRg + pInt.JPg);
            }
//            if (jacobians[1] != nullptr)
//            {
//                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> J(jacobians[1]);
//                J.setZero();
//                J.block<3, 3>(0, 6) = - 0.5*delt*delt*IMU::GRAVITY_MAGNITUDE*I;
//            }
            if (jacobians[1] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = 0.5*delt*delt*Skew(d);
            }
            if (jacobians[2] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[2]);
                J = delt*Skew(d);
            }
            if (jacobians[3] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[3]);
                J = Skew(q1)*err;
            }
            if (jacobians[4] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[4]);
                J = n1;
            }
            if (jacobians[5] != nullptr)
            {
                Eigen::Map<Eigen::Vector3d> J(jacobians[5]);
                J = - DR*n2;
            }
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
    const Eigen::Vector3d n1;
    const Eigen::Vector3d n2;
    const Eigen::Vector3d p1;
    const Eigen::Vector3d q1;
    const IMU::Preintegrated pInt;
    const double delt;
    const Eigen::Vector3d gtba;
    const Eigen::Isometry3d Tbc;
};



#endif // IMU_CERES_H
