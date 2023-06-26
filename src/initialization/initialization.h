
#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

// STL
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include <string>
#include <queue>
#include <random>
// Ceres
#include <ceres/ceres.h>
#include <vector>
//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// Glog
#include <glog/logging.h>

#include "imu_ceres.h"
#include "imu_preintegration.h"
#include "polynomial.h"
#include "interpolation.h"
#include "so3.h"
#include "util/io.h"
#include "util/timer.h"

namespace Init {

const double CamRate = 20;// 一般取相机频率为20Hz
const int interframes = 10;  // 这个为imu频率/相机频率hz
const double InitRate = 10;// 初始化频率4——10Hz

const int PointNum = 10;  // 闭式求解需要的特征点数
const int FrameNum = 20;  // 闭式求解需要的帧数，这个是不含起点的帧数，即实际总帧数为FrameNum+1
const int LineNum = 7;  // 闭式求解需要的线段数


extern double StaTime;
extern double StaErrorv;
extern double StaError;
extern double StaCount;
extern double ErrorPercen;
extern double gravity_error;
extern std::vector<int> statistical;

using namespace io;

struct Input
{
    Input(const Eigen::Isometry3d &T1, const std::uint64_t t1,
          const Eigen::Isometry3d &T2, const std::uint64_t t2,
          std::shared_ptr<IMU::Preintegrated> pInt)
        : t1(t1), t2(t2), T1(T1), T2(T2), pInt(pInt) { }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const std::uint64_t t1, t2;
    Eigen::Isometry3d T1, T2;
    std::shared_ptr<IMU::Preintegrated> pInt;

    std::vector<IMU::Measurement> vMeasurements;
};

struct Result
{
    Result(): success(false) { }

    Result(bool success, std::int64_t solve_ns, double scale,
           const Eigen::Vector3d &bias_g, const Eigen::Vector3d &bias_a,
           const Eigen::Vector3d &gravity)
        : success(success), solve_ns(solve_ns), scale(scale),
          bias_g(bias_g), bias_a(bias_a), gravity(gravity) { }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool success;
    std::int64_t solve_ns;
    double scale;
    Eigen::Vector3d bias_g, bias_a, gravity;
};

struct comp
{
    bool operator()(std::pair<double, double> &a, std::pair<double, double> &b)
    {
        return (a.second > b.second);//大于号是小顶堆 小于号是大顶堆
    }
};


double Temporal_Calibration(const std::vector<Eigen::Matrix3d> &Rot,
                          const std::vector<Eigen::Vector3d> &imu_w);

void Init_ORBSLAM3_Method(const std::vector<Input> &input, Result &result, double &cost, double init_scale,
                          const Eigen::Isometry3d &Tcb = Eigen::Isometry3d::Identity(),
                          bool use_prior = true, double prior = 1e5);

void Init_Analytical_Method(const std::vector<Input> &input, Result &result,
                            const Eigen::Isometry3d &Tcb = Eigen::Isometry3d::Identity());

void Init_Interpolation_Method(const std::vector<Input> &input, Result &result,
                               const std::vector<Eigen::Vector3d> &imu_w,
                               const std::vector<Eigen::Vector3d> &imu_a,
                               const std::vector<Eigen::Isometry3d> &GTPose,
                               const std::vector<Eigen::Vector3d> &GTVel,
                               const Eigen::Isometry3d &Tcb = Eigen::Isometry3d::Identity());

void CloseForm_Solution(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg,
                        const std::vector<IMU::Preintegrated> &pIntsgt);

void CloseForm_Solution2(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg,
                        const std::vector<IMU::Preintegrated> &pIntsgt);

void CloseForm_Solution3(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg,
                        const std::vector<IMU::Preintegrated> &pIntsgt);

void CloseForm_Solution4(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg,
                        const std::vector<IMU::Preintegrated> &pIntsgt);

void Reprojection_Solution(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg, std::vector<Eigen::Vector3d> &gttrans);

void Reprojection_SolutionR21(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg, std::vector<Eigen::Vector3d> &gttrans);

void Reprojection_PAL_Solution(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                               const std::vector<std::vector<Eigen::Vector6d>> &AlignedLns,
                               std::vector<IMU::Preintegrated> &pInts,
                               const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                               const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                               const Eigen::Vector3d &gtg);


void Reprojection_Closeform_Solution(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg);


void Reprojection_3D_Solution(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg, std::vector<Eigen::Vector3d> &gttrans);


void CloseForm_Solution_PAL(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                            const std::vector<std::vector<Eigen::Vector6d>> &AlignedLns,
                            std::vector<IMU::Preintegrated> &pInts,
                            const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                            const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                            const Eigen::Vector3d &gtg,
                            const std::vector<IMU::Preintegrated> &pIntsgt);


}
#endif
