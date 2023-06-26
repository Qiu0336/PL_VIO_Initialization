// STL
#include <algorithm>
#include <cmath>

#include <iterator>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>

#include <unistd.h>
// Boost
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <opencv2/opencv.hpp>

//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>


#include "util/csv.h"
#include "util/io.h"

#include "initialization.h"
#include "imu_preintegration.h"

#include "interpolation.h"
#include "sfm.h"

namespace fs = boost::filesystem;

using namespace io;

Eigen::Isometry3d Tcb;
Eigen::Isometry3d Tbc;

struct evaluation_t {
  evaluation_t(const timestamp_t solve_time, const timestamp_t initialization_time,
               const timestamp_t timestamp, const double scale_error,
               const double gyro_bias_error, const double acc_bias_error, double gravity_error)
    : solve_time(solve_time), initialization_time(initialization_time),
      timestamp(timestamp), scale_error(scale_error),
      gyro_bias_error(gyro_bias_error), acc_bias_error(acc_bias_error), gravity_error(gravity_error)
  { }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  timestamp_t solve_time; // nanoseconds
  timestamp_t initialization_time; // nanoseconds
  timestamp_t timestamp; // nanoseconds
  double scale_error; // percent
  double gyro_bias_error; // percent
  double acc_bias_error; // percent
  double gravity_error; // degrees

};

void save(const std::vector<evaluation_t> &data, const std::string &save_path) {
  Eigen::MatrixXd m(data.size(), 7);

  for (unsigned i = 0; i < data.size(); ++i) {
    Eigen::RowVectorXd row(7);
    row << data[i].solve_time, data[i].initialization_time, data[i].timestamp,
           data[i].scale_error, data[i].gyro_bias_error, data[i].acc_bias_error, data[i].gravity_error;
    m.row(i) = row;
  }

  csv::write(m, save_path);
}


Eigen::Isometry3d Compute_Scale(const std::vector<Init::Input> &input, const State &groundtruth, double &scale_factor)
{
    Trajectory InputTraj;
    InputTraj.emplace_back(input.front().t1, input.front().T1);
    for (const Init::Input &data : input)
        InputTraj.emplace_back(data.t2, data.T2);

    bool simul = false;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> pairs;
    for (const auto it : InputTraj)
    {
        State::const_iterator gt = Find_Iterator<State::value_type, State::const_iterator>
                (groundtruth.cbegin(), groundtruth.cend(), it.timestamp, simul);
        if (!simul) continue;

        Eigen::Vector3d pos1(it.pose.tx, it.pose.ty, it.pose.tz);
        Eigen::Vector3d pos2(gt->pose.tx, gt->pose.ty, gt->pose.tz);

        // 匀速插值，加上与不加结果基本没变。。
//        if (gt->timestamp > it.timestamp && gt != groundtruth.cbegin())
//        {
//            State::const_iterator gt_adj = std::prev(gt);
//            Eigen::Vector3d pos_adj(gt_adj->pose.tx, gt_adj->pose.ty, gt_adj->pose.tz);
//            double dt1, dt2;
//            dt1 = (gt->timestamp - gt_adj->timestamp)*1e-9;
//            dt2 = (gt->timestamp - it.timestamp)*1e-9;
//            pos2 -= (pos2 - pos_adj)*dt2/dt1;
//        }
//        else if (gt->timestamp < it.timestamp && gt != groundtruth.cend())
//        {
//            State::const_iterator gt_adj = std::next(gt);
//            Eigen::Vector3d pos_adj(gt_adj->pose.tx, gt_adj->pose.ty, gt_adj->pose.tz);
//            double dt1, dt2;
//            dt1 = (gt_adj->timestamp - gt->timestamp)*1e-9;
//            dt2 = (it.timestamp - gt->timestamp)*1e-9;
//            pos2 += (pos_adj - pos2)*dt2/dt1;
//        }

        pairs.emplace_back(pos1, pos2);
    }

    const int N = pairs.size();
    CHECK_GE(N, 3) << "At least 3 poses are required!";

    Eigen::MatrixXd src(3, N);
    Eigen::MatrixXd dst(3, N);

    int index = 0;
    for (const auto match : pairs)
    {
        src.col(index) = match.first;
        dst.col(index) = match.second;
        index++;
    }

    Eigen::Matrix4d M = Eigen::umeyama(src, dst, true);

    scale_factor = std::cbrt(M.block<3, 3>(0, 0).determinant());// 三次方根求scale

    Eigen::Isometry3d T;
    T.linear() = M.block<3, 3>(0, 0)/scale_factor;// 求变换矩阵
    T.translation() = M.block<3, 1>(0, 3);
    return T;
}


void Test_All_Methods(const fs::path &sequence_path)
{
    fs::path groundtruth_path = sequence_path / "state_groundtruth_estimate0" / "data.csv";
    CHECK(fs::is_regular_file(groundtruth_path)) << "Path not found: " << groundtruth_path.string();

    fs::path imu_path = sequence_path / "imu0" / "data.csv";
    CHECK(fs::is_regular_file(imu_path)) << "Path not found: " << imu_path.string();

    fs::path trajectory_path = "/home/qjy/Research/VISLAM/Trajectorys/MH_01.txt";
    CHECK(fs::is_regular_file(trajectory_path)) << "Path not found: " << trajectory_path.string();

    State groundtruth = Read_File<State::value_type>(groundtruth_path.string());
    ImuData imu_data = Read_File<ImuData::value_type>(imu_path.string());
    Trajectory trajectory = Read_File<Trajectory::value_type>(trajectory_path.string());

    std::vector<unsigned> possible_nframes = {20};

    bool simul = false;

    for (unsigned nframes : possible_nframes)
    {
        Trajectory::const_iterator traj = trajectory.cbegin();
        State::const_iterator gt = groundtruth.cbegin();
        if (traj->timestamp > gt->timestamp)
        {
            gt = Find_Iterator<State::value_type, State::const_iterator>
                    (gt, groundtruth.cend(), traj->timestamp, simul);
        }
        else
        {
            traj = Find_Iterator<Trajectory::value_type, Trajectory::const_iterator>
                    (traj, trajectory.cend(), gt->timestamp, simul);
        }
        if (!simul)
        {
            LOG(WARNING) << "Couldn't find correspondence timestamps !";
            break;
        }

        std::vector<evaluation_t> proposed_evaluation;// 提出的方法
        std::vector<evaluation_t> iterative_evaluation;// 迭代优化的方法

        Trajectory::const_iterator traj_copy = traj;
        while (traj != trajectory.cend())
        {

            LOG(INFO) << "Running!";

            int count = 0;
            Eigen::Vector3d avgA = Eigen::Vector3d::Zero();
            std::vector<Init::Input> input;
            ImuData::const_iterator im = imu_data.cbegin();
            for (unsigned n = 0; n < nframes; n++)
            {
                im = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                        (im, imu_data.cend(), traj->timestamp, simul);

                if (!simul || im == imu_data.cend())
                {
                    LOG(WARNING) << "Couldn't find IMU measurement at " << im->timestamp;
                    break;
                }
                if(im->timestamp > traj->timestamp && im != imu_data.cbegin())
                    std::advance(im, -1);// 保证imu不大于traj

                timestamp_t forward_time = (1.0f/Init::InitRate)*1e9;
                Trajectory::const_iterator traj_forw = Move_Iterator<Trajectory::value_type, Trajectory::const_iterator>
                        (traj, trajectory.cend(), forward_time, simul);

                if (!simul || traj_forw == trajectory.cend())
                {
                    LOG(WARNING) << "Couldn't find next frame for " << traj->timestamp;
                    break;
                }

                // 两帧之间积分起来
                std::shared_ptr<IMU::Preintegrated> pInt = std::make_shared<IMU::Preintegrated>(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
                while (im != imu_data.cend() && (im->timestamp < traj_forw->timestamp))
                {
                    double delta_t = IMU::dt;
                    timestamp_t t1 = im->timestamp;
                    const Eigen::Vector3d w1(im->w_x, im->w_y, im->w_z);
                    const Eigen::Vector3d a1(im->a_x, im->a_y, im->a_z);
                    std::advance(im, 1);
                    timestamp_t t2 = im->timestamp;
                    const Eigen::Vector3d w2(im->w_x, im->w_y, im->w_z);
                    const Eigen::Vector3d a2(im->a_x, im->a_y, im->a_z);

                    if(t1 < traj->timestamp)
                        delta_t = (t2 - traj->timestamp)*1e-9;

                    if(t2 > traj_forw->timestamp)
                        delta_t = (traj_forw->timestamp - t1)*1e-9;

                    pInt->IntegrateNewMeasurement_Mid(w1, a1, w2, a2, delta_t);
                }

                if (im == imu_data.cend())
                {
                    LOG(WARNING) << "IMU stream ended!";
                    break;
                }

                avgA += pInt->dV/pInt->dT;
                input.emplace_back(traj->pose, traj->timestamp, traj_forw->pose, traj_forw->timestamp, pInt);

                traj = traj_forw;
            }

            if (input.size() < nframes)
            {
                LOG(INFO) << StringPrintf("I don't have %d frames. I think dataset ended...", nframes);
                break;
            }


            gt = Find_Iterator<State::value_type, State::const_iterator>
                    (groundtruth.cbegin(), groundtruth.cend(), traj_copy->timestamp, simul);
            if (!simul)
            {
                LOG(WARNING) << "Couldn't find groundtruth with these trajectory!";
                break;
            }
            Eigen::Vector3d avgBg = Eigen::Vector3d::Zero();// 这个是GT
            Eigen::Vector3d avgBa = Eigen::Vector3d::Zero();

            // Get groundtruth of ba and bg.

            count = 0;
            while (gt != groundtruth.cend() && !IsSimu(gt->timestamp, traj->timestamp))
            {
                avgBg += Eigen::Vector3d(gt->bw_x, gt->bw_y, gt->bw_z);
                avgBa += Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z);
                count ++;
                std::advance(gt, 1);
            }
            if (gt == groundtruth.cend())
            {
                LOG(WARNING) << "groundtruth stream ended!";
                break;
            }

            avgBg /= count;// 平均Bg（GT）// 这个平均是用相邻图像帧的GT加起来的
            avgBa /= count;// 平均Ba（GT)

            avgA /= static_cast<double>(nframes);// 原始数据积分得到的平均速度，用来判断是否存在明显加速
            const double avgA_error = std::abs(avgA.norm() - IMU::GRAVITY_MAGNITUDE) / IMU::GRAVITY_MAGNITUDE;
            if (avgA_error > 5e-3)// 如果有明显加速
            {
                timestamp_t timestamp = input[0].t1;
                timestamp_t visual_time = traj->timestamp - traj_copy->timestamp;// 这个时间是先跑一段视觉的时间

                double true_scale;
                Eigen::Isometry3d Transfrom = Compute_Scale(input, groundtruth, true_scale);

                std::cout << "True_s:" << true_scale << std::endl;

                // proposed method
                {
                    Init::Result proposed_result;
                    Init::Init_Analytical_Method(input, proposed_result, Tcb);
                    if (proposed_result.success)
                    {
                        const double scale_error = 100.*std::abs(proposed_result.scale - true_scale)/true_scale;
                        const double gyro_bias_error = 100.*(proposed_result.bias_g - avgBg).norm() / avgBg.norm();
                        const double acc_bias_error = 100.*(proposed_result.bias_a - avgBa).norm() / avgBa.norm();
                        const double gravity_error = 180.*std::acos((Transfrom.rotation()*proposed_result.gravity).normalized().dot(IMU::GRAVITY_VECTOR.normalized()))/EIGEN_PI;
                        evaluation_t eval_data(proposed_result.solve_ns, visual_time, timestamp,
                                       scale_error, gyro_bias_error, acc_bias_error, gravity_error);
                        proposed_evaluation.push_back(eval_data);
                    }
                    else
                        LOG(ERROR) << "Proposed method failed at " << timestamp;
                }

                // iterative method
                {
                    Init::Result iterative_result;
                    double min_cost = std::numeric_limits<double>::max();
                    std::int64_t max_solve_time = 0;
                    std::vector<double> scale_values = {1., 4., 16.};
                    for (const double scale : scale_values)
                    {
                        double cost;
                        Init::Result result;
                        Init::Init_ORBSLAM3_Method(input, result, cost, scale, Tcb);
                        max_solve_time = std::max(max_solve_time, result.solve_ns);
                        if (cost < min_cost)
                        {
                            iterative_result = result;
                            min_cost = cost;
                        }
                    }
                    iterative_result.solve_ns = max_solve_time;
                    if (iterative_result.success)
                    {
                        const double scale_error = 100.*std::abs(iterative_result.scale - true_scale)/true_scale;
                        const double gyro_bias_error = 100.*(iterative_result.bias_g - avgBg).norm() / avgBg.norm();
                        const double acc_bias_error = 100.*(iterative_result.bias_a - avgBa).norm() / avgBa.norm();
                        const double gravity_error = 180.*std::acos((Transfrom.rotation()*iterative_result.gravity).normalized().dot(IMU::GRAVITY_VECTOR.normalized()))/EIGEN_PI;
                        iterative_evaluation.emplace_back(iterative_result.solve_ns, visual_time, timestamp,
                                                  scale_error, gyro_bias_error, acc_bias_error, gravity_error);
                    }
                    else
                        LOG(ERROR) << "Iterative method failed at " << timestamp;
                }

            }
            traj = Move_Iterator<Trajectory::value_type, Trajectory::const_iterator>
                    (traj_copy, trajectory.cend(), 500000000);
            traj_copy = traj;
        }
        std::string proposed_file = "testing_ours.csv";
        LOG(INFO) << "Saving evaluation data into " << proposed_file;
        save(proposed_evaluation, proposed_file);

        std::string iterative_file = "testing_iterative.csv";
        LOG(INFO) << "Saving evaluation data into " << proposed_file;
        save(iterative_evaluation, iterative_file);
    }
    LOG(INFO) << "done." << std::endl;
}

void Temporal_Calib(const fs::path &sequence_path)
{
    fs::path trajectory_path = sequence_path / "state_groundtruth_estimate0" / "data.csv";
    CHECK(fs::is_regular_file(trajectory_path)) << "Path not found: " << trajectory_path.string();

    fs::path data_path = sequence_path / "imu0" / "data.csv";
    CHECK(fs::is_regular_file(data_path)) << "Path not found: " << data_path.string();

    State trajectory = Read_File<State::value_type>(trajectory_path.string());
    ImuData imu_data = Read_File<ImuData::value_type>(data_path.string());

    State::const_iterator iter_t = trajectory.cbegin();
    ImuData::const_iterator iter_i = imu_data.cbegin();
    while(!IsSimu(iter_t->timestamp, iter_i->timestamp))// 先找到一致的时间戳
    {
        if(iter_t->timestamp > iter_i->timestamp)
            std::advance(iter_i, 1);
        else
            std::advance(iter_t, 1);
    }

    int dur_time = 1;

    int rangetime = dur_time + 2;


    State::const_iterator iter_t_origin = iter_t;
    ImuData::const_iterator iter_i_origin = iter_i;

    double time_error = 0;
    int count = 0;
    while(1)
    {

        iter_t = Move_Iterator<State::value_type, State::const_iterator>
                (iter_t_origin, trajectory.cend(), 10000000000);//移动10s
        iter_i = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                (iter_i_origin, imu_data.cend(), iter_t->timestamp);

        if(iter_t == trajectory.cend() || iter_i == imu_data.cend())
        {
            std::cout << "data end!!!" << std::endl;
            break;
        }

        std::vector<Eigen::Vector3d> wg;
        Eigen::Vector3d wg_ave;
        wg_ave.setZero();
        iter_t = Move_Iterator<State::value_type, State::const_iterator>
                (iter_t_origin, trajectory.cend(), 1000000000);//移动1s
        int manualset = 0;
        std::advance(iter_t, manualset);// manual timeoffset

        std::vector<Eigen::Matrix3d> RotMat;
        for(int i = 0; i <= 20*dur_time; i++)// 20hz, 8s
        {
            RotMat.push_back(Eigen::Isometry3d(iter_t->pose).rotation());
            std::advance(iter_t, 10);
        }

        std::vector<Eigen::Vector3d> imu_w;
        iter_i = iter_i_origin;

        for(int i = 0; i <= 200*rangetime; i++)// 200hz, 10s
        {
            imu_w.push_back(Eigen::Vector3d(iter_i->w_x, iter_i->w_y, iter_i->w_z));
            std::advance(iter_i, 1);
        }

        double time = Init::Temporal_Calibration(RotMat, imu_w) - 1.0f;
        double error = std::abs(time - manualset*0.005f)*10e3;// ms
        time_error += error;
        std::cout << "Res:" << time_error << std::endl;

        iter_t_origin = Move_Iterator<State::value_type, State::const_iterator>
                (iter_t_origin, trajectory.cend(), 1000000000);//移动10s
        iter_i_origin = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                (iter_i_origin, imu_data.cend(), iter_t_origin->timestamp);
        count ++;
    }

    time_error /= count;

    std::cout << time_error << std::endl;
}

void Test_Integration(const fs::path &sequence_path)
{
    fs::path groundtruth_path = sequence_path / "state_groundtruth_estimate0" / "data.csv";
    CHECK(fs::is_regular_file(groundtruth_path)) << "Path not found: " << groundtruth_path.string();

    fs::path data_path = sequence_path / "imu0" / "data.csv";
    CHECK(fs::is_regular_file(data_path)) << "Path not found: " << data_path.string();

    State groundtruth = Read_File<State::value_type>(groundtruth_path.string());
    ImuData imu_data = Read_File<ImuData::value_type>(data_path.string());

    State::const_iterator iter_t = groundtruth.cbegin();
    ImuData::const_iterator iter_i = imu_data.cbegin();

    while(!IsSimu(iter_t->timestamp, iter_i->timestamp))// 先找到一致的时间戳
    {
        if(iter_t->timestamp > iter_i->timestamp)
            std::advance(iter_i, 1);
        else
            std::advance(iter_t, 1);
    }

    State::const_iterator iter_t_origin = iter_t;
    ImuData::const_iterator iter_i_origin = iter_i;

    Eigen::Isometry3d PoseOrigin = iter_t_origin->pose;
    Eigen::Vector3d TransOrigin = PoseOrigin.translation();
    Eigen::Matrix3d RotOrigin = PoseOrigin.rotation();
    Eigen::Vector3d VelOrigin = Eigen::Vector3d(iter_t_origin->v_x, iter_t_origin->v_y, iter_t_origin->v_z);

    int count = 0;
    double error_trans = 0;
    double error_rot = 0;
    double error_vel = 0;
    std::shared_ptr<IMU::Preintegrated> pInt1 = std::make_shared<IMU::Preintegrated>(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    while(1)
    {

        count ++;
        const Eigen::Vector3d ba(iter_t->ba_x, iter_t->ba_y, iter_t->ba_z);
        const Eigen::Vector3d bg(iter_t->bw_x, iter_t->bw_y, iter_t->bw_z);
        pInt1->SetOriginalAccBias(ba);
        pInt1->SetOriginalGyroBias(bg);

//        const Eigen::Vector3d w(iter_i->w_x, iter_i->w_y, iter_i->w_z);
//        const Eigen::Vector3d a(iter_i->a_x, iter_i->a_y, iter_i->a_z);
//        pInt1->IntegrateNewMeasurement(w, a, IMU::dt);
//        std::advance(iter_t, 1);
//        std::advance(iter_i, 1);

        const Eigen::Vector3d w1(iter_i->w_x, iter_i->w_y, iter_i->w_z);
        const Eigen::Vector3d a1(iter_i->a_x, iter_i->a_y, iter_i->a_z);
        std::advance(iter_t, 1);
        std::advance(iter_i, 1);
        const Eigen::Vector3d w2(iter_i->w_x, iter_i->w_y, iter_i->w_z);
        const Eigen::Vector3d a2(iter_i->a_x, iter_i->a_y, iter_i->a_z);
        pInt1->IntegrateNewMeasurement_Pro(w1, a1, w2, a2, IMU::dt);

//        pInt1->IntegrateNewMeasurement_Mid(w1, a1, w2, a2, IMU::dt);

//        pInt1->IntegrateNewMeasurement(0.5*(w1 + w2), 0.5*(a1 + a2), IMU::dt);


        double err_t, err_r, err_v;
        if(count%10 == 0)
        {
            const double deltat = 10*IMU::dt;
            const Eigen::Isometry3d PoseNow = iter_t->pose;
            const Eigen::Vector3d TransNow = PoseNow.translation();
            const Eigen::Matrix3d RotNow = PoseNow.rotation();
            const Eigen::Vector3d VelNow = Eigen::Vector3d(iter_t->v_x, iter_t->v_y, iter_t->v_z);

            err_t = (TransOrigin + VelOrigin*deltat + 0.5*IMU::GRAVITY_VECTOR*deltat*deltat + RotOrigin*pInt1->dP - TransNow).norm();
            err_r = LogSO3(pInt1->dR.transpose()*(RotOrigin.transpose()*RotNow)).norm();
            err_v = (VelOrigin + IMU::GRAVITY_VECTOR*deltat + RotOrigin*pInt1->dV - VelNow).norm();

            error_trans += err_t;
            error_rot += err_r;
            error_vel += err_v;

            pInt1->Initialize(bg, ba);

            TransOrigin = TransNow;
            VelOrigin = VelNow;
            RotOrigin = RotNow;

            std::cout << count << ", Trans:" << err_t << ", Rot:" << err_r << ", Vel:" << err_v << std::endl;
        }
        if(iter_t == groundtruth.cend() || iter_i == imu_data.cend())
        {
            std::cout << count << ", Trans:" << err_t << ", Rot:" << err_r << ", Vel:" << err_v << std::endl;
            break;
        }
    }
    std::cout << "ALLTrans:" << error_trans << ", ALLRot:" << error_rot << ", ALLVel:" << error_vel << std::endl;
}


void ReadPoints(const fs::path &strPointsPath, const fs::path &strPathTimes, AllPoints &AP)
{
    std::ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    while(!fTimes.eof())
    {
        std::string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            std::string pointpath = strPointsPath.c_str() + ss.str() + ".csv";
            if(fs::is_regular_file(pointpath))
            {
                Points pt = Read_File<Points::value_type>(pointpath);
                timestamp_t t;
                ss >> t;
                AP.emplace_back(t, pt);
            }
        }
    }
}

//void Points_Culling(const AllPoints &PInits, std::vector<std::vector<Eigen::Vector3d>> &AlignedPts, bool &state, const AllPoints &APO)
void Points_Culling(const AllPoints &PInits, std::vector<std::vector<Eigen::Vector3d>> &AlignedPts, bool &state)
{
    // culling掉非共有的点


    point_t front_endpt = PInits.front().points.back();
    Points::const_iterator Pt_back_iter;
    Pt_back_iter = std::upper_bound(PInits.back().points.begin(), PInits.back().points.end(), front_endpt,
                                    [](const point_t &a, const point_t &b){return a.id < b.id;});
    int PtNums = std::distance(PInits.back().points.begin(), Pt_back_iter);
    if(PtNums < Init::PointNum)
    {
        std::cout << "Don't have enough common Points!!!" << std::endl;
        state = false;
        return;
    }

    Points RefPts;// 最后一帧中被提取出来的共有的pts
    RefPts.assign(PInits.back().points.begin(), Pt_back_iter);

    std::cout << "FirstID:::" << RefPts.front().id << std::endl;

    int PNumAll = RefPts.size();
    const int PNum = Init::PointNum;


    std::vector<std::vector<Eigen::Vector3d>> AlignedPtsAll;
    for(auto &fpts : PInits)
    {
        int index = 0;
        std::vector<Eigen::Vector3d> Ptstmp;
        for(auto &pts : fpts.points)
        {
            if(pts.id == RefPts[index].id)
            {
                Eigen::Vector3d unitpt(pts.x, pts.y, 1.0f);
                Ptstmp.push_back(unitpt.normalized());// 归一化单位向量
//                Ptstmp.emplace_back(pts.x, pts.y, 1.0f);// 归一化平面
                index ++;
//                if(index == PNumAll) break;
                if(index == PNum) break;
            }
        }
        AlignedPts.push_back(Ptstmp);
//        AlignedPtsAll.push_back(Ptstmp);
    }


    // 选择tracking距离最大的点
    /*
    std::vector<std::pair<double, int>> dist;
    for(int i = 0; i < PNumAll; i++)
    {
        double dis = 0;
        for(int j = 1; j < (int)AlignedPtsAll.size(); j++)
        {
            dis += (AlignedPtsAll[j][i] - AlignedPtsAll[j - 1][i]).norm();
        }
        dist.emplace_back(dis, i);
    }
    std::sort(dist.begin(), dist.end(), [](std::pair<double, int> &a, std::pair<double, int>&b)
    {
        return a.first > b.first;
    });


    for(auto &p : AlignedPtsAll)
    {
        std::vector<Eigen::Vector3d> Ptstmp;
        for(int i = 0; i < PNum; i++)
        {
            int ind = dist[i].second;
            Ptstmp.push_back(p[ind]);
        }
        AlignedPts.push_back(Ptstmp);
    }
    */

    /*
    std::vector<int> PtsIndex;
    std::default_random_engine e(time(0));
    std::uniform_int_distribution<int> u(0, PNumAll - 1);
    for(int i = 0; i < PNum; i++)
    {
        int r = u(e);
        while(std::find(PtsIndex.begin(), PtsIndex.end(), r) != PtsIndex.end())// 如果已经有这个值了
            r = (r + 1) % PNumAll;
        PtsIndex.push_back(r);
    }
    std::sort(PtsIndex.begin(), PtsIndex.end());

    for(auto &APAll : AlignedPtsAll)
    {
        std::vector<Eigen::Vector3d> Ptstmp;
        for(auto &r : PtsIndex)// 按照随机得到的index扔点
        {
            Ptstmp.push_back(APAll[r]);
        }
        AlignedPts.push_back(Ptstmp);
    }
*/

    state = true;
}


// 剔出所有满足条件的点，而不限于PointNum个
void Points_CullingAll(const AllPoints &PInits, std::vector<std::vector<Eigen::Vector3d>> &AlignedPts, bool &state)
{
    // culling掉非共有的点


    point_t front_endpt = PInits.front().points.back();
    Points::const_iterator Pt_back_iter;
    Pt_back_iter = std::upper_bound(PInits.back().points.begin(), PInits.back().points.end(), front_endpt,
                                    [](const point_t &a, const point_t &b){return a.id < b.id;});
    int PtNums = std::distance(PInits.back().points.begin(), Pt_back_iter);
    if(PtNums < Init::PointNum)
    {
        std::cout << "Don't have enough common Points!!!" << std::endl;
        state = false;
        return;
    }

    Points RefPts;// 最后一帧中被提取出来的共有的pts
    RefPts.assign(PInits.back().points.begin(), Pt_back_iter);

    std::cout << "FirstID:::" << RefPts.front().id << std::endl;


    int PNumAll = RefPts.size();

//    if(PNumAll > 20)
//        PNumAll = 20;


    std::vector<std::vector<Eigen::Vector3d>> AlignedPtsAll;

    for(auto &fpts : PInits)
    {
        int index = 0;
        std::vector<Eigen::Vector3d> Ptstmp;
        for(auto &pts : fpts.points)
        {
            if(pts.id == RefPts[index].id)
            {
                Eigen::Vector3d unitpt(pts.x, pts.y, 1.0f);
                Ptstmp.push_back(unitpt.normalized());// 归一化单位向量
//                Ptstmp.emplace_back(pts.x, pts.y, 1.0f);// 归一化平面
                index ++;
                if(index == PNumAll) break;
            }
        }
        AlignedPts.push_back(Ptstmp);
    }
    std::cout << std::endl;




// 选择距离较长的点
/*
    std::vector<std::pair<double, int>> dist;
    for(int i = 0; i < PNumAll; i++)
    {
        double dis = 0;
        for(int j = 1; j < (int)AlignedPtsAll.size(); j++)
        {
            dis += (AlignedPtsAll[j][i] - AlignedPtsAll[j - 1][i]).norm();
        }
        dist.emplace_back(dis, i);
    }
    std::sort(dist.begin(), dist.end(), [](std::pair<double, int> &a, std::pair<double, int>&b)
    {
        return a.first > b.first;
    });

    int threshold = 1000;
    for(int i = 0; i < (int)dist.size(); i++)
    {
        if(dist[i].first < 0.2)
        {
            threshold = i;
            break;
        }
    }
    threshold = std::min(threshold, (int)dist.size());
    if(threshold < Init::PointNum)
    {
        state = false;
        return;
    }

    for(auto &p : AlignedPtsAll)
    {
        std::vector<Eigen::Vector3d> Ptstmp;
        for(int i = 0; i < threshold; i++)
        {
            int ind = dist[i].second;
            Ptstmp.push_back(p[ind]);
        }
        AlignedPts.push_back(Ptstmp);
    }
*/

    state = true;
}



void ReadLines(const fs::path &strLinesPath, const fs::path &strPathTimes, AllLines &AL)
{
    int freq = 0;
    std::ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    while(!fTimes.eof())
    {
        std::string s;
        getline(fTimes,s);
        if(!s.empty() && (freq%2 == 0))// 线特征需要以固定频率提取
        {
            std::stringstream ss;
            ss << s;
            std::string linepath = strLinesPath.c_str() + ss.str() + ".csv";
            if(fs::is_regular_file(linepath))
            {
                Lines ln = Read_File<Lines::value_type>(linepath);
                timestamp_t t;
                ss >> t;
                AL.emplace_back(t, ln);
            }
        }
        freq ++;
    }
}

void Lines_Culling(const AllLines &LInits, std::vector<std::vector<Eigen::Vector6d>> &AlignedLns, bool &state)
{
    // culling掉非共有的线

    line_t front_endln = LInits.front().lines.back();
    Lines::const_iterator Ln_back_iter;
    Ln_back_iter = std::upper_bound(LInits.back().lines.begin(), LInits.back().lines.end(), front_endln,
                                    [](const line_t &a, const line_t &b){return a.id < b.id;});
    int LnNums = std::distance(LInits.back().lines.begin(), Ln_back_iter);
    if(LnNums < Init::LineNum)
    {
        std::cout << "Don't have enough common Lines!!!" << std::endl;
        state = false;
        return;
    }

//    if(LnNums == 0)
//    {
//        AlignedLns.clear();
//        state = false;
//        return;
//    }

    LnNums = Init::LineNum;

    Lines RefLns;// 最后一帧中被提取出来的共有的pts
    RefLns.assign(LInits.back().lines.begin(), Ln_back_iter);

    for(auto &flns : LInits)
    {
        int index = 0;
        std::vector<Eigen::Vector6d> Lnstmp;// 保存两线之间的EndPoint
        for(auto &lns : flns.lines)
        {
            if(lns.id == RefLns[index].id)
            {
                Eigen::Vector6d tmp;
                tmp << lns.startX, lns.startY, 1.0f, lns.endX, lns.endY, 1.0f;
                Lnstmp.push_back(tmp);
                index ++;
                if(index == LnNums) break;
            }
        }
        AlignedLns.push_back(Lnstmp);
    }
    state = true;
}


void Lines_CullingAll(const AllLines &LInits, std::vector<std::vector<Eigen::Vector6d>> &AlignedLns, bool &state)
{
    // culling掉非共有的线

    line_t front_endln = LInits.front().lines.back();
    Lines::const_iterator Ln_back_iter;
    Ln_back_iter = std::upper_bound(LInits.back().lines.begin(), LInits.back().lines.end(), front_endln,
                                    [](const line_t &a, const line_t &b){return a.id < b.id;});
    int LnNums = std::distance(LInits.back().lines.begin(), Ln_back_iter);
    if(LnNums < Init::LineNum)
    {
        std::cout << "Don't have enough common Lines!!!" << std::endl;
        state = false;
        return;
    }

    Lines RefLns;// 最后一帧中被提取出来的共有的pts
    RefLns.assign(LInits.back().lines.begin(), Ln_back_iter);

    const int LNumAll = RefLns.size();

    for(auto &flns : LInits)
    {
        int index = 0;
        std::vector<Eigen::Vector6d> Lnstmp;// 保存两线之间的EndPoint
        for(auto &lns : flns.lines)
        {
            if(lns.id == RefLns[index].id)
            {
                Eigen::Vector6d tmp;
                tmp << lns.startX, lns.startY, 1.0f, lns.endX, lns.endY, 1.0f;
                Lnstmp.push_back(tmp);
                index ++;
                if(index == LNumAll) break;
            }
        }
        AlignedLns.push_back(Lnstmp);
    }
    state = true;
}


void Test_Closeform_Methods(const fs::path &sequence_path)
{
    int countttt = 0;
    fs::path groundtruth_path = sequence_path / "state_groundtruth_estimate0" / "data.csv";
    CHECK(fs::is_regular_file(groundtruth_path)) << "Path not found: " << groundtruth_path.string();

    fs::path imu_path = sequence_path / "imu0" / "data.csv";
    CHECK(fs::is_regular_file(imu_path)) << "Path not found: " << imu_path.string();

    State groundtruth = Read_File<State::value_type>(groundtruth_path.string());
    ImuData imu_data = Read_File<ImuData::value_type>(imu_path.string());

    fs::path point_path = "/home/qjy/Research/VISLAM/trackingpoints/V101/";
//    fs::path point_path = "/home/qjy/Research/VISLAM/trackingORBpoints/V101/";
    fs::path line_path = "/home/qjy/Research/VISLAM/trackinglines/V101/";
    fs::path path_timestamps = "/home/qjy/Dataset/EuRoC/EuRoC_TimeStamps/V101.txt";

    ImuData::const_iterator im = imu_data.cbegin();
    ImuData::const_iterator im_copy = im;
    bool simul;

    AllPoints AP;
    ReadPoints(point_path, path_timestamps, AP);
    AllPoints::const_iterator pts = AP.cbegin();
    AllPoints::const_iterator pts_copy = pts;

//    fs::path pointorigin_path = "/home/qjy/Research/VISLAM/OriginPts/";
//    AllPoints APO;
//    ReadPoints(pointorigin_path, path_timestamps, APO);

//    AllLines AL;
//    ReadLines(line_path, path_timestamps, AL);
//    AllLines::const_iterator lns = AL.cbegin();
//    AllLines::const_iterator lns_copy = lns;



//    for(int i = 0; i < 10; i++)
//    {
//        lns_copy = Move_Iterator<AllLines::value_type, AllLines::const_iterator>
//                (lns_copy, AL.cend(), 500000000);
//    }



    int cc = 0;
    while(pts_copy != AP.cend()) // 这个是用特征点构成的线闭式求解
    {
        std::cout << cc << std::endl;
        cc++;
        pts = pts_copy;
        im_copy = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                (imu_data.cbegin(), imu_data.cend(), pts_copy->timestamp, simul);
        AllPoints PInits;// 用于初始化的点
        PInits.push_back(*pts);

        std::vector<IMU::Preintegrated> pInts;// 用于初始化的预积分
        std::vector<IMU::Preintegrated> pIntsgt;// 用于初始化的预积分

        IMU::Preintegrated pInt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        IMU::Preintegrated pIntgt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        for(int frame_t = 0; frame_t < Init::FrameNum; frame_t++)
        {
            // 先找到与pt对应的IMU
            im = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                    (im_copy, imu_data.cend(), pts->timestamp, simul);

            if(!simul) break;

            if(im->timestamp > pts->timestamp && im != imu_data.cbegin())
                std::advance(im, -1);// 保证imu不大于pt

            timestamp_t forward_time = (1.0f/Init::InitRate)*1e9;
            AllPoints::const_iterator pts_forw = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                    (pts, AP.cend(), forward_time, simul);

            if (!simul || pts_forw == AP.cend())
            {
                LOG(WARNING) << "Couldn't find next frame for " << pts->timestamp;
                break;
            }

            State::const_iterator gt;
            gt = Find_Iterator<State::value_type, State::const_iterator>
                    (groundtruth.cbegin(), groundtruth.cend(), im->timestamp, simul);
            if(!simul) break;
            pIntgt.SetOriginalGyroBias(Eigen::Vector3d(gt->bw_x, gt->bw_y, gt->bw_z));
            pIntgt.SetOriginalAccBias(Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z));

//            pInt.SetOriginalAccBias(Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z));

// 相邻两帧积分时用
            // 两帧之间积分起来
//            IMU::Preintegrated pInt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

//             预积分中添加gt的bg和ba
//            State::const_iterator gt;
//            gt = Find_Iterator<State::value_type, State::const_iterator>
//                    (groundtruth.cbegin(), groundtruth.cend(), im->timestamp, simul);
//            if(!simul) break;
//            IMU::Preintegrated pIntgt(Eigen::Vector3d(gt->bw_x, gt->bw_y, gt->bw_z),
//                                    Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z));

            while (im != imu_data.cend() && (im->timestamp < pts_forw->timestamp))
            {
                double delta_t = IMU::dt;
                timestamp_t t1 = im->timestamp;
                const Eigen::Vector3d w1(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a1(im->a_x, im->a_y, im->a_z);
                std::advance(im, 1);
                timestamp_t t2 = im->timestamp;
                const Eigen::Vector3d w2(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a2(im->a_x, im->a_y, im->a_z);

                if(t1 < pts->timestamp)
                    delta_t = (t2 - pts->timestamp)*1e-9;
                else if(t2 > pts_forw->timestamp)
                    delta_t = (pts_forw->timestamp - t1)*1e-9;

                pInt.IntegrateNewMeasurement(0.5*(w1 + w2), 0.5*(a1 + a2), delta_t);
                pIntgt.IntegrateNewMeasurement(0.5*(w1 + w2), 0.5*(a1 + a2), delta_t);
            }
            if (im == imu_data.cend())
            {
                LOG(WARNING) << "IMU stream ended!";
                break;
            }
            pInts.push_back(pInt);
            pIntsgt.push_back(pIntgt);
            PInits.push_back(*pts_forw);

            pts = pts_forw;
        }

        if(pInts.size() == Init::FrameNum)// 开始优化
        {
            // Points Culling对齐
            std::vector<std::vector<Eigen::Vector3d>> AlignedPts;
//            Points_Culling(PInits, AlignedPts, simul, APO);
            Points_Culling(PInits, AlignedPts, simul);
//            Points_CullingAll(PInits, AlignedPts, simul);

            // 找到对应位置的gt
            bool simul1 = false;
            bool simul2 = false;
            State::const_iterator gt1, gt2;

            gt1 = Find_Iterator<State::value_type, State::const_iterator>
                    (groundtruth.cbegin(), groundtruth.cend(), pts_copy->timestamp, simul1);
            gt2 = Find_Iterator<State::value_type, State::const_iterator>
                    (gt1, groundtruth.cend(), pts->timestamp, simul2);

            Eigen::Vector3d gtg;
            Eigen::Vector3d gtv(gt1->v_x, gt1->v_y, gt1->v_z);
            Eigen::Isometry3d InitPos = gt1->pose;

            gtv = InitPos.rotation().transpose()*gtv;
            gtg = InitPos.rotation().transpose()*IMU::GRAVITY_VECTOR;

            if(simul && simul1 && simul2)//找到足够的公共点，且正确找到gt的两个端点
            {
                Eigen::Vector3d gtbg;
                Eigen::Vector3d gtba;
                gtbg.setZero();
                gtba.setZero();
                double gtcnt = 0;
                while(gt1 != gt2)
                {
                    gtbg += Eigen::Vector3d(gt1->bw_x, gt1->bw_y, gt1->bw_z);
                    gtba += Eigen::Vector3d(gt1->ba_x, gt1->ba_y, gt1->ba_z);
                    std::advance(gt1, 1);
                    gtcnt ++;
                }
                gtbg /= gtcnt;
                gtba /= gtcnt;

                Init::Result closeform_result;
                Init::CloseForm_Solution(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, pIntsgt);

//                std::vector<Eigen::Vector3d> gttrans;
//                Init::Reprojection_SolutionR21(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, gttrans);
//                Init::CloseForm_Solution(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, pIntsgt);
//                Init::CloseForm_Solution_Point2Line(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, pIntsgt);
                countttt++;
            }
        }

        pts_copy = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                (pts_copy, AP.cend(), 500000000);
    }
//    std::cout << "Final Time:" << Init::StaTime / Init::StaCount << std::endl;
//    std::cout << "Final Errorv:" << Init::StaErrorv / Init::StaCount << std::endl;
    std::cout << "Final Error:" << Init::StaError / Init::StaCount << std::endl;
    std::cout << "Final ErrorPercen:" << Init::ErrorPercen / Init::StaCount << std::endl;

    std::cout<<"endl::"<<countttt<<std::endl;
    for(int i = 0; i < 100; i++)
    {
        std::cout << i << ": " << Init::statistical[i] << std::endl;
    }
}



void Test_Closeform_Methods_PAL(const fs::path &sequence_path)
{
    int countttt = 0;
//    fs::path groundtruth_path = sequence_path / "state_groundtruth_estimate0" / "data.csv";
//    CHECK(fs::is_regular_file(groundtruth_path)) << "Path not found: " << groundtruth_path.string();

//    fs::path imu_path = sequence_path / "imu0" / "data.csv";
//    CHECK(fs::is_regular_file(imu_path)) << "Path not found: " << imu_path.string();

//    State groundtruth = Read_File<State::value_type>(groundtruth_path.string());
//    ImuData imu_data = Read_File<ImuData::value_type>(imu_path.string());

//    fs::path point_path = "/home/qjy/Research/VISLAM/trackingpoints/V101/";
//    fs::path line_path = "/home/qjy/Research/VISLAM/trackinglines/V101/";
//    fs::path path_timestamps = "/home/qjy/Dataset/EuRoC/EuRoC_TimeStamps/V101.txt";



    fs::path groundtruth_path = sequence_path / "GroundTruth" / "groundtruth.csv";
    CHECK(fs::is_regular_file(groundtruth_path)) << "Path not found: " << groundtruth_path.string();

    fs::path imu_path = sequence_path / "Imu" / "imudata.csv";
    CHECK(fs::is_regular_file(imu_path)) << "Path not found: " << imu_path.string();

    State groundtruth = Read_File<State::value_type>(groundtruth_path.string());
    ImuData imu_data = Read_File<ImuData::value_type>(imu_path.string());

    fs::path point_path = "/home/qjy/Research/VISLAM/SimulData/Points/";
    fs::path line_path = "/home/qjy/Research/VISLAM/SimulData/Lines/";
    fs::path path_timestamps = "/home/qjy/Research/VISLAM/SimulData/timestamps.txt";


    ImuData::const_iterator im = imu_data.cbegin();
    ImuData::const_iterator im_copy = im;
    bool simul;

    AllPoints AP;
    ReadPoints(point_path, path_timestamps, AP);
    AllPoints::const_iterator pts = AP.cbegin();
    AllPoints::const_iterator pts_copy = pts;

    AllLines AL;
    ReadLines(line_path, path_timestamps, AL);
    AllLines::const_iterator lns = AL.cbegin();
    AllLines::const_iterator lns_copy = lns;


    pts = Find_Iterator<AllPoints::value_type, AllPoints::const_iterator>
            (pts, AP.cend(), lns->timestamp, simul);
    if(!simul)
    {
        std::cout << "Points And Lines Can't Simul !!!!!!!!!!";
        return;
    }

//    for(int i = 0; i < 10; i++)
//    {
//        lns_copy = Move_Iterator<AllLines::value_type, AllLines::const_iterator>
//                (lns_copy, AL.cend(), 500000000);
//    }



    int cc = 0;
    const timestamp_t forward_time = (1.0f/Init::InitRate)*1e9;
    while(pts_copy != AP.cend()) // 这个是用特征点构成的线闭式求解
    {
        std::cout << cc << std::endl;
        cc++;
        pts = pts_copy;
        lns = lns_copy;
        im_copy = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                (imu_data.cbegin(), imu_data.cend(), pts_copy->timestamp, simul);
        AllPoints PInits;// 用于初始化的点
        PInits.push_back(*pts);
        AllLines LInits;// 用于初始化的线
        LInits.push_back(*lns);

        std::vector<IMU::Preintegrated> pInts;// 用于初始化的预积分
        std::vector<IMU::Preintegrated> pIntsgt;// 用于初始化的预积分

        IMU::Preintegrated pInt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        IMU::Preintegrated pIntgt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        for(int frame_t = 0; frame_t < Init::FrameNum; frame_t++)
        {
            // 先找到与pt对应的IMU
            im = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                    (im_copy, imu_data.cend(), pts->timestamp, simul);

            if(!simul) break;

            if(im->timestamp > pts->timestamp && im != imu_data.cbegin())
                std::advance(im, -1);// 保证imu不大于pt

            AllPoints::const_iterator pts_forw = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                    (pts, AP.cend(), forward_time, simul);
            if (!simul || pts_forw == AP.cend())
            {
                LOG(WARNING) << "Couldn't find next frame for " << pts->timestamp;
                break;
            }
            AllLines::const_iterator lns_forw = Move_Iterator<AllLines::value_type, AllLines::const_iterator>
                    (lns, AL.cend(), forward_time, simul);
            if (!simul || lns_forw == AL.cend())
            {
                LOG(WARNING) << "Couldn't find next frame for " << lns->timestamp;
                break;
            }

            State::const_iterator gt;
            gt = Find_Iterator<State::value_type, State::const_iterator>
                    (groundtruth.cbegin(), groundtruth.cend(), im->timestamp, simul);
            if(!simul) break;
            pIntgt.SetOriginalGyroBias(Eigen::Vector3d(gt->bw_x, gt->bw_y, gt->bw_z));
            pIntgt.SetOriginalAccBias(Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z));

//            pInt.SetOriginalAccBias(Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z));

// 相邻两帧积分时用
            // 两帧之间积分起来
//            IMU::Preintegrated pInt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

//             预积分中添加gt的bg和ba
//            State::const_iterator gt;
//            gt = Find_Iterator<State::value_type, State::const_iterator>
//                    (groundtruth.cbegin(), groundtruth.cend(), im->timestamp, simul);
//            if(!simul) break;
//            IMU::Preintegrated pIntgt(Eigen::Vector3d(gt->bw_x, gt->bw_y, gt->bw_z),
//                                    Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z));

            while (im != imu_data.cend() && (im->timestamp < pts_forw->timestamp))
            {
                double delta_t = IMU::dt;
                timestamp_t t1 = im->timestamp;
                const Eigen::Vector3d w1(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a1(im->a_x, im->a_y, im->a_z);
                std::advance(im, 1);
                timestamp_t t2 = im->timestamp;
                const Eigen::Vector3d w2(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a2(im->a_x, im->a_y, im->a_z);

                if(t1 < pts->timestamp)
                    delta_t = (t2 - pts->timestamp)*1e-9;
                else if(t2 > pts_forw->timestamp)
                    delta_t = (pts_forw->timestamp - t1)*1e-9;

                pInt.IntegrateNewMeasurement(0.5*(w1 + w2), 0.5*(a1 + a2), delta_t);
                pIntgt.IntegrateNewMeasurement(0.5*(w1 + w2), 0.5*(a1 + a2), delta_t);
            }
            if (im == imu_data.cend())
            {
                LOG(WARNING) << "IMU stream ended!";
                break;
            }
            pInts.push_back(pInt);
            pIntsgt.push_back(pIntgt);
            PInits.push_back(*pts_forw);
            LInits.push_back(*lns_forw);

            pts = pts_forw;
            lns = lns_forw;
        }

        if(pInts.size() == Init::FrameNum)// 开始优化
        {
            // Points Culling对齐
            std::vector<std::vector<Eigen::Vector3d>> AlignedPts;
            std::vector<std::vector<Eigen::Vector6d>> AlignedLns;
            bool simul1 = false;
            bool simul2 = false;
            bool simul3 = false;
            Points_Culling(PInits, AlignedPts, simul);
//            Points_CullingAll(PInits, AlignedPts, simul);
            Lines_Culling(LInits, AlignedLns, simul1);

            // 找到对应位置的gt

            State::const_iterator gt1, gt2;

            gt1 = Find_Iterator<State::value_type, State::const_iterator>
                    (groundtruth.cbegin(), groundtruth.cend(), pts_copy->timestamp, simul2);
            gt2 = Find_Iterator<State::value_type, State::const_iterator>
                    (gt1, groundtruth.cend(), pts->timestamp, simul3);

            Eigen::Vector3d gtg;
            Eigen::Vector3d gtv(gt1->v_x, gt1->v_y, gt1->v_z);
            Eigen::Isometry3d InitPos = gt1->pose;

            gtv = InitPos.rotation().transpose()*gtv;
            gtg = InitPos.rotation().transpose()*IMU::GRAVITY_VECTOR;

            if(simul && simul1 && simul2 && simul3)//找到足够的公共点，且正确找到gt的两个端点
            {
                Eigen::Vector3d gtbg;
                Eigen::Vector3d gtba;
                gtbg.setZero();
                gtba.setZero();
                double gtcnt = 0;
                while(gt1 != gt2)
                {
                    gtbg += Eigen::Vector3d(gt1->bw_x, gt1->bw_y, gt1->bw_z);
                    gtba += Eigen::Vector3d(gt1->ba_x, gt1->ba_y, gt1->ba_z);
                    std::advance(gt1, 1);
                    gtcnt ++;
                }
                gtbg /= gtcnt;
                gtba /= gtcnt;

                Init::Result closeform_result;
                Init::CloseForm_Solution_PAL(closeform_result, AlignedPts, AlignedLns, pInts, Tbc, gtbg, gtba, gtv, gtg, pIntsgt);

//                std::vector<Eigen::Vector3d> gttrans;
//                Init::Reprojection_SolutionR21(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, gttrans);
//                Init::CloseForm_Solution(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, pIntsgt);
//                Init::CloseForm_Solution_Point2Line(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, pIntsgt);
                countttt++;
            }
        }

        pts_copy = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                (pts_copy, AP.cend(), 500000000);
        lns_copy = Move_Iterator<AllLines::value_type, AllLines::const_iterator>
                (lns_copy, AL.cend(), 500000000);
    }
    std::cout << "Final Time:" << Init::StaTime / Init::StaCount << std::endl;
    std::cout << "Final Errorv:" << Init::StaErrorv / Init::StaCount << std::endl;
    std::cout << "Final Errorg:" << Init::gravity_error / Init::StaCount << std::endl;
    std::cout << "Final Error:" << Init::StaError / Init::StaCount << std::endl;
    std::cout << "Final ErrorPercen:" << Init::ErrorPercen / Init::StaCount << std::endl;

    std::cout<<"endl::"<<countttt<<std::endl;
    for(int i = 0; i < 100; i++)
    {
        std::cout << i << ": " << Init::statistical[i] << std::endl;
    }
}



void Test_Simulated_Data(const fs::path &sequence_path)
{
    int countttt = 0;
    fs::path groundtruth_path = sequence_path / "GroundTruth" / "groundtruth.csv";
    CHECK(fs::is_regular_file(groundtruth_path)) << "Path not found: " << groundtruth_path.string();

    fs::path imu_path = sequence_path / "Imu" / "imudata.csv";
    CHECK(fs::is_regular_file(imu_path)) << "Path not found: " << imu_path.string();

    State groundtruth = Read_File<State::value_type>(groundtruth_path.string());
    ImuData imu_data = Read_File<ImuData::value_type>(imu_path.string());

    fs::path point_path = "/home/qjy/Research/VISLAM/SimulData/Points/";
    fs::path path_timestamps = "/home/qjy/Research/VISLAM/SimulData/timestamps.txt";

    ImuData::const_iterator im = imu_data.cbegin();
    ImuData::const_iterator im_copy = im;
    bool simul;

    AllPoints AP;
    ReadPoints(point_path, path_timestamps, AP);
    AllPoints::const_iterator pts = AP.cbegin();
    AllPoints::const_iterator pts_copy = pts;

    int cc = 0;
    while(pts_copy != AP.cend()) // 这个是用特征点构成的线闭式求解
    {
        std::cout << cc << std::endl;
        cc++;
        pts = pts_copy;
        im_copy = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                (imu_data.cbegin(), imu_data.cend(), pts_copy->timestamp, simul);
        AllPoints PInits;// 用于初始化的点
        PInits.push_back(*pts);

        std::vector<IMU::Preintegrated> pInts;// 用于初始化的预积分
        std::vector<IMU::Preintegrated> pIntsgt;// 用于初始化的预积分

        IMU::Preintegrated pInt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        IMU::Preintegrated pIntgt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        for(int frame_t = 0; frame_t < Init::FrameNum; frame_t++)
        {
            // 先找到与pt对应的IMU
            im = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                    (im_copy, imu_data.cend(), pts->timestamp, simul);

            if(!simul) break;

            if(im->timestamp > pts->timestamp && im != imu_data.cbegin())
                std::advance(im, -1);// 保证imu不大于pt

            timestamp_t forward_time = (1.0f/Init::InitRate)*1e9;
            AllPoints::const_iterator pts_forw = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                    (pts, AP.cend(), forward_time, simul);

            if (!simul || pts_forw == AP.cend())
            {
                LOG(WARNING) << "Couldn't find next frame for " << pts->timestamp;
                break;
            }

            State::const_iterator gt;
            gt = Find_Iterator<State::value_type, State::const_iterator>
                    (groundtruth.cbegin(), groundtruth.cend(), im->timestamp, simul);
            if(!simul) break;
            pIntgt.SetOriginalGyroBias(Eigen::Vector3d(gt->bw_x, gt->bw_y, gt->bw_z));
            pIntgt.SetOriginalAccBias(Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z));

//            pInt.SetOriginalAccBias(Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z));

// 相邻两帧积分时用
            // 两帧之间积分起来
//            IMU::Preintegrated pInt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

//             预积分中添加gt的bg和ba
//            State::const_iterator gt;
//            gt = Find_Iterator<State::value_type, State::const_iterator>
//                    (groundtruth.cbegin(), groundtruth.cend(), im->timestamp, simul);
//            if(!simul) break;
//            IMU::Preintegrated pIntgt(Eigen::Vector3d(gt->bw_x, gt->bw_y, gt->bw_z),
//                                    Eigen::Vector3d(gt->ba_x, gt->ba_y, gt->ba_z));

            while (im != imu_data.cend() && (im->timestamp < pts_forw->timestamp))
            {
                double delta_t = IMU::dt;
                timestamp_t t1 = im->timestamp;
                const Eigen::Vector3d w1(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a1(im->a_x, im->a_y, im->a_z);
                std::advance(im, 1);
                timestamp_t t2 = im->timestamp;
                const Eigen::Vector3d w2(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a2(im->a_x, im->a_y, im->a_z);

                if(t1 < pts->timestamp)
                    delta_t = (t2 - pts->timestamp)*1e-9;
                else if(t2 > pts_forw->timestamp)
                    delta_t = (pts_forw->timestamp - t1)*1e-9;

                pInt.IntegrateNewMeasurement(0.5*(w1 + w2), 0.5*(a1 + a2), delta_t);
                pIntgt.IntegrateNewMeasurement(0.5*(w1 + w2), 0.5*(a1 + a2), delta_t);
            }
            if (im == imu_data.cend())
            {
                LOG(WARNING) << "IMU stream ended!";
                break;
            }
            pInts.push_back(pInt);
            pIntsgt.push_back(pIntgt);
            PInits.push_back(*pts_forw);

            pts = pts_forw;
        }

        if(pInts.size() == Init::FrameNum)// 开始优化
        {
            // Points Culling对齐
            std::vector<std::vector<Eigen::Vector3d>> AlignedPts;
            Points_Culling(PInits, AlignedPts, simul);

            // 找到对应位置的gt
            bool simul1 = false;
            bool simul2 = false;
            State::const_iterator gt1, gt2;

            gt1 = Find_Iterator<State::value_type, State::const_iterator>
                    (groundtruth.cbegin(), groundtruth.cend(), pts_copy->timestamp, simul1);
            gt2 = Find_Iterator<State::value_type, State::const_iterator>
                    (gt1, groundtruth.cend(), pts->timestamp, simul2);

            Eigen::Vector3d gtg;
            Eigen::Vector3d gtv(gt1->v_x, gt1->v_y, gt1->v_z);
            Eigen::Isometry3d InitPos = gt1->pose;

            gtv = InitPos.rotation().transpose()*gtv;
            gtg = InitPos.rotation().transpose()*IMU::GRAVITY_VECTOR;

            if(simul && simul1 && simul2)//找到足够的公共点，且正确找到gt的两个端点
            {
                Eigen::Vector3d gtbg;
                Eigen::Vector3d gtba;
                gtbg.setZero();
                gtba.setZero();
                double gtcnt = 0;
                while(gt1 != gt2)
                {
                    gtbg += Eigen::Vector3d(gt1->bw_x, gt1->bw_y, gt1->bw_z);
                    gtba += Eigen::Vector3d(gt1->ba_x, gt1->ba_y, gt1->ba_z);
                    std::advance(gt1, 1);
                    gtcnt ++;
                }
                gtbg /= gtcnt;
                gtba /= gtcnt;

                Init::Result closeform_result;
                std::vector<std::vector<Eigen::Vector6d>> AlignedLns;
//                Init::CloseForm_Solution(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, pIntsgt);
                Init::CloseForm_Solution_PAL(closeform_result, AlignedPts, AlignedLns, pInts, Tbc, gtbg, gtba, gtv, gtg, pIntsgt);
                countttt++;
            }
        }

        pts_copy = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                (pts_copy, AP.cend(), 500000000);
    }

    std::cout << "Final Time:" << Init::StaTime / Init::StaCount << std::endl;
    std::cout << "Final Errorv:" << Init::StaErrorv / Init::StaCount << std::endl;
    std::cout << "Final Errorg:" << Init::gravity_error / Init::StaCount << std::endl;
    std::cout << "Final Error:" << Init::StaError / Init::StaCount << std::endl;
    std::cout << "Final ErrorPercen:" << Init::ErrorPercen / Init::StaCount << std::endl;

    std::cout<<"endl::"<<countttt<<std::endl;
    for(int i = 0; i < 100; i++)
    {
        std::cout << i << ": " << Init::statistical[i] << std::endl;
    }
}


void Test_Reprojection_Methods(const fs::path &sequence_path)
{
    int countttt = 0;
    fs::path groundtruth_path = sequence_path / "state_groundtruth_estimate0" / "data.csv";
    CHECK(fs::is_regular_file(groundtruth_path)) << "Path not found: " << groundtruth_path.string();

    fs::path imu_path = sequence_path / "imu0" / "data.csv";
    CHECK(fs::is_regular_file(imu_path)) << "Path not found: " << imu_path.string();

    State groundtruth = Read_File<State::value_type>(groundtruth_path.string());
    ImuData imu_data = Read_File<ImuData::value_type>(imu_path.string());

    fs::path point_path = "/home/qjy/Research/VISLAM/trackingpoints/V101/";
    fs::path line_path = "/home/qjy/Research/VISLAM/trackinglines/V101/";
    fs::path path_timestamps = "/home/qjy/Dataset/EuRoC/EuRoC_TimeStamps/V101.txt";

    ImuData::const_iterator im = imu_data.cbegin();
    ImuData::const_iterator im_copy = im;
    bool simul;

    AllPoints AP;
    ReadPoints(point_path, path_timestamps, AP);
    AllPoints::const_iterator pts = AP.cbegin();
    AllPoints::const_iterator pts_copy = pts;

//    AllLines AL;
//    ReadLines(line_path, path_timestamps, AL);
//    AllLines::const_iterator lns = AL.cbegin();
//    AllLines::const_iterator lns_copy = lns;

    int cc = 0;
    while(pts_copy != AP.cend()) // 这个是用特征点构成的线闭式求解
    {
        std::cout << cc << std::endl;
        cc++;
        pts = pts_copy;
        im_copy = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                (imu_data.cbegin(), imu_data.cend(), pts_copy->timestamp, simul);
        AllPoints PInits;// 用于初始化的点
        PInits.push_back(*pts);

        std::vector<IMU::Preintegrated> pInts;// 用于初始化的预积分

        for(int frame_t = 0; frame_t < Init::FrameNum; frame_t++)
        {
            IMU::Preintegrated pInt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            pInts.push_back(pInt);

            // 先找到与pt对应的IMU
            im = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                    (im_copy, imu_data.cend(), pts->timestamp, simul);

            if(!simul) break;

            if(im->timestamp > pts->timestamp && im != imu_data.cbegin())
                std::advance(im, -1);// 保证imu不大于pt

            timestamp_t forward_time = (1.0f/Init::InitRate)*1e9;
            AllPoints::const_iterator pts_forw = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                    (pts, AP.cend(), forward_time, simul);

            if (!simul || pts_forw == AP.cend())
            {
                LOG(WARNING) << "Couldn't find next frame for " << pts->timestamp;
                break;
            }

            while (im != imu_data.cend() && (im->timestamp < pts_forw->timestamp))
            {
                double delta_t = IMU::dt;
                timestamp_t t1 = im->timestamp;
                const Eigen::Vector3d w1(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a1(im->a_x, im->a_y, im->a_z);
                std::advance(im, 1);
                timestamp_t t2 = im->timestamp;
                const Eigen::Vector3d w2(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a2(im->a_x, im->a_y, im->a_z);

                if(t1 < pts->timestamp)
                    delta_t = (t2 - pts->timestamp)*1e-9;
                else if(t2 > pts_forw->timestamp)
                    delta_t = (pts_forw->timestamp - t1)*1e-9;

                for(auto &p : pInts)
                    p.IntegrateNewMeasurement(0.5*(w1 + w2), 0.5*(a1 + a2), delta_t);
            }
            if (im == imu_data.cend())
            {
                LOG(WARNING) << "IMU stream ended!";
                break;
            }

            PInits.push_back(*pts_forw);
            pts = pts_forw;
        }

        if(pInts.size() == Init::FrameNum)// 开始优化
        {
            // Points Culling对齐
            std::vector<std::vector<Eigen::Vector3d>> AlignedPts;
//            Points_Culling(PInits, AlignedPts, simul);
            Points_CullingAll(PInits, AlignedPts, simul);

            // 找到对应位置的gt
            bool simul1 = false;
            bool simul2 = false;
            State::const_iterator gt1, gt2;

            gt1 = Find_Iterator<State::value_type, State::const_iterator>
                    (groundtruth.cbegin(), groundtruth.cend(), pts_copy->timestamp, simul1);
            gt2 = Find_Iterator<State::value_type, State::const_iterator>
                    (gt1, groundtruth.cend(), pts->timestamp, simul2);

            Eigen::Vector3d gtg;
            Eigen::Vector3d gtv(gt2->v_x, gt2->v_y, gt2->v_z);
            Eigen::Isometry3d InitPos = gt2->pose;

            gtv = InitPos.rotation().transpose()*gtv;
            gtg = InitPos.rotation().transpose()*IMU::GRAVITY_VECTOR;


            if(simul && simul1 && simul2)//找到足够的公共点，且正确找到gt的两个端点
            {
                // 这里是获取t12的groundtruth
                auto gt1cop = gt1;
                auto gt2cop = gt2;
                std::vector<Eigen::Vector3d> Vecgttrans;
                timestamp_t forward_time = (1.0f/Init::InitRate)*1e9;
                while(gt1cop != gt2cop)
                {
                    Eigen::Vector3d gttrans1(gt1cop->pose.tx, gt1cop->pose.ty, gt1cop->pose.tz);
                    Eigen::Vector3d gttrans2(gt2cop->pose.tx, gt2cop->pose.ty, gt2cop->pose.tz);
                    InitPos = gt1cop->pose;
                    Eigen::Vector3d gttrans = InitPos.rotation().transpose()*(gttrans2 - gttrans1);
                    Vecgttrans.push_back(gttrans);
                    gt1cop = Move_Iterator<State::value_type, State::const_iterator>
                    (gt1cop, groundtruth.cend(), forward_time, simul);
                }


                Eigen::Vector3d gtbg;
                Eigen::Vector3d gtba;
                gtbg.setZero();
                gtba.setZero();
                double gtcnt = 0;
                while(gt1 != gt2)
                {
                    gtbg += Eigen::Vector3d(gt1->bw_x, gt1->bw_y, gt1->bw_z);
                    gtba += Eigen::Vector3d(gt1->ba_x, gt1->ba_y, gt1->ba_z);
                    std::advance(gt1, 1);
                    gtcnt ++;
                }
                gtbg /= gtcnt;
                gtba /= gtcnt;

                Init::Result closeform_result;
                Init::Reprojection_Solution(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, Vecgttrans);
//                Init::Reprojection_Closeform_Solution(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg);
//                Init::Reprojection_3D_Solution(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, Vecgttrans);
                countttt++;
            }
        }

        pts_copy = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                (pts_copy, AP.cend(), 500000000);
    }


//    std::cout << "Final Time:" << Init::StaTime / Init::StaCount << std::endl;
//    std::cout << "Final Errorv:" << Init::StaErrorv / Init::StaCount << std::endl;
    std::cout << "Final Error:" << Init::StaError / Init::StaCount << std::endl;
    std::cout << "Final ErrorPercen:" << Init::ErrorPercen / Init::StaCount << std::endl;
    std::cout << "Gravity Error:" << Init::gravity_error / Init::StaCount << std::endl;

    std::cout<<"endl::"<<countttt<<std::endl;
    for(int i = 0; i < 100; i++)
    {
        std::cout << i << ": " << Init::statistical[i] << std::endl;
    }
}



void Test_Simulated_Reprojection_Methods(const fs::path &sequence_path)
{

    int countttt = 0;
    fs::path groundtruth_path = sequence_path / "GroundTruth" / "groundtruth.csv";
    CHECK(fs::is_regular_file(groundtruth_path)) << "Path not found: " << groundtruth_path.string();

    fs::path imu_path = sequence_path / "Imu" / "imudata.csv";
    CHECK(fs::is_regular_file(imu_path)) << "Path not found: " << imu_path.string();

    State groundtruth = Read_File<State::value_type>(groundtruth_path.string());
    ImuData imu_data = Read_File<ImuData::value_type>(imu_path.string());

    fs::path point_path = "/home/qjy/Research/VISLAM/SimulData/Points/";
    fs::path path_timestamps = "/home/qjy/Research/VISLAM/SimulData/timestamps.txt";

    ImuData::const_iterator im = imu_data.cbegin();
    ImuData::const_iterator im_copy = im;
    bool simul;

    AllPoints AP;
    ReadPoints(point_path, path_timestamps, AP);
    AllPoints::const_iterator pts = AP.cbegin();
    AllPoints::const_iterator pts_copy = pts;



    int cc = 0;
    while(pts_copy != AP.cend()) // 这个是用特征点构成的线闭式求解
    {
        std::cout << cc << std::endl;
        cc++;
        pts = pts_copy;
        im_copy = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                (imu_data.cbegin(), imu_data.cend(), pts_copy->timestamp, simul);
        AllPoints PInits;// 用于初始化的点
        PInits.push_back(*pts);

        std::vector<IMU::Preintegrated> pInts;// 用于初始化的预积分

        for(int frame_t = 0; frame_t < Init::FrameNum; frame_t++)
        {
            IMU::Preintegrated pInt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            pInts.push_back(pInt);

            // 先找到与pt对应的IMU
            im = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                    (im_copy, imu_data.cend(), pts->timestamp, simul);

            if(!simul) break;

            if(im->timestamp > pts->timestamp && im != imu_data.cbegin())
                std::advance(im, -1);// 保证imu不大于pt

            timestamp_t forward_time = (1.0f/Init::InitRate)*1e9;
            AllPoints::const_iterator pts_forw = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                    (pts, AP.cend(), forward_time, simul);

            if (!simul || pts_forw == AP.cend())
            {
                LOG(WARNING) << "Couldn't find next frame for " << pts->timestamp;
                break;
            }

            while (im != imu_data.cend() && (im->timestamp < pts_forw->timestamp))
            {
                double delta_t = IMU::dt;
                timestamp_t t1 = im->timestamp;
                const Eigen::Vector3d w1(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a1(im->a_x, im->a_y, im->a_z);
                std::advance(im, 1);
                timestamp_t t2 = im->timestamp;
                const Eigen::Vector3d w2(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a2(im->a_x, im->a_y, im->a_z);

                if(t1 < pts->timestamp)
                    delta_t = (t2 - pts->timestamp)*1e-9;
                else if(t2 > pts_forw->timestamp)
                    delta_t = (pts_forw->timestamp - t1)*1e-9;

                for(auto &p : pInts)
                    p.IntegrateNewMeasurement(0.5*(w1 + w2), 0.5*(a1 + a2), delta_t);
            }
            if (im == imu_data.cend())
            {
                LOG(WARNING) << "IMU stream ended!";
                break;
            }

            PInits.push_back(*pts_forw);
            pts = pts_forw;
        }

        if(pInts.size() == Init::FrameNum)// 开始优化
        {
            // Points Culling对齐
            std::vector<std::vector<Eigen::Vector3d>> AlignedPts;
            Points_Culling(PInits, AlignedPts, simul);
//            Points_CullingAll(PInits, AlignedPts, simul);

            // 找到对应位置的gt
            bool simul1 = false;
            bool simul2 = false;
            State::const_iterator gt1, gt2;

            gt1 = Find_Iterator<State::value_type, State::const_iterator>
                    (groundtruth.cbegin(), groundtruth.cend(), pts_copy->timestamp, simul1);
            gt2 = Find_Iterator<State::value_type, State::const_iterator>
                    (gt1, groundtruth.cend(), pts->timestamp, simul2);

            Eigen::Vector3d gtg;
            Eigen::Vector3d gtv(gt2->v_x, gt2->v_y, gt2->v_z);
            Eigen::Isometry3d InitPos = gt2->pose;

            gtv = InitPos.rotation().transpose()*gtv;
            gtg = InitPos.rotation().transpose()*IMU::GRAVITY_VECTOR;

            if(simul && simul1 && simul2)//找到足够的公共点，且正确找到gt的两个端点
            {
                Eigen::Vector3d gtbg;
                Eigen::Vector3d gtba;
                gtbg.setZero();
                gtba.setZero();
                double gtcnt = 0;
                while(gt1 != gt2)
                {
                    gtbg += Eigen::Vector3d(gt1->bw_x, gt1->bw_y, gt1->bw_z);
                    gtba += Eigen::Vector3d(gt1->ba_x, gt1->ba_y, gt1->ba_z);
                    std::advance(gt1, 1);
                    gtcnt ++;
                }
                gtbg /= gtcnt;
                gtba /= gtcnt;

                Init::Result closeform_result;
                std::vector<Eigen::Vector3d> nullvec;
                Init::Reprojection_Solution(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg, nullvec);
                countttt++;
            }
        }

        pts_copy = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                (pts_copy, AP.cend(), 500000000);
    }


//    std::cout << "Final Time:" << Init::StaTime / Init::StaCount << std::endl;
//    std::cout << "Final Errorv:" << Init::StaErrorv / Init::StaCount << std::endl;
    std::cout << "Final Error:" << Init::StaError / Init::StaCount << std::endl;
    std::cout << "Final ErrorPercen:" << Init::ErrorPercen / Init::StaCount << std::endl;

    std::cout<<"endl::"<<countttt<<std::endl;
    for(int i = 0; i < 100; i++)
    {
        std::cout << i << ": " << Init::statistical[i] << std::endl;
    }
}




void Test_Reprojection_PAL_Methods(const fs::path &sequence_path)
{
    int countttt = 0;
    fs::path groundtruth_path = sequence_path / "state_groundtruth_estimate0" / "data.csv";
    CHECK(fs::is_regular_file(groundtruth_path)) << "Path not found: " << groundtruth_path.string();

    fs::path imu_path = sequence_path / "imu0" / "data.csv";
    CHECK(fs::is_regular_file(imu_path)) << "Path not found: " << imu_path.string();

    State groundtruth = Read_File<State::value_type>(groundtruth_path.string());
    ImuData imu_data = Read_File<ImuData::value_type>(imu_path.string());

    fs::path point_path = "/home/qjy/Research/VISLAM/trackingpoints/MH03/";
    fs::path line_path = "/home/qjy/Research/VISLAM/trackinglines/MH03/";
    fs::path path_timestamps = "/home/qjy/Dataset/EuRoC/EuRoC_TimeStamps/MH03.txt";

    ImuData::const_iterator im = imu_data.cbegin();
    ImuData::const_iterator im_copy = im;
    bool simul;

    AllPoints AP;
    ReadPoints(point_path, path_timestamps, AP);
    AllPoints::const_iterator pts = AP.cbegin();
    AllPoints::const_iterator pts_copy = pts;

    AllLines AL;
    ReadLines(line_path, path_timestamps, AL);
    AllLines::const_iterator lns = AL.cbegin();
    AllLines::const_iterator lns_copy = lns;

    pts_copy = Find_Iterator<AllPoints::value_type, AllPoints::const_iterator>
            (AP.cbegin(), AP.cend(), lns_copy->timestamp, simul);
    if(!simul) return;

    int cc = 0;
    while(pts_copy != AP.cend()) // 这个是用特征点构成的线闭式求解
    {
        std::cout << cc << std::endl;
        cc++;
        pts = pts_copy;
        lns = lns_copy;
        im_copy = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                (imu_data.cbegin(), imu_data.cend(), pts_copy->timestamp, simul);
        AllPoints PInits;// 用于初始化的点
        AllLines LInits;// 用于初始化的点
        PInits.push_back(*pts);
        LInits.push_back(*lns);

        std::vector<IMU::Preintegrated> pInts;// 用于初始化的预积分

        for(int frame_t = 0; frame_t < Init::FrameNum; frame_t++)
        {
            IMU::Preintegrated pInt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            pInts.push_back(pInt);

            // 先找到与pt对应的IMU
            im = Find_Iterator<ImuData::value_type, ImuData::const_iterator>
                    (im_copy, imu_data.cend(), pts->timestamp, simul);

            if(!simul) break;

            if(im->timestamp > pts->timestamp && im != imu_data.cbegin())
                std::advance(im, -1);// 保证imu不大于pt

            timestamp_t forward_time = (1.0f/Init::InitRate)*1e9;
            AllPoints::const_iterator pts_forw = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                    (pts, AP.cend(), forward_time, simul);

            if (!simul || pts_forw == AP.cend())
            {
                LOG(WARNING) << "Couldn't find next frame for " << pts->timestamp;
                break;
            }

            AllLines::const_iterator lns_forw = Move_Iterator<AllLines::value_type, AllLines::const_iterator>
                    (lns, AL.cend(), forward_time, simul);

            if (!simul || lns_forw == AL.cend())
            {
                LOG(WARNING) << "Couldn't find next frame for " << lns->timestamp;
                break;
            }

            while(im != imu_data.cend() && (im->timestamp < pts_forw->timestamp))
            {
                double delta_t = IMU::dt;
                timestamp_t t1 = im->timestamp;
                const Eigen::Vector3d w1(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a1(im->a_x, im->a_y, im->a_z);
                std::advance(im, 1);
                timestamp_t t2 = im->timestamp;
                const Eigen::Vector3d w2(im->w_x, im->w_y, im->w_z);
                const Eigen::Vector3d a2(im->a_x, im->a_y, im->a_z);

                if(t1 < pts->timestamp)
                    delta_t = (t2 - pts->timestamp)*1e-9;
                else if(t2 > pts_forw->timestamp)
                    delta_t = (pts_forw->timestamp - t1)*1e-9;

                for(auto &p : pInts)
                    p.IntegrateNewMeasurement(0.5*(w1 + w2), 0.5*(a1 + a2), delta_t);
            }
            if (im == imu_data.cend())
            {
                LOG(WARNING) << "IMU stream ended!";
                break;
            }

            PInits.push_back(*pts_forw);
            pts = pts_forw;

            LInits.push_back(*lns_forw);
            lns = lns_forw;
        }

        if(pInts.size() == Init::FrameNum)// 开始优化
        {
            // Points Culling对齐
            std::vector<std::vector<Eigen::Vector3d>> AlignedPts;
            std::vector<std::vector<Eigen::Vector6d>> AlignedLns;

            bool simul1 = false;
            bool simul2 = false;
            bool simul3 = false;

            Points_Culling(PInits, AlignedPts, simul);
            Lines_Culling(LInits, AlignedLns, simul1);


            // 找到对应位置的gt

            State::const_iterator gt1, gt2;
            gt1 = Find_Iterator<State::value_type, State::const_iterator>
                    (groundtruth.cbegin(), groundtruth.cend(), pts_copy->timestamp, simul2);
            gt2 = Find_Iterator<State::value_type, State::const_iterator>
                    (gt1, groundtruth.cend(), pts->timestamp, simul3);

            Eigen::Vector3d gtg;
            Eigen::Vector3d gtv(gt2->v_x, gt2->v_y, gt2->v_z);
            Eigen::Isometry3d InitPos = gt2->pose;

            gtv = InitPos.rotation().transpose()*gtv;
            gtg = InitPos.rotation().transpose()*IMU::GRAVITY_VECTOR;

//            if(simul && simul1 && simul2 && simul3)//找到足够的公共点和线，且正确找到gt的两个端点
            if(simul && simul2 && simul3)//找到足够的公共点和线，且正确找到gt的两个端点
            {
                Eigen::Vector3d gtbg;
                Eigen::Vector3d gtba;
                gtbg.setZero();
                gtba.setZero();
                int gtcnt = 0;
                while(gt1 != gt2)
                {
                    gtbg += Eigen::Vector3d(gt1->bw_x, gt1->bw_y, gt1->bw_z);
                    gtba += Eigen::Vector3d(gt1->ba_x, gt1->ba_y, gt1->ba_z);
                    std::advance(gt1, 1);
                    gtcnt ++;
                }
                gtbg /= gtcnt;
                gtba /= gtcnt;

                Init::Result closeform_result;
                if(simul1)
                    Init::Reprojection_PAL_Solution(closeform_result, AlignedPts, AlignedLns, pInts, Tbc, gtbg, gtba, gtv, gtg);
                else
//                    Init::Reprojection_Solution(closeform_result, AlignedPts, pInts, Tbc, gtbg, gtba, gtv, gtg);
                countttt++;
            }
        }

        pts_copy = Move_Iterator<AllPoints::value_type, AllPoints::const_iterator>
                (pts_copy, AP.cend(), 500000000);
        lns_copy = Move_Iterator<AllLines::value_type, AllLines::const_iterator>
                (lns_copy, AL.cend(), 500000000);
    }


//    std::cout << "Final Time:" << Init::StaTime / Init::StaCount << std::endl;
//    std::cout << "Final Errorv:" << Init::StaErrorv / Init::StaCount << std::endl;
    std::cout << "Final Error:" << Init::StaError / Init::StaCount << std::endl;
    std::cout << "Final ErrorPercen:" << Init::ErrorPercen / Init::StaCount << std::endl;

    std::cout<<"endl::"<<countttt<<std::endl;
    for(int i = 0; i < 100; i++)
    {
        std::cout << i << ": " << Init::statistical[i] << std::endl;
    }
}


int main(int argc, char* argv[]) {

    IMU::Sigma.block<3, 3>(0, 0) = IMU::rate*IMU::ng*IMU::ng * Eigen::Matrix3d::Identity();
    IMU::Sigma.block<3, 3>(3, 3) = IMU::rate*IMU::na*IMU::na * Eigen::Matrix3d::Identity();

    Eigen::Matrix3d Rbc;
    Eigen::Vector3d tbc;

//    Rbc << 0.0148655429818, -0.999880929698, 0.00414029679422,
//            0.999557249008, 0.0149672133247, 0.025715529948,
//            -0.0257744366974, 0.00375618835797, 0.999660727178;
//    tbc << -0.0216401454975, -0.064676986768, 0.00981073058949;

    Rbc << 0, 0, -1,
          -1, 0, 0,
           0, 1, 0;
    tbc << 0.05, 0.04, 0.03;

    Tbc.linear() = Rbc;
    Tbc.translation() = tbc;

    Tcb.linear() = Rbc.transpose();
    Tcb.translation() = - Rbc.transpose()*tbc;

//    Test_All_Methods("/home/qjy/Dataset/EuRoC/MH_01_easy/mav0");
//    Temporal_Calib("/home/qjy/Dataset/EuRoC/MH_01_easy/mav0");
//    Test_Integration("/home/qjy/Dataset/EuRoC/MH_01_easy/mav0");
//    Test_Closeform_Methods("/home/qjy/Dataset/EuRoC/V1_01_easy/mav0");
//    Test_Closeform_Methods_PAL("/home/qjy/Dataset/EuRoC/V1_01_easy/mav0");
    Test_Closeform_Methods_PAL("/home/qjy/Research/VISLAM/SimulData");
//    Test_Simulated_Data("/home/qjy/Research/VISLAM/SimulData");

//    Test_Reprojection_Methods("/home/qjy/Dataset/EuRoC/V1_01_easy/mav0");
//    Test_Reprojection_PAL_Methods("/home/qjy/Dataset/EuRoC/MH_03_medium/mav0");

//    Test_Simulated_Reprojection_Methods("/home/qjy/Research/VISLAM/SimulData");



//    std::vector<int> PtsIndex;

//    std::default_random_engine e(time(0));
//    std::uniform_int_distribution<int> u(0, 99);
//    for(int i = 0; i < 90; i++)
//    {
//        int r = u(e);
//        while(std::find(PtsIndex.begin(), PtsIndex.end(), r) != PtsIndex.end())// 如果已经有这个值了
//            r = (r + 1) % 100;
//        PtsIndex.push_back(r);
//    }

//    std::sort(PtsIndex.begin(), PtsIndex.end());

//    for(auto &aa : PtsIndex)
//    {
//        std::cout << aa << std::endl;
//    }


    /*  测试find distance
    std::vector<int> aa;
    for(int i = 0; i < 10; i++)
    {
        aa.push_back(i);
    }
    auto a = std::find(aa.begin(), aa.end(), 14);
    int i = std::distance(aa.begin(), a);
    cout << i << endl;
*/




    /*  测试find_if
    std::vector<std::pair<int, Eigen::Matrix3d>> aa;
    for(int i = 0; i < 20; i++)
    {
        Eigen::Matrix3d T;
        T.setIdentity();
        T*=i*i;
        aa.push_back(std::pair<int, Eigen::Matrix3d>(i, T));
    }

    {
        auto sd = std::find_if(aa.begin(), aa.end(), [](std::pair<int, Eigen::Matrix3d> &a)
        {
            return a.first == 9;
        });
        sd->first = 100;
        sd->second.setZero();
    }
    for(auto au:aa)
    {
        std::cout << au.first << std::endl;
        std::cout << au.second << std::endl;
    }
*/

    return 0;
}
