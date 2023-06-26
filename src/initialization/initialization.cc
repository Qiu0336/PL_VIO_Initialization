
#include "initialization.h"

namespace Init {
/*

Rot timeline:            |**************************|

imu_w timeline:   |**********************************************|
                  |      |
return time:      |******|

imu_w覆盖范围要大于Rot,返回从imu_w[0]开始到对应Rot[0]的时间延迟
*/

double StaTime = 0;
double StaErrorv = 0;
double StaError = 0;
double StaCount = 0;
double ErrorPercen = 0;
double gravity_error = 0;
std::vector<int> statistical(100, 0);

double Temporal_Calibration(const std::vector<Eigen::Matrix3d> &Rot,
                          const std::vector<Eigen::Vector3d> &imu_w)
{
    Timer timer;
    timer.Start();

    std::vector<Eigen::Vector3d> wg;
    Eigen::Vector3d wg_ave;
    wg_ave.setZero();

    int rangeframes = Rot.size() - 1;

    for(int i = 0; i < rangeframes; i++)// 20hz, 8s
    {
        Eigen::Vector3d wgt = LogSO3(Rot[i].transpose()*Rot[i + 1])*CamRate;
        wg_ave += wgt;
        wg.push_back(wgt);// 20hz
    }
    wg_ave /= rangeframes;
    Eigen::Matrix3d Covwg;
    Covwg.setZero();
    for(int i = 0; i < rangeframes; i++)// 20hz, 8s
    {
        Covwg += (wg[i] - wg_ave)*(wg[i] - wg_ave).transpose();
    }
    Covwg /= rangeframes - 1;

    // 优先队列选取最大的三个数，比较规则在comp中定义
    std::priority_queue<std::pair<double, double>, std::vector<std::pair<double, double>>, comp> Max3;
    Max3.emplace(0, 0);
    Max3.emplace(0, 0);
    Max3.emplace(0, 0);

    for(int t = 0; t + 10*rangeframes <= int(imu_w.size()); t++)// 5ms一次位移
    {
        std::vector<Eigen::Vector3d> wi;
        Eigen::Vector3d wi_ave;
        wi_ave.setZero();

        for(int i = 0; i < rangeframes; i++)// 中值积分计算
        {
            Eigen::Vector3d wit;
            wit.setZero();
            for(int j = 0; j <= 10; j++)
            {
                wit += imu_w[t + 10*i + j];
            }

            wit -= 0.5*imu_w[t + 10*i];
            wit -= 0.5*imu_w[t + 10*i + 10];

            wit /= 10;
            wi_ave += wit;
            wi.push_back(wit);
        }


        wi_ave /= rangeframes;

        Eigen::Matrix3d Covwi;
        Eigen::Matrix3d Covwgi;
        Eigen::Matrix3d Covwig;
        Covwi.setZero();
        Covwgi.setZero();
        Covwig.setZero();
        for(int i = 0; i < rangeframes; i++)// 20hz, 8s
        {
            Covwi  += (wi[i] - wi_ave)*(wi[i] - wi_ave).transpose();
            Covwig += (wi[i] - wi_ave)*(wg[i] - wg_ave).transpose();
            Covwgi += (wg[i] - wg_ave)*(wi[i] - wi_ave).transpose();
        }
        Covwi  /= rangeframes - 1;
        Covwig /= rangeframes - 1;
        Covwgi /= rangeframes - 1;


        double rw = std::sqrt((Covwi.inverse()*Covwig*Covwg.inverse()*Covwgi).trace() / 3.0f);

        if(rw > Max3.top().second)
        {
            double offset = t*0.005f;
            Max3.emplace(offset, rw);
            Max3.pop();
        }
    }

    double finalres;

    double x0 = Max3.top().first;
    double y0 = Max3.top().second;
    Max3.pop();
    double x1 = Max3.top().first;
    double y1 = Max3.top().second;
    Max3.pop();
    double x2 = Max3.top().first;
    double y2 = Max3.top().second;
    Max3.pop();

    finalres = 0.5*((y0 - y1)*x2*x2 + (y1 - y2)*x0*x0 + (y2 - y0)*x1*x1)
            /((y0 - y1)*x2 + (y1 - y2)*x0 + (y2 - y0)*x1);

    std::cout << "Time:" << timer.ElapsedNanoSeconds()/10e6 << ", ";

    return finalres;
}



void Init_ORBSLAM3_Method(const std::vector<Input> &input, Result &result, double &cost, double init_scale,
                          const Eigen::Isometry3d &Tcb, bool use_prior, double prior)
{
    Timer timer;
    timer.Start();

    std::vector<double*> pointers;
    std::vector<double**> pointers2;

    // Global parameters 先定义需要初始化的参数
    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    double* ba_ptr = new double[3];
    pointers.push_back(ba_ptr);
    Eigen::Map<Eigen::Vector3d> ba(ba_ptr);
    ba.setZero();

    double* Rwg_ptr = new double[9];
    pointers.push_back(Rwg_ptr);
    Eigen::Map<Eigen::Matrix3d> Rwg(Rwg_ptr);
    Rwg.setIdentity();

    double* s_ptr = new double[1];
    pointers.push_back(s_ptr);
    s_ptr[0] = init_scale;

    // Local parameters (for each keyframe)
    double* v0_ptr = new double[3];
    pointers.push_back(v0_ptr);
    Eigen::Map<Eigen::Vector3d> v0(v0_ptr);
    v0.setZero();

    double** parameters = new double*[6];
    pointers2.push_back(parameters);
    parameters[0] = v0_ptr;  // v1
    parameters[1] = nullptr; // v2
    parameters[2] = bg_ptr;  // bg
    parameters[3] = ba_ptr;  // ba
    parameters[4] = Rwg_ptr; // Rwg
    parameters[5] = s_ptr;   // scale

    ceres::Problem problem;// 构建最小二乘问题

    Eigen::Vector3d dirG;
    dirG.setZero();

    for (unsigned i = 0; i < input.size(); ++i)
    {
        // 获取input的数据，包括相邻两针的GT位姿和测量数据的预积分值
        const Eigen::Isometry3d &T1 = input[i].T1;
        const Eigen::Isometry3d &T2 = input[i].T2;
        const std::shared_ptr<IMU::Preintegrated> pInt = input[i].pInt;

        double* v1_ptr = parameters[0];
        Eigen::Map<Eigen::Vector3d> v1(v1_ptr);

        double* v2_ptr = new double[3];
        pointers.push_back(v2_ptr);
        Eigen::Map<Eigen::Vector3d> v2(v2_ptr);

        v2 = (T2.translation() - T1.translation())/pInt->dT;// 计算真实速度
        v1 = v2;

        parameters[1] = v2_ptr;

        // Rwg initialization，得到主要为重力g产生的速度
        dirG -= T1.linear()*pInt->GetUpdatedDeltaVelocity();

        // ceres损失函数构造，这里是使用给定的解析式求导
        ceres::CostFunction* cost_function = new InertialCostFunction(pInt,
                                                          T1.linear(), T1.translation(),
                                                          T2.linear(), T2.translation(),
                                                          Tcb);
        problem.AddResidualBlock(cost_function, nullptr, parameters, 6);// 添加残差块
        //其中nullptr为核函数，parameters为所有待优化参数，6为参数维度

        double** parameters_ = new double*[6];
        pointers2.push_back(parameters_);
        parameters_[0] = parameters[1];
        parameters_[1] = nullptr;
        parameters_[2] = bg_ptr;// 除了速度，以下这些量都是常量，在初始化过程中不更新
        parameters_[3] = ba_ptr;
        parameters_[4] = Rwg_ptr;// 重力的方向是相对世界坐标系下的旋转而言，是个常量
        parameters_[5] = s_ptr;

        parameters = parameters_;
    }

    if (use_prior)// acc bias先验
    {
        ceres::CostFunction* prior_cost_function = new BiasPriorCostFunction(prior);//prior先验权重10e5
        problem.AddResidualBlock(prior_cost_function, nullptr, ba_ptr);
    }

    // Initialize Rwg estimate// 平均重力方向为初始Rwg方向
    dirG = dirG.normalized();
    const Eigen::Vector3d gI = IMU::GRAVITY_VECTOR.normalized();
    const Eigen::Vector3d v = gI.cross(dirG);
    const double cos_theta = gI.dot(dirG);
    const double theta = std::acos(cos_theta);
    Rwg = ExpSO3(v*theta/v.norm());

    // Add local parameterizations// 这里是设置待优化参数的更新方式
    // Local parameters 避免过参数化问题，允许用户自定义参数化
    GravityParameterization* gravity_local_parameterization = new GravityParameterization;
    problem.SetParameterization(Rwg_ptr, gravity_local_parameterization);

    ScaleParameterization* scale_local_parameterization = new ScaleParameterization;
    problem.SetParameterization(s_ptr, scale_local_parameterization);

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);// 开始求解

    cost = summary.final_cost;
    timer.Pause();

    bool converged = (summary.termination_type == ceres::CONVERGENCE);
    if (converged)// 成功收敛，优化完成
    {
        result.success = true;
        result.solve_ns = timer.ElapsedNanoSeconds();
        result.scale = s_ptr[0];
        result.bias_g = bg;
        result.bias_a = ba;
        result.gravity = Rwg*IMU::GRAVITY_VECTOR;
    }
    else// 优化失败
    {
        result.success = false;
        result.solve_ns = timer.ElapsedNanoSeconds();
    }

    // Free memory 释放内存
    for (double* ptr : pointers)
        delete[] ptr;
    for (double** ptr : pointers2)
        delete[] ptr;
}

void Init_Analytical_Method(const std::vector<Input> &input, Result &result,
                            const Eigen::Isometry3d &Tcb)
{
    Timer timer;
    timer.Start();

//*******************Firstly estimate gyro bias:

    double** parameters = new double*[1];
    parameters[0] = new double[3];
    Eigen::Map<Eigen::Vector3d> bg(parameters[0]);
    bg.setZero();

    ceres::Problem problem;
    for (unsigned i = 0; i < input.size(); ++i)
    {
        const Eigen::Isometry3d &T1 = input[i].T1;
        const Eigen::Isometry3d &T2 = input[i].T2;
        const std::shared_ptr<IMU::Preintegrated> pInt = input[i].pInt;

        ceres::CostFunction* cost_function =
                new GyroscopeBiasCostFunction(pInt, T1.linear()*Tcb.rotation(),
                                              T2.linear()*Tcb.rotation());

        problem.AddResidualBlock(cost_function, nullptr, parameters, 1);
    }
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    result.solve_ns = timer.ElapsedNanoSeconds();

    bool converged = (summary.termination_type == ceres::CONVERGENCE);

    delete[] parameters[0];
    delete[] parameters;

    if (!converged)
    {
        result.solve_ns = timer.ElapsedNanoSeconds();
        result.success = false;
        return;
    }
    result.bias_g = bg;

//*******************Then estimate the rest parameters:

    constexpr int n = 7;
    constexpr int q = 4;

    Eigen::MatrixXd M(n, n);
    M.setZero();

    Eigen::VectorXd m(n);
    m.setZero();

    double Q = 0.;

    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    //先计算cost function中的M，m，Q
    for (unsigned i = 1; i < input.size(); ++i)
    {
        const Eigen::Isometry3d &T1 = input[i-1].T1;
        const Eigen::Isometry3d &T2 = input[i].T1;
        const Eigen::Isometry3d &T3 = input[i].T2;
        const IMU::Preintegrated &pInt12 = *(input[i-1].pInt);
        const IMU::Preintegrated &pInt23 = *(input[i].pInt);

        Eigen::Matrix3d R1 = T1.linear()*Tcb.linear(); // Rwb
        Eigen::Matrix3d R2 = T2.linear()*Tcb.linear(); // Rwb

        Eigen::Matrix3d A = R1/pInt12.dT;
        Eigen::Matrix3d B = R2/pInt23.dT;

        Eigen::MatrixXd M_k(3, n);// 先计算M_k，3*n的维度(n=7)
        M_k.setZero();

        M_k.col(0) = (T3.translation() - T2.translation())/pInt23.dT
        - (T2.translation() - T1.translation())/pInt12.dT;
        M_k.block<3, 3>(0, 1) = A*pInt12.JPa - B*pInt23.JPa - R1*pInt12.JVa;
        M_k.block<3, 3>(0, q) = -0.5*(pInt12.dT + pInt23.dT)*Eigen::Matrix3d::Identity();

        Eigen::Vector3d pi_k;// 再计算π_k，3*1的维度
        pi_k = B*pInt23.GetDeltaPosition(bg, ba) - A*pInt12.GetDeltaPosition(bg, ba) + R1*pInt12.GetDeltaVelocity(bg, ba)
        + (T2.linear()-T1.linear())*Tcb.translation()/pInt12.dT
        - (T3.linear()-T2.linear())*Tcb.translation()/pInt23.dT;

        // 协方差矩阵，根据π中预积分的协方差相加得到
        Eigen::Matrix3d Covariance;
        Covariance  = A*pInt12.C.block<3, 3>(6, 6)*A.transpose();
        Covariance += B*pInt23.C.block<3, 3>(6, 6)*B.transpose();
        Covariance += T1.linear()*pInt12.C.block<3, 3>(3, 3)*T1.linear().transpose();

        Eigen::Matrix3d Information = Covariance.inverse();
//        Eigen::Matrix3d Information = Eigen::Matrix3d::Identity();

        M +=  M_k.transpose()*Information*M_k;
        m += -2.*M_k.transpose()*Information*pi_k;
        Q +=  pi_k.transpose()*Information*pi_k;
    }

    // Solve，开始求解
    Eigen::Matrix4d A = 2.*M.block<4, 4>(0, 0);
    Eigen::MatrixXd Bt = 2.*M.block<3, 4>(q, 0);
    Eigen::MatrixXd BtAi = Bt*A.inverse();

    Eigen::Matrix3d D = 2.*M.block<3, 3>(q, q);
    Eigen::Matrix3d S = D - BtAi*Bt.transpose();

    // 得到A,B,S,D矩阵

    // 以下是求解论文中等式(42)左边每一项的系数
    Eigen::Matrix3d Sa = S.determinant()*S.inverse();// S的伴随矩阵
    Eigen::Matrix3d U = S.trace()*Eigen::Matrix3d::Identity() - S;

    Eigen::Vector3d v1 = BtAi*m.head<q>();
    Eigen::Vector3d m2 = m.tail<3>();

    Eigen::Matrix3d X; Eigen::Vector3d Xm2;

    // X = I
    const double c4 = 16.*(v1.dot(  v1) - 2.*v1.dot( m2) + m2.dot( m2));

    X = U; Xm2 = X*m2;
    const double c3 = 16.*(v1.dot(X*v1) - 2.*v1.dot(Xm2) + m2.dot(Xm2));

    X = 2.*Sa + U*U; Xm2 = X*m2;
    const double c2 =  4.*(v1.dot(X*v1) - 2.*v1.dot(Xm2) + m2.dot(Xm2));

    X = Sa*U + U*Sa; Xm2 = X*m2;
    const double c1 =  2.*(v1.dot(X*v1) - 2.*v1.dot(Xm2) + m2.dot(Xm2));

    X = Sa*Sa; Xm2 = X*m2;
    const double c0 =     (v1.dot(X*v1) - 2.*v1.dot(Xm2) + m2.dot(Xm2));

    // 以下是求解P(λ)的解析式，这里P(λ) = 8λ^3 + 4t1λ^2 + 2t2λ + t3
    const double s00 = S(0, 0), s01 = S(0, 1), s02 = S(0, 2);
    const double s11 = S(1, 1), s12 = S(1, 2), s22 = S(2, 2);

    const double t1 = s00 + s11 + s22;
    const double t2 = s00*s11 + s00*s22 + s11*s22
                       - std::pow(s01, 2) - std::pow(s02, 2) - std::pow(s12, 2);
    const double t3 = s00*s11*s22 + 2.*s01*s02*s12
                       - s00*std::pow(s12, 2) - s11*std::pow(s02, 2) - s22*std::pow(s01, 2);

    // 首先赋值为P(λ)的解析式产生的6次方程的项
    Eigen::VectorXd coeffs(7);
    coeffs << 64.,
              64.*t1,
              16.*(std::pow(t1, 2) + 2.*t2),
              16.*(t1*t2 + t3),
               4.*(std::pow(t2, 2) + 2.*t1*t3),
               4.*t3*t2,
              std::pow(t3, 2);

    // 然后减去论文中等式(42)左边产生的项
    const double G2i = 1. / std::pow(IMU::GRAVITY_MAGNITUDE, 2);
    coeffs(2) -= c4*G2i;
    coeffs(3) -= c3*G2i;
    coeffs(4) -= c2*G2i;
    coeffs(5) -= c1*G2i;
    coeffs(6) -= c0*G2i;

    Eigen::VectorXd real, imag;
    // 高次方程求根，coeffs为高次方程系数，real和imag为求出根的实部和虚部
    if (!FindPolynomialRootsCompanionMatrix(coeffs, &real, &imag))
    {
        result.success = false;
        result.solve_ns = timer.ElapsedNanoSeconds();
        return;
    }

    // 这里找到有效的λ值，
    Eigen::VectorXd lambdas = real_roots(real, imag);// 筛选出实数根
    if (lambdas.size() == 0)
    {
        result.success = false;
        result.solve_ns = timer.ElapsedNanoSeconds();
        return;
    }

    Eigen::MatrixXd W(n, n);
    W.setZero();
    W.block<3, 3>(q, q) = Eigen::Matrix3d::Identity();

    Eigen::VectorXd solution;
    double min_cost = std::numeric_limits<double>::max();
    for (Eigen::VectorXd::Index i = 0; i < lambdas.size(); ++i)
    {
        const double lambda = lambdas(i);

        Eigen::FullPivLU<Eigen::MatrixXd> lu(2.*M + 2.*lambda*W);
        Eigen::VectorXd x_ = -lu.inverse()*m;

        // 在诸多λ解中，找寻cost function最小的一个
        double cost = x_.transpose()*M*x_;
        cost += m.transpose()*x_;
        cost += Q;

        if (cost < min_cost)
        {
            solution = x_;
            min_cost = cost;
        }
    }

    const double constraint = solution.transpose()*W*solution;// 最后来一波重力约束验证
    if (solution[0] < 1e-3 || constraint < 0.
        || std::abs(std::sqrt(constraint) - IMU::GRAVITY_MAGNITUDE)/IMU::GRAVITY_MAGNITUDE > 1e-3)
    {
        result.solve_ns = timer.ElapsedNanoSeconds();
        result.success = false;
        return;
    }

    // 求解成功，赋值
    result.solve_ns = timer.ElapsedNanoSeconds();
    result.success = true;
    result.scale = solution[0];
    result.bias_a = solution.segment<3>(1);
    result.gravity = solution.segment<3>(4);

}


void Init_Interpolation_Method(const std::vector<Input> &input, Result &result,
                               const std::vector<Eigen::Vector3d> &imu_w,
                               const std::vector<Eigen::Vector3d> &imu_a,
                               const std::vector<Eigen::Isometry3d> &GTPose,
                               const std::vector<Eigen::Vector3d> &GTVel,
                               const Eigen::Isometry3d &Tcb)
{

    Timer timer;
    timer.Start();

    const double dt = IMU::dt;
    const double rate = IMU::rate;

    int nframes = input.size();
    BSpline_Cubic BSpTrans(nframes + 1);// 平移B样条插值

    std::vector<Eigen::Vector3d> pos;
    pos.push_back(input[0].T1.translation());
    for(int k = 0; k < nframes; k++)
        pos.push_back(input[k].T2.translation());
    BSpTrans.Set_ControlPoints(pos);

    std::vector<Eigen::Matrix3d> RotMat;// 所有旋转矩阵
    std::vector<Eigen::Vector3d> PosVec;// 所有平移坐标
    std::vector<Eigen::Vector3d> VelVec;// 所有线速度

    double t_inv = CamRate/(BSpTrans.frames - 1);

    double delta_t = 1.0f/((BSpTrans.frames - 1)*interframes);// 10为相机频率/IMU频率
    double start_time = 2.0f/(BSpTrans.frames - 1);// 前后舍去2帧
    double end_time   = 1 - start_time;

    int count = 0;
    for(double t = start_time; t < end_time + 0.5*delta_t; t += delta_t)
    {
        PosVec.push_back(BSpTrans.Get_Interpolation(t));
        VelVec.push_back(t_inv*BSpTrans.Get_Interpolation_FirstOrder(t));
//        PosVec.push_back(GTPose[count].translation());
//        VelVec.push_back(GTVel[count]);
//        RotMat.push_back(GTPose[count].rotation());
        count ++;
    }

    // 测量真值与插值的误差
//    count = 0;
//    double er_pos = 0;
//    double er_vel = 0;
//    for(double t = start_time; t < end_time + 0.5*delta_t; t += delta_t)
//    {
//        Eigen::Vector3d Pos = BSpTrans.Get_Interpolation(t);
//        Eigen::Vector3d Vel = t_inv*BSpTrans.Get_Interpolation_FirstOrder(t);
//        er_pos += (Pos - GTPose[count].translation()).norm();
//        er_vel += (Vel - GTVel[count]).norm();
//        count ++;
//    }
//    std::cout << "error_pos:" << er_pos << ", error_vel:" << er_vel << std::endl;

    int size = PosVec.size();
    // bg估计

    std::vector<Eigen::Vector3d> bg_Vec;
    Eigen::Vector3d bg;
    bg.setZero();

    int interNums = interframes;// 两帧之间待插值个数
    count = 0;
    for(int t = 0, rotindex = 0; t < size - interNums; t += interNums, rotindex++)
    {
        std::vector<Eigen::Vector3d> raw_w;
        for(int i = 0; i <= interNums; i++)
            raw_w.push_back(imu_w[t + i]);

        std::vector<Eigen::Matrix3d> Ri;
        Ri = Rot_Interpolation(input[rotindex + 2].T1.rotation(), input[rotindex + 2].T2.rotation(), raw_w, dt);

        for(int i = 0; i < interNums; i++)
        {
            Eigen::Vector3d bg_once = 0.5*(raw_w[i] + raw_w[i + 1]) - LogSO3(Ri[i].transpose()*Ri[i + 1])*rate;
            bg_Vec.push_back(bg_once);
            bg += bg_once;
            count ++;
            RotMat.push_back(Ri[i]);
        }
    }


//    count = 0;
//    for(int i = 0; i + 1 < RotMat.size(); i++)
//    {
//        Eigen::Vector3d bg_once = 0.5*(imu_w[i] + imu_w[i + 1]) - LogSO3(RotMat[i].transpose()*RotMat[i + 1])*rate;
//        bg_Vec.push_back(bg_once);
//        bg += bg_once;
//        count ++;
//    }

    bg /= count;


    // 对bg做RANSAC：


//    double max = 0; double min = 1;
//    for(int i = 0; i < bg_Vec.size(); i++)
//    {
//        double tmp = (bg_Vec[i] - bg).norm();
//        max = std::max(tmp, max);
//        min = std::min(tmp, min);
//    }
//        std::cout << "bg max:" << max << ", bg min:" << min << std::endl;

    for(int k = 0; k < 1; k++)
    {
        Eigen::Vector3d bgRef;
        bgRef.setZero();
        count = 0;
        for(std::vector<Eigen::Vector3d>::iterator it = bg_Vec.begin(); it != bg_Vec.end(); it ++)
        {
            double tmp = (*it - bg).norm();
            if(tmp > 0.001)
                continue;
            bgRef += *it;
            count ++;
        }
        if(count == 0)
            break;
        bg = bgRef/count;
    }

    // s,g估计
    double s;
    Eigen::Vector3d g;
    Eigen::Vector3d ba;
    ba.setZero();

//    Eigen::MatrixXd A(4, 4);
//    Eigen::VectorXd b(4);
//    A.setZero();
//    b.setZero();

    Eigen::VectorXd x(4);
    x.setZero();

    // 用于验证
    Eigen::MatrixXd Valid_A(7, 7);
    Eigen::VectorXd Valid_b(7);
    Valid_A.setZero();
    Valid_b.setZero();


    Eigen::Vector3d SigmaV;
    Eigen::Vector3d SigmaRa;
    Eigen::Matrix3d SigmaR;
    SigmaV.setZero();
    SigmaRa.setZero();
    SigmaR.setZero();

    count = 0;
    int batch = 30;
//    std::vector<Eigen::Vector3d> Vec_SigmaV;
//    std::vector<Eigen::Vector3d> Vec_SigmaRa;
//    std::vector<Eigen::Matrix3d> Vec_SigmaR;

//    Eigen::MatrixXd Mkk(3, 7);
//    Eigen::Vector3d pikk;
//    Mkk.setZero();
//    pikk.setZero();


//    // step = 1
//    for(count = 0; count + batch < size; count ++)
//    {
//        Eigen::MatrixXd Mk(3, 7);
//        Eigen::Vector3d pik;

//        if(count == 0)
//        {
//            for(int i = 0; i < batch; i++)
//            {
//                Eigen::Matrix3d Ri = RotMat[i];
//                SigmaV += VelVec[i];
//                SigmaRa += Ri*imu_a[i];
//                SigmaR += Ri;
//            }
//        }
//        else {
//            SigmaV -= VelVec[count - 1];
//            Eigen::Matrix3d Ri = RotMat[count - 1];
//            SigmaRa -= Ri*imu_a[count - 1];
//            SigmaR -= Ri;

//            SigmaV += VelVec[count + batch - 1];
//            Ri = RotMat[count + batch - 1];
//            SigmaRa += Ri*imu_a[count + batch - 1];
//            SigmaR += Ri;
//        }

//        Mk.col(0) = (PosVec[count] - PosVec[count + batch]) + SigmaV*dt;// s前面的系数
//        Mk.block<3, 3>(0, 1) = 0.5*batch*Eigen::Matrix3d::Identity()*dt*dt;// g前面的系数
//        Mk.block<3, 3>(0, 4) = - 0.5*SigmaR*dt*dt;// ba前面的系数
//        pik = - 0.5*SigmaRa*dt*dt;

////        Eigen::MatrixXd Mk_seg(3, 4);
////        Mk_seg = Mk.block<3, 4>(0, 0);

////        A += Mk_seg.transpose()*Mk_seg;
////        b += Mk_seg.transpose()*pik;

////        Mkk += Mk;
////        pikk += pik;

//        Valid_A += Mk.transpose()*Mk;
//        Valid_b += Mk.transpose()*pik;

////        Vec_SigmaV.push_back(SigmaV);
////        Vec_SigmaRa.push_back(SigmaRa);
////        Vec_SigmaR.push_back(SigmaR);
//    }


    // step = 3
//    for(count = 0; count + batch < size; count += 4)
//    {
//        Eigen::MatrixXd Mk(3, 7);
//        Eigen::Vector3d pik;

//        if(count == 0)
//        {
//            for(int i = 0; i < batch; i++)
//            {
//                SigmaV += VelVec[i];
//                SigmaRa += RotMat[i]*imu_a[i];
//                SigmaR += RotMat[i];
//            }
//        }
//        else {
//            SigmaV -= VelVec[count - 1];
//            Eigen::Matrix3d Ri = RotMat[count - 1];
//            SigmaRa -= Ri*imu_a[count - 1];
//            SigmaR -= Ri;

//            SigmaV -= VelVec[count - 2];
//            Ri = RotMat[count - 2];
//            SigmaRa -= Ri*imu_a[count - 2];
//            SigmaR -= Ri;

//            SigmaV -= VelVec[count - 3];
//            Ri = RotMat[count - 3];
//            SigmaRa -= Ri*imu_a[count - 3];
//            SigmaR -= Ri;

//            SigmaV -= VelVec[count - 4];
//            Ri = RotMat[count - 4];
//            SigmaRa -= Ri*imu_a[count - 4];
//            SigmaR -= Ri;

//            SigmaV += VelVec[count + batch - 1];
//            Ri = RotMat[count + batch - 1];
//            SigmaRa += Ri*imu_a[count + batch - 1];
//            SigmaR += Ri;

//            SigmaV += VelVec[count + batch - 2];
//            Ri = RotMat[count + batch - 2];
//            SigmaRa += Ri*imu_a[count + batch - 2];
//            SigmaR += Ri;

//            SigmaV += VelVec[count + batch - 3];
//            Ri = RotMat[count + batch - 3];
//            SigmaRa += Ri*imu_a[count + batch - 3];
//            SigmaR += Ri;

//            SigmaV += VelVec[count + batch - 4];
//            Ri = RotMat[count + batch - 4];
//            SigmaRa += Ri*imu_a[count + batch - 4];
//            SigmaR += Ri;
//        }


//        Mk.col(0) = (PosVec[count] - PosVec[count + batch]) + SigmaV*dt;// s前面的系数
//        Mk.block<3, 3>(0, 1) = 0.5*batch*Eigen::Matrix3d::Identity()*dt*dt;// g前面的系数
//        Mk.block<3, 3>(0, 4) = - 0.5*SigmaR*dt*dt;// ba前面的系数
//        pik = - 0.5*SigmaRa*dt*dt;
//        Valid_A += Mk.transpose()*Mk;
//        Valid_b += Mk.transpose()*pik;
//    }





    for(count = 0; count + batch + 1 < size; count ++)
    {
        Eigen::MatrixXd Mk(3, 7);
        Eigen::Vector3d pik;

        if(count == 0)
        {
            for(int i = 0; i <= batch; i++)
            {
                SigmaRa += RotMat[i]*imu_a[i];
                SigmaR += RotMat[i];
            }
            SigmaRa = SigmaRa - 0.5*RotMat[0]*imu_a[0] - 0.5*RotMat[batch]*imu_a[batch];
            SigmaR = SigmaR - 0.5*RotMat[0] - 0.5*RotMat[batch];
        }
        else {
            int ind = count - 1;
            SigmaRa = SigmaRa - 0.5*RotMat[ind]*imu_a[ind] - 0.5*RotMat[ind + 1]*imu_a[ind + 1];
            SigmaR = SigmaR - 0.5*RotMat[ind] - 0.5*RotMat[ind + 1];

            ind = count + batch - 1;
            SigmaRa = SigmaRa + 0.5*RotMat[ind]*imu_a[ind] + 0.5*RotMat[ind + 1]*imu_a[ind + 1];
            SigmaR = SigmaR + 0.5*RotMat[ind] + 0.5*RotMat[ind + 1];
        }

        Mk.col(0) = (PosVec[count + 1] - PosVec[count])
                    - (PosVec[count + batch + 1] - PosVec[count + batch]);// s前面的系数
        Mk.block<3, 3>(0, 1) = batch*Eigen::Matrix3d::Identity()*dt*dt;// g前面的系数
        Mk.block<3, 3>(0, 4) = - SigmaR*dt*dt;// ba前面的系数
        pik = - SigmaRa*dt*dt;

        Valid_A += Mk.transpose()*Mk;
        Valid_b += Mk.transpose()*pik;
    }




    Eigen::MatrixXd ASeg(7, 4);
    ASeg = Valid_A.block<7, 4>(0, 0);

    Eigen::MatrixXd ASeg2(7, 4);
    ASeg2.col(0) = Valid_A.col(0);
    ASeg2.block<7, 3>(0, 1) = Valid_A.block<7, 3>(0, 4);

    x = ASeg.colPivHouseholderQr().solve(Valid_b);

//    Eigen::Vector4d xgt(1.0, 0.0, 0.0, -9.81);
//    std::cout << "????::" << (ASeg*xgt - Valid_b).norm() << std::endl;
//    std::cout << "????::" << (ASeg*x - Valid_b).norm() << std::endl;

    s = x(0);
    g = x.segment<3>(1);
    g = g.normalized()*IMU::GRAVITY_MAGNITUDE;
    Eigen::Vector4d xseg;
    xseg = ASeg2.colPivHouseholderQr().solve(Valid_b - Valid_A.block<7, 3>(0, 1)*g);
    Eigen::VectorXd x0(7);
    x0(0) = xseg(0);
    x0.segment(1, 3) = g;
    x0.segment(4, 3) = xseg.segment<3>(1);
//    x0(0) = x(0);
//    x0.segment(1, 3) = g;
//    x0.segment(4, 3) = Eigen::Vector3d::Zero();
    Eigen::VectorXd ErrorX0(3);
    ErrorX0 = x0.segment(4, 3);
    double err0 = (Valid_A*x0 - Valid_b).norm()*10e8;
    std::cout << "x0 :" << std::abs(x0(0) - 1) << ", "
              << 180.*std::acos(g.normalized().dot(IMU::GRAVITY_VECTOR.normalized()))/EIGEN_PI
              << ", Error0:" << err0 << ", ErrorX0: " << ErrorX0.norm() << std::endl;

    Eigen::VectorXd x7(7);
    x7 = Valid_A.ldlt().solve(Valid_b);

    s = x7(0);
    g = x7.segment<3>(1);
    ba = x7.segment<3>(4);
    g = g.normalized()*IMU::GRAVITY_MAGNITUDE;
    Eigen::Vector4d xseg2;
    xseg2 = ASeg2.colPivHouseholderQr().solve(Valid_b - Valid_A.block<7, 3>(0, 1)*g);
    Eigen::VectorXd x00(7);
    x00(0) = xseg2(0);
    x00.segment(1, 3) = g;
    x00.segment(4, 3) = xseg2.segment<3>(1);
//    x00(0) = x7(0);
//    x00.segment(1, 3) = g;
//    x00.segment(4, 3) = x7.segment<3>(4);
    Eigen::VectorXd ErrorX1(3);
    ErrorX1 = x00.segment(4, 3);
    double err00 = (Valid_A*x00 - Valid_b).norm()*10e8;
//    double er00 = (Mkk*x00 - pikk).norm()*10e8;
    std::cout << "x00:" << std::abs(x00(0) - 1) << ", "
              << 180.*std::acos(g.normalized().dot(IMU::GRAVITY_VECTOR.normalized()))/EIGEN_PI
              << ", Error00:" << err00 << ", ErrorX1: " << ErrorX1.norm() << std::endl;


/*
//    x = A.ldlt().solve(b);
    s = x(0);
    g = x.segment<3>(1);

    std::cout << "before:" << std::abs(s - 1) << ", "
              << 180.*std::acos(g.normalized().dot(IMU::GRAVITY_VECTOR.normalized()))/EIGEN_PI << std::endl;
    Eigen::VectorXd Res1(7);
    Res1(0) = s;
    Res1.segment(1, 3) = g;
    Res1.segment(4, 3) = Eigen::Vector3d::Zero();
    double err1 = (Valid_A*Res1 - Valid_b).norm()*10e8;

    // 优化ba：
    Eigen::Vector3d gw = g.normalized();
    Eigen::Vector3d gi = IMU::GRAVITY_VECTOR.normalized();
    Eigen::Vector3d v = (gi.cross(gw)).normalized();
    double theta = atan2((gi.cross(gw)).norm(), gi.dot(gw));
    Eigen::Matrix3d Rwi = ExpSO3(v*theta);

    g = gw*IMU::GRAVITY_MAGNITUDE;
    Eigen::Matrix3d deltaG = - 0.5*batch*dt*dt*IMU::GRAVITY_MAGNITUDE*Rwi*Skew(gi);

    Eigen::MatrixXd AA(6, 6);
    Eigen::VectorXd bb(6);
    Eigen::VectorXd xx(6);
    AA.setZero();
    bb.setZero();
    xx.setZero();

    for(count = 0; count + batch < size; count ++)
    {
        Eigen::MatrixXd M_k(3, 6);
        Eigen::Vector3d pi_k;
        M_k.col(0) = (PosVec[count] - PosVec[count + batch]) + Vec_SigmaV[count]*dt;// s前面的系数
        M_k.col(1) = deltaG.col(0);// g前面的系数
        M_k.col(2) = deltaG.col(1);// g前面的系数
        M_k.block<3, 3>(0, 3) = - 0.5*dt*dt*Vec_SigmaR[count];// ba前面的系数
        pi_k = - 0.5*Vec_SigmaRa[count]*dt*dt - 0.5*batch*g*dt*dt - s*M_k.col(0);
//        pi_k = - 0.5*Vec_SigmaRa[count]*dt*dt - 0.5*batch*g*dt*dt
//                - s*M_k.col(0) + 0.5*dt*dt*Vec_SigmaR[count]*ba;

        AA += M_k.transpose()*M_k;
        bb += M_k.transpose()*pi_k;
    }

    xx = AA.ldlt().solve(bb);
    Rwi = Rwi*ExpSO3(xx(1), xx(2), 0);
    g = Rwi*IMU::GRAVITY_VECTOR;
    s += xx(0);
    ba = xx.segment<3>(3);

    Eigen::VectorXd Res2(7);
    Res2(0) = s;
    Res2.segment(1, 3) = g;
    Res2.segment(4, 3) = ba;
    double err2 = (Valid_A*Res2 - Valid_b).norm()*10e8;


    std::cout << "Error1: " << err1 << ", Error2: " << err2 << std::endl;
//    std::cout << "Error2: " << err2 << std::endl;
    std::cout << "after:" << std::abs(s - 1) << ", " << (ba-gtba).norm() << ", "
              << 180.*std::acos(g.normalized().dot(IMU::GRAVITY_VECTOR.normalized()))/EIGEN_PI << std::endl;
*/
    Eigen::VectorXd FinalRes(7);
//    FinalRes = x0;

    if(x00.segment(4, 3).norm() < 0.8)
        FinalRes = x00;
    else
        FinalRes = x0;

//    if(err00 < err0) // ba优化成功
//        FinalRes = x00;
//    else
//        FinalRes = x0;

    result.solve_ns = timer.ElapsedNanoSeconds();
    result.success = true;
    result.bias_g = bg;
    result.scale = FinalRes(0);
//    result.scale = x(0);
    result.bias_a = FinalRes.segment(4, 3);
    result.gravity = FinalRes.segment(1, 3);
//    result.gravity = x.segment(1, 3);

}


void CloseForm_Solution(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg,
                        const std::vector<IMU::Preintegrated> &pIntsgt)
{
    int PNum = PointNum;
    int FrmNum = FrameNum + 1;

//    Timer timer;
//    timer.Start();



    // 加上g一起计算·改，用AtA计算
/*
    Eigen::MatrixXd A(6 + PNum*FrmNum, 6 + PNum*FrmNum);
    Eigen::VectorXd X(6 + PNum*FrmNum);
    Eigen::VectorXd B(6 + PNum*FrmNum);
    A.setZero();
    B.setZero();

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rbc = Tbc.rotation();
    Eigen::Vector3d tbc = Tbc.translation();
    double dt = 1.0f / InitRate;

    for(int i = 0; i < PNum; i++)
    {
        for(int j = 1; j < FrmNum; j++)
        {
            double delt = j*dt;
            Eigen::MatrixXd SubA(3, 6 + PNum*FrmNum);
            SubA.setZero();
            SubA.block(0, 0, 3, 3) = 0.5*delt*delt*I;
            SubA.block(0, 3, 3, 3) = delt*I;
            SubA.col(i*FrmNum + 6) = - Rbc*AlignedPts[0][i];
            SubA.col(i*FrmNum + j + 6) = pIntsgt[j - 1].dR*Rbc*AlignedPts[j][i];
            Eigen::Vector3d SubB = - pIntsgt[j - 1].dS - (pIntsgt[j - 1].dR - I)*tbc;

            A += SubA.transpose()*SubA;
            B += SubA.transpose()*SubB;
        }
    }

    X = A.ldlt().solve(B);

    Eigen::Vector3d g = X.head(3);
    Eigen::Vector3d v = X.segment(3, 3);

    std::cout << X.transpose() << std::endl << std::endl;
*/


    // 加上g一起计算·改，用论文中方式计算,时间会更快

    Eigen::Vector3d g;
    Eigen::Vector3d v;

    Eigen::MatrixXd A(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd X(6 + PNum*FrmNum);
    Eigen::VectorXd S(3*(FrmNum - 1)*PNum);
    A.setZero();
    S.setZero();

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rbc = Tbc.rotation();
    Eigen::Vector3d tbc = Tbc.translation();
    double dt = 1.0f / InitRate;

    for(int i = 0; i < PNum; i++)
    {
        for(int j = 1; j < FrmNum; j++)
        {
//            Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(gtbg);
//            Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(gtbg, gtba);
//            Eigen::Matrix3d DR = pIntsgt[j - 1].dR;
//            Eigen::Vector3d DP = pIntsgt[j - 1].dP;
            Eigen::Matrix3d DR = pInts[j - 1].dR;
            Eigen::Vector3d DP = pInts[j - 1].dP;
            double delt = double(j)*dt;
            int colindex = 3*(i*(FrmNum - 1) + j - 1);
            A.block(colindex, 0, 3, 3) = 0.5*delt*delt*I;
            A.block(colindex, 3, 3, 3) = delt*I;
            A.block(colindex, i*FrmNum + 6, 3, 1) = - Rbc*AlignedPts[0][i];
            A.block(colindex, i*FrmNum + j + 6, 3, 1) = DR*Rbc*AlignedPts[j][i];
            S.segment(colindex, 3) = - DP - (DR - I)*tbc;
        }
    }

    X = A.colPivHouseholderQr().solve(S);


    g = X.head(3);
    v = X.segment(3, 3);


//    std::cout << X.transpose() << std::endl;

//    std::cout << "v:" << v.transpose() << std::endl;
//    std::cout << "gtv:" << gtv.transpose() << std::endl;
    std::cout << "Before:::: " << std::endl;
    std::cout << "Errorv:" << (v - gtv).norm() << std::endl;
    std::cout << "gnorm:" << g.norm() << std::endl;
    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;


    // 消元求解：先不加ba，感觉并没有什么卵用？？？？
/*
    Eigen::MatrixXd V(3*(FrmNum - 1)*PNum, 6);
    Eigen::MatrixXd W(3*(FrmNum - 1)*PNum, 3*PNum);
    Eigen::MatrixXd Q(3*(FrmNum - 1)*PNum, (FrmNum - 1)*PNum);
    Eigen::VectorXd C(3*(FrmNum - 1)*PNum);
    V.setZero();
    W.setZero();
    Q.setZero();

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rbc = Tbc.rotation();
    Eigen::Vector3d tbc = Tbc.translation();
    double dt = 1.0f / InitRate;

    int cur_row = 0;
    for(int i = 0; i < PNum; i++)// 按行添加元素
    {
        for(int j = 0; j < FrmNum - 1; j++)
        {
            Eigen::Matrix3d DR = pInts[j].GetDeltaRotation(gtbg);
            Eigen::Vector3d DP = pInts[j].GetDeltaPosition(gtbg, gtba);
            double delt = (j + 1)*dt;
//            V.block(3*i*(FrmNum - 1) + 3*j, 0    , 3, 3) = delt*I;
//            V.block(3*i*(FrmNum - 1) + 3*j, 3    , 3, 3) = 0.5*delt*delt*I;
//            W.block(3*i*(FrmNum - 1) + 3*j, 3*i  , 3, 3) = - Rbc;
//            Q.block(3*i*(FrmNum - 1) + 3*j, i*(FrmNum - 1) + j, 3, 1) = DR*Rbc*AlignedPts[j + 1][i];
//            C.segment(3*i*(FrmNum - 1) + 3*j, 3) = (I - DR)*tbc - DP;

            V.block(cur_row, 0    , 3, 3) = delt*I;
            V.block(cur_row, 3    , 3, 3) = 0.5*delt*delt*I;
            W.block(cur_row, 3*i  , 3, 3) = - Rbc;
            Q.block(cur_row, i*(FrmNum - 1) + j, 3, 1) = DR*Rbc*AlignedPts[j + 1][i];
            C.segment(cur_row, 3) = (I - DR)*tbc - DP;
            cur_row += 3;
        }
    }

    Eigen::MatrixXd P(3*(FrmNum - 1)*PNum, 3*(FrmNum - 1)*PNum);
    Eigen::MatrixXd G(3*(FrmNum - 1)*PNum, 3*(FrmNum - 1)*PNum);
    Eigen::MatrixXd H(3*PNum, 3*PNum);
    P.setIdentity();
    P -= Q*Q.transpose();
    H = W.transpose()*P*W;
    for(int i = 0; i < PNum; i++)
    {
        H.block(3*i, 3*i, 3, 3) = H.block(3*i, 3*i, 3, 3).inverse();
    }
    G.setIdentity();
    G -= W*H*W.transpose()*P;

    // AZ = b 其中A = PGV， b = PGC
    Eigen::VectorXd Z(6);
    Z = (P*G*V).colPivHouseholderQr().solve(P*G*C);
//    Z = (V).colPivHouseholderQr().solve(C);

    Eigen::Vector3d v = Z.head(3);
    Eigen::Vector3d g = Z.segment(3, 3);


//    std::cout << (A*X - S).norm() << std::endl;
//    std::cout << "estv:" << v.transpose() << ", gtv:" << gtv.transpose() << std::endl;
//    std::cout << "estg:" << g.transpose() << ", gtg:" << gtg.transpose() << std::endl;
//    std::cout << "gnorm:" << g.norm() << std::endl << std::endl;

    std::cout << "Estv:" << v.transpose() << std::endl;
    std::cout << "gtv:" << gtv.transpose() << std::endl;
    std::cout << "Errorv:" << (v - gtv).norm() << std::endl;
    std::cout << "Errorg:" << (g - gtg).norm() << std::endl;
    std::cout << "gnorm:" << g.norm() << std::endl << std::endl;
    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;

//    std::cout << X.transpose() << std::endl;
*/


    // Ceres 优化所有项

    std::vector<double*> pointers;// 装所有lamda和bg，g

    for(int i = 0; i < PNum*FrmNum; i++)
    {
        double* lamda = new double[1];
        lamda[0] = X[i + 6];
//        lamda[0] = 1;
        pointers.push_back(lamda);
    }

    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    double* v0_ptr = new double[3];
    pointers.push_back(v0_ptr);
    Eigen::Map<Eigen::Vector3d> v0(v0_ptr);
    v0 = v;

//    double* Rwg_ptr = new double[9];
//    pointers.push_back(Rwg_ptr);
//    Eigen::Map<Eigen::Matrix3d> Rwg(Rwg_ptr);
//    Rwg.setIdentity();

//    Eigen::Vector3d dirG = g.normalized();
//    const Eigen::Vector3d gI = IMU::GRAVITY_VECTOR.normalized();
//    const Eigen::Vector3d vI = gI.cross(dirG);
//    const double cos_theta = gI.dot(dirG);
//    const double theta = std::acos(cos_theta);
//    Rwg = ExpSO3(vI*theta/vI.norm());

    double* g_ptr = new double[3];
    pointers.push_back(g_ptr);
    Eigen::Map<Eigen::Vector3d> g0(g_ptr);
//    g0.setZero();
    g0 = g;



    double** parameters = new double*[5];
    parameters[0] = bg_ptr;
//    parameters[1] = Rwg_ptr;
    parameters[1] = g_ptr;
    parameters[2] = v0_ptr;
    parameters[3] = nullptr;
    parameters[4] = nullptr;

    ceres::Problem problem;

    for(int i = 0; i < PNum; i++)
    {
        parameters[3] = pointers[i*FrmNum];//lamda0
        Eigen::Vector3d u1 = AlignedPts[0][i];
        for(int j = 1; j < FrmNum; j++)
        {
            double delt = j*dt;
            Eigen::Vector3d u2 = AlignedPts[j][i];
            parameters[4] = pointers[i*FrmNum + j];//lamda1
            ceres::CostFunction* cost_function = new CloseformCostFunction(u1, u2, pInts[j - 1], delt, gtba, Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters, 5);
//            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), parameters, 5);
//            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(10.0), parameters, 5);
//            problem.SetParameterLowerBound(parameters[3], 0, 1e-5);
//            problem.SetParameterLowerBound(parameters[4], 0, 1e-5);
        }
    }

    // gyro bias = 0的先验，感觉用处不大
//    ceres::CostFunction* prior_cost_function = new BiasPriorCostFunction(1);//prior先验权重10e5
//    problem.AddResidualBlock(prior_cost_function, nullptr, bg_ptr);

//    GravityParameterization* gravity_local_parameterization = new GravityParameterization;
//    problem.SetParameterization(Rwg_ptr, gravity_local_parameterization);


    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);


    bool converged = (summary.termination_type == ceres::CONVERGENCE);

//    const Eigen::Vector3d g0 = Rwg*IMU::GRAVITY_VECTOR;

    if(converged)
    {
        std::cout << "After:::: " << std::endl;
//        std::cout << "PointsNum: " << PointNum << std::endl;
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "Est:" << bg.transpose() << std::endl;
        std::cout << "GT :" << gtbg.transpose() << std::endl;
        std::cout << "Errorv:" << (v0 - gtv).norm() << std::endl;
        std::cout << "gnorm:" << g0.norm() << std::endl;
        std::cout << "DirectionError:" << 180.*std::acos(g0.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;


        ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
        StaError += (gtbg - bg).norm();
        StaCount ++;

        int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
        if(idi > 99)
            idi = 99;
        statistical[idi] ++;

    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }


//    std::cout << pointers[PNum*FrmNum + 1][0] << ", " << pointers[PNum*FrmNum + 1][1] << ", " << pointers[PNum*FrmNum + 1][2] << ", ";
//    std::cout << pointers[PNum*FrmNum + 2][0] << ", " << pointers[PNum*FrmNum + 2][1] << ", " << pointers[PNum*FrmNum + 2][2] << ", ";
//    for (int i = 0; i < PNum*FrmNum; i++)
//    {
//        std::cout << pointers[i][0] << ", ";
//    }

    delete[] parameters;
    for (double* ptr : pointers)
    {
//        std::cout << ptr[0] << ", ";
        delete[] ptr;
    }

}

//   不拆开，合在一起求解

void CloseForm_Solution2(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg,
                        const std::vector<IMU::Preintegrated> &pIntsgt)
{
    int PNum = PointNum;
    const int FrmNum = FrameNum + 1;


    Eigen::Vector3d g;
    Eigen::Vector3d v;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rbc = Tbc.rotation();
    Eigen::Vector3d tbc = Tbc.translation();
    double dt = 1.0f / InitRate;


    Eigen::MatrixXd A(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd X(6 + PNum*FrmNum);
    Eigen::VectorXd S(3*(FrmNum - 1)*PNum);
    A.setZero();
    S.setZero();
    for(int i = 0; i < PNum; i++)
    {
        for(int j = 1; j < FrmNum; j++)
        {
//            Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(gtbg);
//            Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(gtba, gtba);
            Eigen::Matrix3d DR = pInts[j - 1].dR;
            Eigen::Vector3d DP = pInts[j - 1].dP;
            double delt = double(j)*dt;
            int colindex = 3*(i*(FrmNum - 1) + j - 1);
            A.block(colindex, 0, 3, 3) = 0.5*delt*delt*I;
            A.block(colindex, 3, 3, 3) = delt*I;
            A.block(colindex, i*FrmNum + 6, 3, 1) = - Rbc*AlignedPts[0][i];
            A.block(colindex, i*FrmNum + j + 6, 3, 1) = DR*Rbc*AlignedPts[j][i];
            S.segment(colindex, 3) = - DP - (DR - I)*tbc;
        }
    }

    X = A.colPivHouseholderQr().solve(S);

//    Eigen::VectorXd XX(6 + PNum*FrmNum);
//    XX = (A.transpose()*A).inverse()*A.transpose()*S;

    g = X.head(3);
    v = X.segment(3, 3);

//    std::cout << "X:::" << X.transpose() << std::endl;
//    std::cout << "XX:::" << XX.transpose() << std::endl;

/*
    // Ceres 只优化bg
    std::vector<double*> pointers;// 这里只装bg

    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    double** parameters = new double*[1];
    parameters[0] = bg_ptr;

    ceres::Problem problem;

    ceres::CostFunction* cost_function = new CloseformCostFunction2(AlignedPts, pInts, X, dt, gtba, Tbc);
    problem.AddResidualBlock(cost_function, nullptr, parameters, 1);

    ceres::Solver::Options options;

    options.max_num_iterations = 300;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    bool converged = (summary.termination_type == ceres::CONVERGENCE);

    if(converged)
    {
        std::cout << "estbg:" << bg.transpose() << std::endl;
        std::cout << " gtbg:" << gtbg.transpose() << std::endl;
    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }

    ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
    StaError += (gtbg - bg).norm();
    StaCount ++;

    delete[] parameters;
    for (double* ptr : pointers)
    {
        delete[] ptr;
    }
*/


    // Ceres 全部一起优化

    std::vector<double*> pointers;// 装所有lamda和bg，g

    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    double* g_ptr = new double[3];
    pointers.push_back(g_ptr);
    Eigen::Map<Eigen::Vector3d> g0(g_ptr);
    g0 = g;

//    double* Rwg_ptr = new double[9];
//    pointers.push_back(Rwg_ptr);
//    Eigen::Map<Eigen::Matrix3d> Rwg(Rwg_ptr);
//    Rwg.setIdentity();
//    Eigen::Vector3d dirG = g.normalized();
//    const Eigen::Vector3d gI = IMU::GRAVITY_VECTOR.normalized();
//    const Eigen::Vector3d vI = gI.cross(dirG);
//    const double cos_theta = gI.dot(dirG);
//    const double theta = std::acos(cos_theta);
//    Rwg = ExpSO3(vI*theta/vI.norm());
/*
    double* v0_ptr = new double[3];
    pointers.push_back(v0_ptr);
    Eigen::Map<Eigen::Vector3d> v0(v0_ptr);
    v0 = v;

    double* lamda_ptr = new double[PNum*FrmNum];
    pointers.push_back(lamda_ptr);
    Eigen::Map<Eigen::Matrix<double, PNum*FrmNum, 1>> lamda(lamda_ptr);
    lamda.setOnes();

    double** parameters = new double*[4];
    parameters[0] = bg_ptr;
    parameters[1] = g_ptr;
//    parameters[1] = Rwg_ptr;
    parameters[2] = v0_ptr;
    parameters[3] = lamda_ptr;

    ceres::Problem problem;

    ceres::CostFunction* cost_function = new CloseformCostFunction3(AlignedPts, pInts, dt, gtba, Tbc);
    problem.AddResidualBlock(cost_function, nullptr, parameters, 4);

//    GravityParameterization* gravity_local_parameterization = new GravityParameterization;
//    problem.SetParameterization(Rwg_ptr, gravity_local_parameterization);

    ceres::Solver::Options options;

    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    bool converged = (summary.termination_type == ceres::CONVERGENCE);

    if(converged)
    {
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "estbg:" << bg.transpose() << std::endl;
        std::cout << " gtbg:" << gtbg.transpose() << std::endl;

//        Eigen::Vector3d g0 = Rwg*IMU::GRAVITY_VECTOR;
        std::cout << "gnorm:" << g0.norm() << std::endl;
        std::cout << "DirectionError:" << 180.*std::acos(g0.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;
    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }

    ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
    StaError += (gtbg - bg).norm();
    StaCount ++;

    delete[] parameters;
    for (double* ptr : pointers)
    {
        delete[] ptr;
    }
*/

    // 每次求解X后带入ceres优化，迭代N次
    /*
    Eigen::Vector3d Est_bg;
    Est_bg.setZero();

    for(int cct = 0; cct < 1; cct++)
    {
        Eigen::MatrixXd A(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
        Eigen::VectorXd X(6 + PNum*FrmNum);
        Eigen::VectorXd S(3*(FrmNum - 1)*PNum);
        A.setZero();
        S.setZero();
        for(int i = 0; i < PNum; i++)
        {
            for(int j = 1; j < FrmNum; j++)
            {
                Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(Est_bg);
                Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(Est_bg, gtba);
    //            Eigen::Matrix3d DR = pIntsgt[j - 1].dR;
    //            Eigen::Vector3d DP = pIntsgt[j - 1].dP;
                double delt = double(j)*dt;
                int colindex = 3*(i*(FrmNum - 1) + j - 1);
                A.block(colindex, 0, 3, 3) = 0.5*delt*delt*I;
                A.block(colindex, 3, 3, 3) = delt*I;
                A.block(colindex, i*FrmNum + 6, 3, 1) = - Rbc*AlignedPts[0][i];
                A.block(colindex, i*FrmNum + j + 6, 3, 1) = DR*Rbc*AlignedPts[j][i];
                S.segment(colindex, 3) = - DP - (DR - I)*tbc;
            }
        }

        X = A.colPivHouseholderQr().solve(S);
        // Ceres 只优化bg(不拆散优化)
        std::vector<double*> pointers;// 这里只装bg

        double* bg_ptr = new double[3];
        pointers.push_back(bg_ptr);
        Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
        bg = Est_bg;
//        bg.setZero();

        double** parameters = new double*[1];
        parameters[0] = bg_ptr;

        ceres::Problem problem;

        ceres::CostFunction* cost_function = new CloseformCostFunction2(AlignedPts, pInts, X, dt, gtba, Tbc);
        problem.AddResidualBlock(cost_function, nullptr, parameters, 1);

        ceres::Solver::Options options;

        options.max_num_iterations = 300;
        ceres::Solver::Summary summary;

        ceres::Solve(options, &problem, &summary);

        bool converged = (summary.termination_type == ceres::CONVERGENCE);

        if(converged)
        {
            Est_bg = bg;
            delete[] parameters;
            for (double* ptr : pointers)
            {
                delete[] ptr;
            }
        }
        else
        {
            Est_bg.setZero();
            delete[] parameters;
            for (double* ptr : pointers)
            {
                delete[] ptr;
            }
            break;
        }

    }


    if(Est_bg.isZero())
    {
        std::cout << "est_failed !!!!!!!!!!!!" << std::endl;
        return;
    }
    std::cout << "estbg:" << Est_bg.transpose() << std::endl;
    std::cout << " gtbg:" << gtbg.transpose() << std::endl;
    ErrorPercen += (gtbg - Est_bg).norm()/gtbg.norm();
    StaError += (gtbg - Est_bg).norm();
    StaCount ++;
*/
}


// 数值求导，不用ceres
void CloseForm_Solution3(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg,
                        const std::vector<IMU::Preintegrated> &pIntsgt)
{
    const int PNum = PointNum;
    const int FrmNum = FrameNum + 1;

    Eigen::Vector3d g;
    Eigen::Vector3d v;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rbc = Tbc.rotation();
    Eigen::Vector3d tbc = Tbc.translation();
    double dt = 1.0f / InitRate;
    Eigen::Vector3d bg;
    bg.setZero();

    Eigen::Vector3d ba;
    ba = gtba;
//    ba.setZero();

    Eigen::MatrixXd A(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd X(6 + PNum*FrmNum);
    Eigen::VectorXd B(3*(FrmNum - 1)*PNum);
    A.setZero();
    B.setZero();
    int rowindex = 0;
    for(int j = 1; j < FrmNum; j++)
    {
        double delt = double(j)*dt;
        for(int i = 0; i < PNum; i++)
        {
            A.block(rowindex, 0, 3, 3) = 0.5*delt*delt*I;
            A.block(rowindex, 3, 3, 3) = delt*I;
            A.block(rowindex, 6 + i, 3, 1) = - Rbc*AlignedPts[0][i];
            rowindex += 3;
        }
    }

    Eigen::MatrixXd Ax(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd Xx(6 + PNum*FrmNum);
    Eigen::VectorXd Bx(3*(FrmNum - 1)*PNum);
    Eigen::MatrixXd Ay(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd Xy(6 + PNum*FrmNum);
    Eigen::VectorXd By(3*(FrmNum - 1)*PNum);
    Eigen::MatrixXd Az(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd Xz(6 + PNum*FrmNum);
    Eigen::VectorXd Bz(3*(FrmNum - 1)*PNum);

    double delta = 1e-8;

    Ax = A;
    Ay = A;
    Az = A;

    for(int cct = 0; cct < 5; cct++)
    {
        rowindex = 0;
        int colindex = PNum + 6;
        for(int j = 1; j < FrmNum; j++)
        {
            Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(bg);
            Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(bg, ba);

            Eigen::Matrix3d DRx = pInts[j - 1].GetDeltaRotation(bg + Eigen::Vector3d(delta, 0, 0));
            Eigen::Vector3d DPx = pInts[j - 1].GetDeltaPosition(bg + Eigen::Vector3d(delta, 0, 0), ba);
            Eigen::Matrix3d DRy = pInts[j - 1].GetDeltaRotation(bg + Eigen::Vector3d(0, delta, 0));
            Eigen::Vector3d DPy = pInts[j - 1].GetDeltaPosition(bg + Eigen::Vector3d(0, delta, 0), ba);
            Eigen::Matrix3d DRz = pInts[j - 1].GetDeltaRotation(bg + Eigen::Vector3d(0, 0, delta));
            Eigen::Vector3d DPz = pInts[j - 1].GetDeltaPosition(bg + Eigen::Vector3d(0, 0, delta), ba);
            for(int i = 0; i < PNum; i++)
            {
                Eigen::Vector3d TmpP = Rbc*AlignedPts[j][i];
                A.block(rowindex, colindex, 3, 1) = DR*TmpP;
                B.segment(rowindex, 3) = - DP - (DR - I)*tbc;
                Ax.block(rowindex, colindex, 3, 1) = DRx*TmpP;
                Bx.segment(rowindex, 3) = - DPx - (DRx - I)*tbc;
                Ay.block(rowindex, colindex, 3, 1) = DRy*TmpP;
                By.segment(rowindex, 3) = - DPy - (DRy - I)*tbc;
                Az.block(rowindex, colindex, 3, 1) = DRz*TmpP;
                Bz.segment(rowindex, 3) = - DPz - (DRz - I)*tbc;
                rowindex += 3;
                colindex ++;
            }
        }

        X = A.householderQr().solve(B);
        Xx = Ax.householderQr().solve(Bx);
        Xy = Ay.householderQr().solve(By);
        Xz = Az.householderQr().solve(Bz);

//        X = A.colPivHouseholderQr().solve(B);
//        Xx = Ax.colPivHouseholderQr().solve(Bx);
//        Xy = Ay.colPivHouseholderQr().solve(By);
//        Xz = Az.colPivHouseholderQr().solve(Bz);

        Eigen::VectorXd f(3*(FrmNum - 1)*PNum);
        f = A*X - B;

        Eigen::MatrixXd JT(3*(FrmNum - 1)*PNum, 3);
        JT.col(0) = ((Ax*Xx - Bx) - f)/delta;
        JT.col(1) = ((Ay*Xy - By) - f)/delta;
        JT.col(2) = ((Az*Xz - Bz) - f)/delta;

//        double lam = 1;
//        Eigen::Vector3d dx = (JT.transpose()*JT + lam*Eigen::Matrix3d::Identity()).ldlt().solve( - JT.transpose()*f);
        Eigen::Vector3d dx = (JT.transpose()*JT).ldlt().solve( - JT.transpose()*f);
        bg += dx;

        std::cout << " gravity: " << X.head(3).norm() << std::endl;
        std::cout << cct << " iterations: " << bg.transpose() << std::endl;
        std::cout << " FinalError: " << f.norm() << std::endl;
    }
    std::cout << " gtbg: " << gtbg.transpose() << std::endl;

    std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;

    ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
    StaError += (gtbg - bg).norm();
    StaCount ++;


// 获取gt——error
    /*
    Eigen::MatrixXd Agt(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd Xgt(6 + PNum*FrmNum);
    Eigen::VectorXd Bgt(3*(FrmNum - 1)*PNum);
    Agt = A;
    Agt = A;
    Agt = A;

    rowindex = 0;
    int colindex = PNum + 6;
    for(int j = 1; j < FrmNum; j++)
    {
        Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(gtbg);
        Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(gtbg, gtba);
        for(int i = 0; i < PNum; i++)
        {
            Agt.block(rowindex, colindex, 3, 1) = DR*Rbc*AlignedPts[j][i];
            Bgt.segment(rowindex, 3) = - DP - (DR - I)*tbc;
            rowindex += 3;
            colindex ++;
        }
    }

    Xgt = Agt.colPivHouseholderQr().solve(Bgt);
    Eigen::VectorXd fgt(3*(FrmNum - 1)*PNum);
    fgt = Agt*Xgt - Bgt;

    std::cout << " Final GTError: " << fgt.norm() << std::endl;
*/
}



void CloseForm_Solution4(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg,
                        const std::vector<IMU::Preintegrated> &pIntsgt)
{
    int PNum = PointNum;
    int FrmNum = FrameNum + 1;

    // 加上g一起计算·改，用论文中方式计算,时间会更快
    Eigen::Vector3d g;
    Eigen::Vector3d v;

    Eigen::MatrixXd A(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd X(6 + PNum*FrmNum);
    Eigen::VectorXd S(3*(FrmNum - 1)*PNum);
    A.setZero();
    S.setZero();

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rbc = Tbc.rotation();
    Eigen::Vector3d tbc = Tbc.translation();
    double dt = 1.0f / InitRate;

    for(int i = 0; i < PNum; i++)
    {
        for(int j = 1; j < FrmNum; j++)
        {
//            Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(gtbg);
//            Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(gtba, gtba);
            Eigen::Matrix3d DR = pInts[j - 1].dR;
            Eigen::Vector3d DP = pInts[j - 1].dP;
            double delt = double(j)*dt;
            int colindex = 3*(i*(FrmNum - 1) + j - 1);
            A.block(colindex, 0, 3, 3) = 0.5*delt*delt*I;
            A.block(colindex, 3, 3, 3) = delt*I;
            A.block(colindex, i*FrmNum + 6, 3, 1) = - Rbc*AlignedPts[0][i];
            A.block(colindex, i*FrmNum + j + 6, 3, 1) = DR*Rbc*AlignedPts[j][i];
            S.segment(colindex, 3) = - DP - (DR - I)*tbc;
        }
    }

    X = A.colPivHouseholderQr().solve(S);


    g = X.head(3);
    v = X.segment(3, 3);

    // Ceres 优化所有项
    std::vector<double*> pointers;// 装所有lamda和bg，g

    for(int i = 0; i < PNum*FrmNum; i++)
    {
        double* lamda = new double[1];
//        lamda[0] = X[i + 6];
        lamda[0] = 1;
        pointers.push_back(lamda);

        double* duv = new double[2];
        duv[0] = 0;
        duv[1] = 0;
        pointers.push_back(duv);
    }

    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    double* v0_ptr = new double[3];
    pointers.push_back(v0_ptr);
    Eigen::Map<Eigen::Vector3d> v0(v0_ptr);
    v0 = v;

//    double* Rwg_ptr = new double[9];
//    pointers.push_back(Rwg_ptr);
//    Eigen::Map<Eigen::Matrix3d> Rwg(Rwg_ptr);
//    Rwg.setIdentity();

//    Eigen::Vector3d dirG = g.normalized();
//    const Eigen::Vector3d gI = IMU::GRAVITY_VECTOR.normalized();
//    const Eigen::Vector3d vI = gI.cross(dirG);
//    const double cos_theta = gI.dot(dirG);
//    const double theta = std::acos(cos_theta);
//    Rwg = ExpSO3(vI*theta/vI.norm());

    double* g_ptr = new double[3];
    pointers.push_back(g_ptr);
    Eigen::Map<Eigen::Vector3d> g0(g_ptr);
    g0 = g;



    double** parameters = new double*[7];
    parameters[0] = bg_ptr;
//    parameters[1] = Rwg_ptr;
    parameters[1] = g_ptr;
    parameters[2] = v0_ptr;
    parameters[3] = nullptr;
    parameters[4] = nullptr;
    parameters[5] = nullptr;
    parameters[6] = nullptr;

    ceres::Problem problem;

    for(int i = 0; i < PNum; i++)
    {
        parameters[3] = pointers[2*i*FrmNum];//lamda0
        parameters[5] = pointers[2*i*FrmNum + 1];//duv1
        Eigen::Vector3d u1 = AlignedPts[0][i];
        for(int j = 1; j < FrmNum; j++)
        {
            double delt = j*dt;
            Eigen::Vector3d u2 = AlignedPts[j][i];
            parameters[4] = pointers[2*(i*FrmNum + j)];//lamda1
            parameters[6] = pointers[2*(i*FrmNum + j) + 1];//lamda1
            ceres::CostFunction* cost_function = new CloseformCostFunction4(u1, u2, pInts[j - 1], delt, gtba, Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters, 7);
        }
    }

//    GravityParameterization* gravity_local_parameterization = new GravityParameterization;
//    problem.SetParameterization(Rwg_ptr, gravity_local_parameterization);

    for(int i = 0; i < PNum*FrmNum; i++)
    {
        ceres::CostFunction* prior_cost_function = new DuvCostFunction(1000);//prior先验权重10e5
        problem.AddResidualBlock(prior_cost_function, nullptr, pointers[2*i + 1]);
    }


    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);


    bool converged = (summary.termination_type == ceres::CONVERGENCE);

    if(converged)
    {
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "Est:" << bg.transpose() << std::endl;
        std::cout << "GT :" << gtbg.transpose() << std::endl;
//        std::cout << "DirectionError:" << 180.*std::acos(g0.normalized().dot(gtg.normalized()))/EIGEN_PI << ", " << g0.norm() << std::endl;

        ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
        StaError += (gtbg - bg).norm();
        StaCount ++;

        int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
        if(idi > 99)
            idi = 99;
        statistical[idi] ++;
    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }

    delete[] parameters;
    for (double* ptr : pointers)
    {
        delete[] ptr;
    }
}


void Reprojection_Solution(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg, std::vector<Eigen::Vector3d> &gttrans)
{
    int PNum = AlignedPts.front().size();
    int FrmNum = FrameNum + 1;

    Eigen::Vector3d g;
    Eigen::Vector3d v;

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rbc = Tbc.rotation();
    Eigen::Vector3d tbc = Tbc.translation();
    double dt = 1.0f / InitRate;
    int cur_row = 0;


/*
    Eigen::MatrixXd A(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd X(6 + PNum*FrmNum);
    Eigen::VectorXd S(3*(FrmNum - 1)*PNum);
    A.setZero();
    S.setZero();


    for(int i = 0; i < FrmNum - 1; i++)
    {
//        Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(gtbg);
//        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(gtbg, gtba);
//        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(gtbg, gtba);
        Eigen::Matrix3d DR = pInts[i].dR;
        Eigen::Vector3d DV = pInts[i].dV;
        Eigen::Vector3d DP = pInts[i].dP;
        double delt = double(FrmNum - 1 - i)*dt;
        Eigen::Vector3d SubS = DV*delt - DP - (DR - I)*tbc;
        for(int j = 0; j < PNum; j++)
        {
            A.block(cur_row, 0, 3, 3) = - 0.5*DR*delt*delt;
            A.block(cur_row, 3, 3, 3) = DR*delt;
            A.block(cur_row, j + 6, 3, 1) = DR*Rbc*AlignedPts.back()[j];
            A.block(cur_row, (i + 1)*PNum + j + 6, 3, 1) = - Rbc*AlignedPts[i][j];
            S.segment(cur_row, 3) = SubS;
            cur_row += 3;
        }
    }
//    Timer timer;
//    timer.Start();
    X = A.householderQr().solve(S);

//    std::cout << "time: ";
//    timer.PrintSeconds();

    g = X.head(3);
    v = X.segment(3, 3);
    /*
    for(int i = 0; i < PNum; i++)
    {
        std::cout << X[i + 6] << ", ";
//        Eigen::Vector3d pos = X[i + 6]*AlignedPts[0][i];
//        std::cout << pos.transpose() << std::endl;
    }
    */
/*
    std::cout << std::endl;

    std::cout << "Before::::::::::::::::::::" << std::endl;

    std::cout << "v:" << v.transpose() << std::endl;
    std::cout << "gtv:" << gtv.transpose() << std::endl;
    std::cout << "Errorv:" << (v - gtv).norm() << std::endl;
    std::cout << "gnorm:" << g.norm() << std::endl;
    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;
*/








    // ceres 优化时把t12拆开，即优化v和bg
/*
    std::vector<double*> pointers;// 装所有lamda和bg，g

    for(int i = 0; i < PNum; i++)
    {
        double* lamda = new double[1];
//        lamda[0] = 1/X[i + 6];
        lamda[0] = 1;
        pointers.push_back(lamda);
    }

    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    double* v0_ptr = new double[3];
    pointers.push_back(v0_ptr);
    Eigen::Map<Eigen::Vector3d> v0(v0_ptr);
    v0 = v;

//    double* Rwg_ptr = new double[9];
//    pointers.push_back(Rwg_ptr);
//    Eigen::Map<Eigen::Matrix3d> Rwg(Rwg_ptr);
//    Rwg.setIdentity();

//    Eigen::Vector3d dirG = g.normalized();
//    const Eigen::Vector3d gI = IMU::GRAVITY_VECTOR.normalized();
//    const Eigen::Vector3d vI = gI.cross(dirG);
//    const double cos_theta = gI.dot(dirG);
//    const double theta = std::acos(cos_theta);
//    Rwg = ExpSO3(vI*theta/vI.norm());

    double* g_ptr = new double[3];
    pointers.push_back(g_ptr);
    Eigen::Map<Eigen::Vector3d> g0(g_ptr);
    g0 = g;

    double** parameters = new double*[4];
    parameters[0] = bg_ptr;
    parameters[1] = v0_ptr;
    parameters[2] = g_ptr;
    parameters[3] = nullptr;

    ceres::Problem problem;

    for(int i = 0; i < PNum; i++)
    {
        parameters[3] = pointers[i];
        Eigen::Vector3d u2 = AlignedPts.back()[i];
        for(int j = 0; j < FrmNum - 1; j++)
        {
            double delt = (FrmNum - 1 - j)*dt;
            Eigen::Vector3d u1 = AlignedPts[j][i];
            ceres::CostFunction* cost_function = new ReprojectionCostFunction2(u1, u2, pInts[j], gtba, delt, Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters, 4);
        }
    }



    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    bool converged = (summary.termination_type == ceres::CONVERGENCE);


//    if(converged)
//    {
//        bool failed = false;
//        for(int i = 0; i < PNum; i++)
//        {
//            double lamda = pointers[i][0];
//            if(lamda < 0)
//            {
//                failed = true;
//                break;
//            }
//        }
//        if(failed)
//        {
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            delete[] parameters;
//            for (double* ptr : pointers)
//                delete[] ptr;
//            return;
//        }
//    }


    if(converged)
    {
        std::cout << "After::::::::::::::::::::" << std::endl;

        for(int i = 0; i < PNum; i++)
        {
            double lamda = pointers[i][0];
            std::cout << 1/lamda << ", ";
        }
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "Est:" << bg.transpose() << std::endl;
        std::cout << "GT :" << gtbg.transpose() << std::endl;
        std::cout << "DirectionError:" << 180.*std::acos(g0.normalized().dot(gtg.normalized()))/EIGEN_PI << ", " << g0.norm() << std::endl;
        std::cout << "v0" << v0.transpose() << ", gtv0:" << gtv.transpose() << std::endl;

        ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
        StaError += (gtbg - bg).norm();
        StaCount ++;

        int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
        if(idi > 99)
            idi = 99;
        statistical[idi] ++;
    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }
*/






    // Ceres 优化所有项
    // 优化每一帧到最后一帧的t12

    std::vector<double*> pointers;// 装所有lamda和bg，g

    for(int i = 0; i < PNum; i++)
    {
        double* lamda = new double[1];
//        lamda[0] = 1/X[i + 6];
        lamda[0] = 1;
        pointers.push_back(lamda);
    }

    for(int i = 0; i < FrmNum - 1; i++)
    {
        double* t12 = new double[3];
        t12[0] = 0;
        t12[1] = 0;
        t12[2] = 0;
        pointers.push_back(t12);
    }

    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();


    double** parameters = new double*[3];
    parameters[0] = bg_ptr;
    parameters[1] = nullptr;
    parameters[2] = nullptr;

    ceres::Problem problem;
    for(int i = 0; i < PNum; i++)
    {
        parameters[2] = pointers[i];
        Eigen::Vector3d u2 = AlignedPts.back()[i];

        for(int j = 0; j < FrmNum - 1; j++)
        {
            parameters[1] = pointers[PNum + j];
            Eigen::Vector3d u1 = AlignedPts[j][i];
            ceres::CostFunction* cost_function = new ReprojectionCostFunction(u1, u2, pInts[j], Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters, 3);
        }
    }

    ceres::Solver::Options options;

//    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.max_solver_time_in_seconds = 0.2;

    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    bool converged = (summary.termination_type == ceres::CONVERGENCE);


//    if(converged)
//    {
//        bool failed = false;
//        for(int i = 0; i < PNum; i++)
//        {
//            double lamda = pointers[i][0];
//            if(lamda < 0)
//            {
//                failed = true;
//                break;
//            }
//        }
//        if(failed)
//        {
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            delete[] parameters;
//            for (double* ptr : pointers)
//                delete[] ptr;
//            return;
//        }
//    }


    std::vector<double> Optimize_depth;
    if(converged)
    {

        for(int i = 0; i < PNum; i++)
        {
            double lamda = 1/pointers[i][0];
            Optimize_depth.push_back(lamda);
        }

//        std::cout << "PointsNum: " << PointNum << std::endl;
        std::cout << "bgError:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
//        std::cout << "Est:" << bg.transpose() << std::endl;
//        std::cout << "GT :" << gtbg.transpose() << std::endl;

//        Eigen::Vector3d esttrans(pointers[PNum][0], pointers[PNum][1], pointers[PNum][2]);
//        std::cout << "EstTrans:::" << esttrans.transpose() << std::endl;
//        std::cout << "gtTrans:::" << gttrans.transpose() << std::endl;


        ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
        StaError += (gtbg - bg).norm();
        StaCount ++;

        int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
        if(idi > 99)
            idi = 99;
        statistical[idi] ++;
    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }


/*

*/

    // 几何一致性验证

    PNum = 10;

    Eigen::MatrixXd A(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd X(6 + PNum*FrmNum);
    Eigen::VectorXd S(3*(FrmNum - 1)*PNum);
    A.setZero();
    S.setZero();


    for(int i = 0; i < FrmNum - 1; i++)
    {
        Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(bg);
        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(bg, Eigen::Vector3d::Zero());
        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(bg, Eigen::Vector3d::Zero());
//        Eigen::Matrix3d DR = pInts[i].dR;
//        Eigen::Vector3d DV = pInts[i].dV;
//        Eigen::Vector3d DP = pInts[i].dP;
        double delt = double(FrmNum - 1 - i)*dt;
        Eigen::Vector3d SubS = DV*delt - DP - (DR - I)*tbc;
        for(int j = 0; j < PNum; j++)
        {
            A.block(cur_row, 0, 3, 3) = - 0.5*DR*delt*delt;
            A.block(cur_row, 3, 3, 3) = DR*delt;
            A.block(cur_row, j + 6, 3, 1) = DR*Rbc*AlignedPts.back()[j];
            A.block(cur_row, (i + 1)*PNum + j + 6, 3, 1) = - Rbc*AlignedPts[i][j];
            S.segment(cur_row, 3) = SubS;
            cur_row += 3;
        }
    }
//    Timer timer;
//    timer.Start();
    X = A.householderQr().solve(S);

//    std::cout << "time: ";
//    timer.PrintSeconds();

    g = X.head(3);
    v = X.segment(3, 3);

    std::cout << "Est::::::::::::::::::::" << std::endl;

    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << ", " << g.norm() << std::endl;
    gravity_error += 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI;
    std::cout << "estv:" << v.transpose() << std::endl;
    std::cout << "gtv:" << gtv.transpose() << std::endl;
    std::cout << "depth ::: ";


    std::vector<double> scales;
    for(int i = 0; i < PNum; i++)
    {
        scales.emplace_back(X[i + 6]/Optimize_depth[i]);
    }
    std::cout << std::endl;



/*
    A.setZero();
    S.setZero();
    cur_row = 0;
    for(int i = 0; i < FrmNum - 1; i++)
    {
        Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(gtbg);
        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(gtbg, gtba);
        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(gtbg, gtba);
//        Eigen::Matrix3d DR = pInts[i].dR;
//        Eigen::Vector3d DV = pInts[i].dV;
//        Eigen::Vector3d DP = pInts[i].dP;
        double delt = double(FrmNum - 1 - i)*dt;
        Eigen::Vector3d SubS = DV*delt - DP - (DR - I)*tbc;
        for(int j = 0; j < PNum; j++)
        {
            A.block(cur_row, 0, 3, 3) = - 0.5*DR*delt*delt;
            A.block(cur_row, 3, 3, 3) = DR*delt;
            A.block(cur_row, j + 6, 3, 1) = DR*Rbc*AlignedPts.back()[j];
            A.block(cur_row, (i + 1)*PNum + j + 6, 3, 1) = - Rbc*AlignedPts[i][j];
            S.segment(cur_row, 3) = SubS;
            cur_row += 3;
        }
    }
    X = A.householderQr().solve(S);
    g = X.head(3);
    v = X.segment(3, 3);
    /*
    for(int i = 0; i < PNum; i++)
    {
        std::cout << X[i + 6] << ", ";
//        Eigen::Vector3d pos = X[i + 6]*AlignedPts[0][i];
//        std::cout << pos.transpose() << std::endl;
    }
    */
/*
    std::cout << std::endl;

    std::cout << "Truth::::::::::::::::::::" << std::endl;

    std::cout << "v:" << v.transpose() << std::endl;
    std::cout << "gtv:" << gtv.transpose() << std::endl;
    std::cout << "Errorv:" << (v - gtv).norm() << std::endl;
    std::cout << "gnorm:" << g.norm() << std::endl;
    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;
    std::cout << "depth ::: ";
    for(int i = 6; i < 6 + PNum; i++)
    {
        std::cout << X[i] << ", ";
    }
*/



    // 线性求解

/*

    std::vector<Eigen::Vector3d> Vect;
    for(int i = 0; i < FrmNum - 1; i++)
    {
        double* t12 = pointers[PNum + i];
        Eigen::Vector3d t(t12[0], t12[1], t12[2]);
        Vect.push_back(t);
    }
    Eigen::Vector3d v;
    Eigen::Vector3d g;

    Eigen::MatrixXd A(6, 6);
    Eigen::VectorXd X(6);
    Eigen::VectorXd B(6);
    A.setZero();
    B.setZero();
    double dt = 1.0f / InitRate;

    for(int i = 0; i < FrmNum - 1; i++)
    {
        double delt = (FrmNum - 1 - i)*dt;
        Eigen::MatrixXd SubA(3, 6);
        Eigen::Vector3d SubB;
        Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(bg);
        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(bg, Eigen::Vector3d::Zero());
        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(bg, Eigen::Vector3d::Zero());
//        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(bg, gtba);
//        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(bg, gtba);

        SubA.block(0, 0, 3, 3) = DR*delt;
        SubA.block(0, 3, 3, 3) = - 0.5*DR*delt*delt;
//        SubB = gttrans[i] - DP + DV*delt;// 用gttrans计算
        SubB = Vect[i] - DP + DV*delt;// 用估计的trans计算

        A += SubA.transpose()*SubA;
        B += SubA.transpose()*SubB;
    }
    X = A.ldlt().solve(B);

    v = X.head(3);
    g = X.tail(3);

    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << ", " << g.norm() << std::endl;

    gravity_error += 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI;

    std::cout << "estv:" << v.transpose() << std::endl;
    std::cout << "gtv:" << gtv.transpose() << std::endl;

    // 加入ba优化，效果很烂
//    Eigen::Vector3d v;
//    Eigen::Vector3d g;
//    Eigen::Vector3d ba;

//    Eigen::MatrixXd A(9, 9);
//    Eigen::VectorXd X(9);
//    Eigen::VectorXd B(9);
//    A.setZero();
//    B.setZero();
//    double dt = 1.0f / InitRate;

//    for(int i = 0; i < FrmNum - 1; i++)
//    {
//        double delt = (FrmNum - 1 - i)*dt;
//        Eigen::MatrixXd SubA(3, 9);
//        Eigen::Vector3d SubB;
//        Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(bg);
//        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(bg, Eigen::Vector3d::Zero());
//        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(bg, Eigen::Vector3d::Zero());

//        SubA.block(0, 0, 3, 3) = DR*delt;
//        SubA.block(0, 3, 3, 3) = - 0.5*DR*delt*delt;
//        SubA.block(0, 6, 3, 3) = pInts[i].JPa - pInts[i].JVa*delt;
//        SubB = Vect[i] - DP + DV*delt;

//        A += SubA.transpose()*SubA;
//        B += SubA.transpose()*SubB;
//    }
//    X = A.ldlt().solve(B);

//    v = X.head(3);
//    g = X.segment(3, 3);
//    ba = X.tail(3);
*/

//    delete[] parameters;
//    for (double* ptr : pointers)
//    {
//        delete[] ptr;
//    }


}



void Reprojection_SolutionR21(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg, std::vector<Eigen::Vector3d> &gttrans)
{
    int PNum = AlignedPts.front().size();
    int FrmNum = FrameNum + 1;

    Eigen::Vector3d g;
    Eigen::Vector3d v;

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rbc = Tbc.rotation();
    Eigen::Vector3d tbc = Tbc.translation();
    double dt = 1.0f / InitRate;
    int cur_row = 0;

    Eigen::MatrixXd A(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd X(6 + PNum*FrmNum);
    Eigen::VectorXd S(3*(FrmNum - 1)*PNum);
    A.setZero();
    S.setZero();
    for(int i = 0; i < PNum; i++)
    {
        for(int j = 1; j < FrmNum; j++)
        {
//            Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(gtbg);
//            Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(gtbg, gtba);
//            Eigen::Matrix3d DR = pIntsgt[j - 1].dR;
//            Eigen::Vector3d DP = pIntsgt[j - 1].dP;
            Eigen::Matrix3d DR = pInts[j - 1].dR;
            Eigen::Vector3d DP = pInts[j - 1].dP;
            double delt = double(j)*dt;
            A.block(cur_row, 0, 3, 3) = 0.5*delt*delt*I;
            A.block(cur_row, 3, 3, 3) = delt*I;
            A.block(cur_row, i + 6, 3, 1) = - Rbc*AlignedPts[0][i];
            A.block(cur_row, j*PNum + i + 6, 3, 1) = DR*Rbc*AlignedPts[j][i];
            S.segment(cur_row, 3) = - DP - (DR - I)*tbc;
            cur_row += 3;
        }
    }

    X = A.colPivHouseholderQr().solve(S);


    g = X.head(3);
    v = X.segment(3, 3);
    for(int i = 0; i < PNum; i++)
    {
        std::cout << X[i + 6] << ", ";
//        Eigen::Vector3d pos = X[i + 6]*AlignedPts[0][i];
//        std::cout << pos.transpose() << std::endl;
    }
    std::cout << std::endl;

    std::cout << "v:" << v.transpose() << std::endl;
    std::cout << "gtv:" << gtv.transpose() << std::endl;
    std::cout << "Errorv:" << (v - gtv).norm() << std::endl;
    std::cout << "gnorm:" << g.norm() << std::endl;
    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;



    // 消元求解(速度很慢)
    /*
    Eigen::MatrixXd V(3*(FrmNum - 1)*PNum, 6);
    Eigen::MatrixXd W(3*(FrmNum - 1)*PNum, 3*PNum);
    Eigen::MatrixXd Q(3*(FrmNum - 1)*PNum, (FrmNum - 1)*PNum);
    Eigen::VectorXd C(3*(FrmNum - 1)*PNum);
    V.setZero();
    W.setZero();
    Q.setZero();

    cur_row = 0;
    for(int i = 0; i < PNum; i++)// 按行添加元素
    {
        for(int j = 0; j < FrmNum - 1; j++)
        {
            Eigen::Matrix3d DR = pInts[j].GetDeltaRotation(gtbg);
            Eigen::Vector3d DP = pInts[j].GetDeltaPosition(gtbg, gtba);
            double delt = (j + 1)*dt;
//            V.block(3*i*(FrmNum - 1) + 3*j, 0    , 3, 3) = delt*I;
//            V.block(3*i*(FrmNum - 1) + 3*j, 3    , 3, 3) = 0.5*delt*delt*I;
//            W.block(3*i*(FrmNum - 1) + 3*j, 3*i  , 3, 3) = - Rbc;
//            Q.block(3*i*(FrmNum - 1) + 3*j, i*(FrmNum - 1) + j, 3, 1) = DR*Rbc*AlignedPts[j + 1][i];
//            C.segment(3*i*(FrmNum - 1) + 3*j, 3) = (I - DR)*tbc - DP;

            V.block(cur_row, 0    , 3, 3) = delt*I;
            V.block(cur_row, 3    , 3, 3) = 0.5*delt*delt*I;
            W.block(cur_row, 3*i  , 3, 3) = - Rbc;
            Q.block(cur_row, i*(FrmNum - 1) + j, 3, 1) = DR*Rbc*AlignedPts[j + 1][i];
            C.segment(cur_row, 3) = (I - DR)*tbc - DP;
            cur_row += 3;
        }
    }

    Eigen::MatrixXd P(3*(FrmNum - 1)*PNum, 3*(FrmNum - 1)*PNum);
    Eigen::MatrixXd G(3*(FrmNum - 1)*PNum, 3*(FrmNum - 1)*PNum);
    Eigen::MatrixXd H(3*PNum, 3*PNum);
    P.setIdentity();
    P -= Q*Q.transpose();
    H = W.transpose()*P*W;
    for(int i = 0; i < PNum; i++)
    {
        H.block(3*i, 3*i, 3, 3) = H.block(3*i, 3*i, 3, 3).inverse();
    }
    G.setIdentity();
    G -= W*H*W.transpose()*P;

    // AZ = b 其中A = PGV， b = PGC
    Eigen::VectorXd Z(6);
    Z = (P*G*V).colPivHouseholderQr().solve(P*G*C);
//    Z = (V).colPivHouseholderQr().solve(C);

    v = Z.head(3);
    g = Z.segment(3, 3);


//    std::cout << (A*X - S).norm() << std::endl;
//    std::cout << "estv:" << v.transpose() << ", gtv:" << gtv.transpose() << std::endl;
//    std::cout << "estg:" << g.transpose() << ", gtg:" << gtg.transpose() << std::endl;
//    std::cout << "gnorm:" << g.norm() << std::endl << std::endl;

    std::cout << "Estv:" << v.transpose() << std::endl;
    std::cout << "gtv:" << gtv.transpose() << std::endl;
    std::cout << "Errorv:" << (v - gtv).norm() << std::endl;
    std::cout << "Errorg:" << (g - gtg).norm() << std::endl;
    std::cout << "gnorm:" << g.norm() << std::endl << std::endl;
    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;

//    std::cout << X.transpose() << std::endl;
*/


    // Ceres 优化t12

    std::vector<double*> pointers;// 装所有lamda和bg，g

    for(int i = 0; i < PNum; i++)
    {
        double* lamda = new double[1];
        lamda[0] = 1;
//        lamda[0] = X[i + 6];
        pointers.push_back(lamda);
    }

    for(int i = 0; i < FrmNum - 1; i++)
    {
        double* t21 = new double[3];
        t21[0] = 0;
        t21[1] = 0;
        t21[2] = 0;
        pointers.push_back(t21);
    }

    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    double** parameters = new double*[2];
    parameters[0] = bg_ptr;
    parameters[1] = nullptr;
    parameters[2] = nullptr;

    ceres::Problem problem;

    for(int i = 0; i < PNum; i++)
    {
        parameters[2] = pointers[i];
        Eigen::Vector3d u1 = AlignedPts[0][i];
        for(int j = 1; j < FrmNum; j++)
        {
            parameters[1] = pointers[PNum + j - 1];
            const double delt = j*dt;
            Eigen::Vector3d u2 = AlignedPts[j][i];
            ceres::CostFunction* cost_function = new ReprojectionCostFunctionR21(u1, u2, pInts[j - 1], delt, Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters, 3);
        }
    }

    ceres::Solver::Options options;

//    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.max_solver_time_in_seconds = 0.2;

    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    bool converged = (summary.termination_type == ceres::CONVERGENCE);


//    if(converged)
//    {
//        bool failed = false;
//        for(int i = 0; i < PNum; i++)
//        {
//            double lamda = pointers[i][0];
//            if(lamda < 0)
//            {
//                failed = true;
//                break;
//            }
//        }
//        if(failed)
//        {
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            delete[] parameters;
//            for (double* ptr : pointers)
//                delete[] ptr;
//            return;
//        }
//    }


    if(converged)
    {

//        for(int i = 0; i < PNum; i++)
//        {
//            double lamda = pointers[i][0];
//            std::cout << 1/lamda << ", ";
//        }

        std::cout << std::endl;

        std::cout << "PointsNum: " << PointNum << std::endl;
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "Est:" << bg.transpose() << std::endl;
        std::cout << "GT :" << gtbg.transpose() << std::endl;

//        Eigen::Vector3d esttrans(pointers[PNum][0], pointers[PNum][1], pointers[PNum][2]);
//        std::cout << "EstTrans:::" << esttrans.transpose() << std::endl;
//        std::cout << "gtTrans:::" << gttrans.transpose() << std::endl;


        ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
        StaError += (gtbg - bg).norm();
        StaCount ++;

        int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
        if(idi > 99)
            idi = 99;
        statistical[idi] ++;
    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }


/*
    std::vector<Eigen::Vector3d> Vect;
    for(int i = 0; i < FrmNum - 1; i++)
    {
        double* t12 = pointers[PNum + i];
        Eigen::Vector3d t(t12[0], t12[1], t12[2]);
        Vect.push_back(t);
    }

    // 线性求解


    Eigen::Vector3d v;
    Eigen::Vector3d g;

    Eigen::MatrixXd A(6, 6);
    Eigen::VectorXd X(6);
    Eigen::VectorXd B(6);
    A.setZero();
    B.setZero();
    double dt = 1.0f / InitRate;

    for(int i = 0; i < FrmNum - 1; i++)
    {
        double delt = (FrmNum - 1 - i)*dt;
        Eigen::MatrixXd SubA(3, 6);
        Eigen::Vector3d SubB;
        Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(bg);
        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(bg, Eigen::Vector3d::Zero());
        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(bg, Eigen::Vector3d::Zero());
//        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(bg, gtba);
//        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(bg, gtba);

        SubA.block(0, 0, 3, 3) = DR*delt;
        SubA.block(0, 3, 3, 3) = - 0.5*DR*delt*delt;
//        SubB = gttrans[i] - DP + DV*delt;// 用gttrans计算
        SubB = Vect[i] - DP + DV*delt;// 用估计的trans计算

        A += SubA.transpose()*SubA;
        B += SubA.transpose()*SubB;
    }
    X = A.ldlt().solve(B);

    v = X.head(3);
    g = X.tail(3);

    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << ", " << g.norm() << std::endl;

    gravity_error += 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI;

    std::cout << "estv:" << v.transpose() << std::endl;
    std::cout << "gtv:" << gtv.transpose() << std::endl;

    */

    // 加入ba优化，效果很烂
//    Eigen::Vector3d v;
//    Eigen::Vector3d g;
//    Eigen::Vector3d ba;

//    Eigen::MatrixXd A(9, 9);
//    Eigen::VectorXd X(9);
//    Eigen::VectorXd B(9);
//    A.setZero();
//    B.setZero();
//    double dt = 1.0f / InitRate;

//    for(int i = 0; i < FrmNum - 1; i++)
//    {
//        double delt = (FrmNum - 1 - i)*dt;
//        Eigen::MatrixXd SubA(3, 9);
//        Eigen::Vector3d SubB;
//        Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(bg);
//        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(bg, Eigen::Vector3d::Zero());
//        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(bg, Eigen::Vector3d::Zero());

//        SubA.block(0, 0, 3, 3) = DR*delt;
//        SubA.block(0, 3, 3, 3) = - 0.5*DR*delt*delt;
//        SubA.block(0, 6, 3, 3) = pInts[i].JPa - pInts[i].JVa*delt;
//        SubB = Vect[i] - DP + DV*delt;

//        A += SubA.transpose()*SubA;
//        B += SubA.transpose()*SubB;
//    }
//    X = A.ldlt().solve(B);

//    v = X.head(3);
//    g = X.segment(3, 3);
//    ba = X.tail(3);

/*
    delete[] parameters;
    for (double* ptr : pointers)
    {
        delete[] ptr;
    }
*/

}



void Reprojection_PAL_Solution(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                               const std::vector<std::vector<Eigen::Vector6d>> &AlignedLns,
                               std::vector<IMU::Preintegrated> &pInts,
                               const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                               const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                               const Eigen::Vector3d &gtg)
{
    const int PNum = AlignedPts.front().size();
    const int LNum = AlignedLns.front().size();
    const int FrmNum = FrameNum + 1;

    // Ceres 优化所有项
    // 优化每一帧到最后一帧的t12

    std::vector<double*> para_points;//点特征的深度变量
    for(int i = 0; i < PNum; i++)
    {
        double* lamda = new double[1];
        lamda[0] = 1;
        para_points.push_back(lamda);
    }

    std::vector<double*> para_lines;//线特征的深度变量
    for(int i = 0; i < 2*LNum; i++)
    {
        double* lamda = new double[1];
        lamda[0] = 1;
        para_lines.push_back(lamda);
    }

    std::vector<double*> para_t;//平移变量
    for(int i = 0; i < FrmNum - 1; i++)
    {
        double* t12 = new double[3];
        t12[0] = 0;
        t12[1] = 0;
        t12[2] = 0;
        para_t.push_back(t12);
    }

    double* bg_ptr = new double[3];
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    ceres::Problem problem;

    // 先加入点特征
    double** parameters = new double*[3];
    parameters[0] = bg_ptr;
    parameters[1] = nullptr;
    parameters[2] = nullptr;
    for(int i = 0; i < PNum; i++)
    {
        parameters[2] = para_points[i];
        Eigen::Vector3d u2 = AlignedPts.back()[i];

        for(int j = 0; j < FrmNum - 1; j++)
        {
            parameters[1] = para_t[j];
            Eigen::Vector3d u1 = AlignedPts[j][i];
            ceres::CostFunction* cost_function = new ReprojectionCostFunction(u1, u2, pInts[j], Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters, 3);
        }
    }

    // 再加入线特征
    double** parameters2 = new double*[4];
    parameters2[0] = bg_ptr;
    parameters2[1] = nullptr;
    parameters2[2] = nullptr;
    parameters2[3] = nullptr;

    for(int i = 0; i < 2*LNum; i+=2)
    {
        parameters2[2] = para_lines[i];
        parameters2[3] = para_lines[i + 1];
        Eigen::Vector6d l2 = AlignedLns.back()[i/2];

        for(int j = 0; j < FrmNum - 1; j++)
        {
            parameters2[1] = para_t[j];
            Eigen::Vector6d l1 = AlignedLns[j][i/2];
            ceres::CostFunction* cost_function = new ReprojectionCostFunctionLines(
                        l1.head(3), l1.tail(3), l2.head(3), l2.tail(3), pInts[j], Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters2, 4);
        }
    }



    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    bool converged = (summary.termination_type == ceres::CONVERGENCE);

/*
    if(converged)
    {
        bool failed = false;
        for(int i = 0; i < PNum; i++)
        {
            double lamda = para_points[i][0];
            if(lamda < 0)
            {
                failed = true;
                break;
            }
        }

        for(int i = 0; i < 2*LNum; i++)
        {
            double lamda = para_lines[i][0];
            if(lamda < 0)
            {
                failed = true;
                break;
            }
        }

        if(failed)
        {
            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
            delete[] parameters;
            delete[] parameters2;
            for(double* ptr : para_points)
                delete[] ptr;
            for(double* ptr : para_lines)
                delete[] ptr;
            for(double* ptr : para_t)
                delete[] ptr;
            return;
        }
    }
*/

    if(converged)
    {
        std::cout << "PointsNum: " << PointNum << std::endl;
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "Est:" << bg.transpose() << std::endl;
        std::cout << "GT :" << gtbg.transpose() << std::endl;


        ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
        StaError += (gtbg - bg).norm();
        StaCount ++;

        int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
        if(idi > 99)
            idi = 99;
        statistical[idi] ++;
    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }



    std::vector<Eigen::Vector3d> Vect;
    for(int i = 0; i < FrmNum - 1; i++)
    {
        double* t12 = para_t[i];
        Eigen::Vector3d t(t12[0], t12[1], t12[2]);
        Vect.push_back(t);
    }

    // 线性求解


    Eigen::Vector3d v;
    Eigen::Vector3d g;

    Eigen::MatrixXd A(6, 6);
    Eigen::VectorXd X(6);
    Eigen::VectorXd B(6);
    A.setZero();
    B.setZero();
    double dt = 1.0f / InitRate;

    for(int i = 0; i < FrmNum - 1; i++)
    {
        double delt = (FrmNum - 1 - i)*dt;
        Eigen::MatrixXd SubA(3, 6);
        Eigen::Vector3d SubB;
        Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(bg);
//        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(bg, Eigen::Vector3d::Zero());
//        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(bg, Eigen::Vector3d::Zero());
        Eigen::Vector3d DV = pInts[i].GetDeltaVelocity(bg, gtba);
        Eigen::Vector3d DP = pInts[i].GetDeltaPosition(bg, gtba);

        SubA.block(0, 0, 3, 3) = DR*delt;
        SubA.block(0, 3, 3, 3) = - 0.5*DR*delt*delt;
        SubB = Vect[i] - DP + DV*delt;

        A += SubA.transpose()*SubA;
        B += SubA.transpose()*SubB;
    }
    X = A.ldlt().solve(B);

    v = X.head(3);
    g = X.tail(3);

    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << ", " << g.norm() << std::endl;

    delete[] parameters;
    delete[] parameters2;
    for(double* ptr : para_points)
        delete[] ptr;
    for(double* ptr : para_lines)
        delete[] ptr;
    for(double* ptr : para_t)
        delete[] ptr;


}





void Reprojection_Closeform_Solution(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg)
{

    int PNum = AlignedPts.front().size();
    int FrmNum = FrameNum + 1;

    const Eigen::Matrix3d Rbc = Tbc.rotation();
    const Eigen::Matrix3d Rcb = Rbc.transpose();
    const Eigen::Vector3d tbc = Tbc.translation();
    const Eigen::Vector3d tcb = - Rcb*tbc;
/*
    Eigen::Vector3d bg;
    bg.setZero();

    Eigen::MatrixXd A(2*(FrmNum - 1)*PNum, PNum + 3*(FrmNum - 1));
    Eigen::VectorXd B(2*(FrmNum - 1)*PNum);
    A.setZero();
    B.setZero();

    Eigen::MatrixXd Ax(2*(FrmNum - 1)*PNum, PNum + 3*(FrmNum - 1));
    Eigen::MatrixXd Ay(2*(FrmNum - 1)*PNum, PNum + 3*(FrmNum - 1));
    Eigen::MatrixXd Az(2*(FrmNum - 1)*PNum, PNum + 3*(FrmNum - 1));
    Eigen::VectorXd Bx(2*(FrmNum - 1)*PNum);
    Eigen::VectorXd By(2*(FrmNum - 1)*PNum);
    Eigen::VectorXd Bz(2*(FrmNum - 1)*PNum);
    Ax.setZero();
    Ay.setZero();
    Az.setZero();
    Bx.setZero();
    By.setZero();
    Bz.setZero();

    int cur_row = 0;// 矩阵中确定的帧先算出来
    for(int i = 0; i < FrmNum - 1; i++)
    {
        for(int j = 0; j < PNum; j++)
        {
            Eigen::Vector3d u1 = AlignedPts[i][j];
            A.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(0, 0, 1, 3) - u1[0]*Rcb.block(2, 0, 1, 3);
            Ax.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(0, 0, 1, 3) - u1[0]*Rcb.block(2, 0, 1, 3);
            Ay.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(0, 0, 1, 3) - u1[0]*Rcb.block(2, 0, 1, 3);
            Az.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(0, 0, 1, 3) - u1[0]*Rcb.block(2, 0, 1, 3);
            ++ cur_row;
            A.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(1, 0, 1, 3) - u1[1]*Rcb.block(2, 0, 1, 3);
            Ax.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(1, 0, 1, 3) - u1[1]*Rcb.block(2, 0, 1, 3);
            Ay.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(1, 0, 1, 3) - u1[1]*Rcb.block(2, 0, 1, 3);
            Az.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(1, 0, 1, 3) - u1[1]*Rcb.block(2, 0, 1, 3);
            ++ cur_row;
        }
    }

    double delta = 1e-4;
//    for(int iter = 0; iter < 10; iter ++)
//    {
        cur_row = 0;
        for(int i = 0; i < FrmNum - 1; i++)
        {
            Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(gtbg);
            Eigen::Matrix3d PreX1 = Rcb*DR*Rbc;
            Eigen::Vector3d X2 = Rcb*DR*tbc + tcb;

            Eigen::Matrix3d DRx = pInts[i].GetDeltaRotation(bg + Eigen::Vector3d(delta, 0, 0));
            Eigen::Matrix3d DRy = pInts[i].GetDeltaRotation(bg + Eigen::Vector3d(0, delta, 0));
            Eigen::Matrix3d DRz = pInts[i].GetDeltaRotation(bg + Eigen::Vector3d(0, 0, delta));
            Eigen::Matrix3d PreX1x = Rcb*DRx*Rbc;
            Eigen::Vector3d X2x = Rcb*DRx*tbc + tcb;
            Eigen::Matrix3d PreX1y = Rcb*DRy*Rbc;
            Eigen::Vector3d X2y = Rcb*DRy*tbc + tcb;
            Eigen::Matrix3d PreX1z = Rcb*DRz*Rbc;
            Eigen::Vector3d X2z = Rcb*DRz*tbc + tcb;

            for(int j = 0; j < PNum; j++)
            {
                Eigen::Vector3d u1 = AlignedPts[i][j];

                Eigen::Vector3d X1 = PreX1*AlignedPts.back()[j];
                A(cur_row, j) = X1[0] - u1[0]*X1[2];
    //            A.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(0, 0, 1, 3) - u1[0]*Rcb.block(2, 0, 1, 3);
                B(cur_row) = u1[0]*X2[2] - X2[0];

                Eigen::Vector3d X1x = PreX1x*AlignedPts.back()[j];
                Eigen::Vector3d X1y = PreX1y*AlignedPts.back()[j];
                Eigen::Vector3d X1z = PreX1z*AlignedPts.back()[j];
                Ax(cur_row, j) = X1x[0] - u1[0]*X1x[2];
    //            Ax.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(0, 0, 1, 3) - u1[0]*Rcb.block(2, 0, 1, 3);
                Bx(cur_row) = u1[0]*X2x[2] - X2x[0];
                Ay(cur_row, j) = X1y[0] - u1[0]*X1y[2];
    //            Ay.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(0, 0, 1, 3) - u1[0]*Rcb.block(2, 0, 1, 3);
                By(cur_row) = u1[0]*X2y[2] - X2y[0];
                Az(cur_row, j) = X1z[0] - u1[0]*X1z[2];
    //            Az.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(0, 0, 1, 3) - u1[0]*Rcb.block(2, 0, 1, 3);
                Bz(cur_row) = u1[0]*X2z[2] - X2z[0];

                ++ cur_row;

                A(cur_row, j) = X1[1] - u1[1]*X1[2];
    //            A.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(1, 0, 1, 3) - u1[1]*Rcb.block(2, 0, 1, 3);
                B(cur_row) = u1[1]*X2[2] - X2[1];

                Ax(cur_row, j) = X1x[1] - u1[1]*X1x[2];
    //            Ax.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(1, 0, 1, 3) - u1[1]*Rcb.block(2, 0, 1, 3);
                Bx(cur_row) = u1[1]*X2x[2] - X2x[1];
                Ay(cur_row, j) = X1y[1] - u1[1]*X1y[2];
    //            Ay.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(1, 0, 1, 3) - u1[1]*Rcb.block(2, 0, 1, 3);
                By(cur_row) = u1[1]*X2y[2] - X2y[1];
                Az(cur_row, j) = X1z[1] - u1[1]*X1z[2];
    //            Az.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(1, 0, 1, 3) - u1[1]*Rcb.block(2, 0, 1, 3);
                Bz(cur_row) = u1[1]*X2z[2] - X2z[1];
                ++ cur_row;
            }
        }

        Eigen::VectorXd X(PNum + 3*(FrmNum - 1));
        Eigen::VectorXd Xx(PNum + 3*(FrmNum - 1));
        Eigen::VectorXd Xy(PNum + 3*(FrmNum - 1));
        Eigen::VectorXd Xz(PNum + 3*(FrmNum - 1));
        X = A.householderQr().solve(B);

        std::cout << "X ::" << X.transpose() << std::endl;
        Xx = Ax.householderQr().solve(Bx);
        Xy = Ay.householderQr().solve(By);
        Xz = Az.householderQr().solve(Bz);

        Eigen::VectorXd f(2*(FrmNum - 1)*PNum);
        f = A*X - B;

        Eigen::MatrixXd JT(2*(FrmNum - 1)*PNum, 3);
        JT.col(0) = ((Ax*Xx - Bx) - f)/delta;
        JT.col(1) = ((Ay*Xy - By) - f)/delta;
        JT.col(2) = ((Az*Xz - Bz) - f)/delta;
        Eigen::Vector3d dx = (JT.transpose()*JT).ldlt().solve( - JT.transpose()*f);
        bg += dx;

//        std::cout << "Iter:" << iter << std::endl;
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "Est:" << bg.transpose() << std::endl;
        std::cout << "GT :" << gtbg.transpose() << std::endl;
//    }






//    std::cout << "PointsNum: " << PointNum << std::endl;
//    std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
//    std::cout << "Est:" << bg.transpose() << std::endl;
//    std::cout << "GT :" << gtbg.transpose() << std::endl;


    ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
    StaError += (gtbg - bg).norm();
    StaCount ++;

    int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
    if(idi > 99)
        idi = 99;
    statistical[idi] ++;
*/










    /*
    Eigen::Vector3d bg;
    bg.setZero();

    Eigen::MatrixXd A(2*(FrmNum - 1)*PNum, PNum + 3*(FrmNum - 1));
    Eigen::VectorXd B(2*(FrmNum - 1)*PNum);
    A.setZero();
    B.setZero();

    int cur_row = 0;
    for(int i = 0; i < FrmNum - 1; i++)
    {
        Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(gtbg);
        Eigen::Matrix3d PreX1 = Rcb*DR*Rbc;
        Eigen::Vector3d X2 = Rcb*DR*tbc + tcb;

        for(int j = 0; j < PNum; j++)
        {
            Eigen::Vector3d u1 = AlignedPts[i][j];

            Eigen::Vector3d X1 = PreX1*AlignedPts.back()[j];
            A(cur_row, j) = X1[0] - u1[0]*X1[2];
            A.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(0, 0, 1, 3) - u1[0]*Rcb.block(2, 0, 1, 3);
            B(cur_row) = u1[0]*X2[2] - X2[0];

            ++ cur_row;

            A(cur_row, j) = X1[1] - u1[1]*X1[2];
            A.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(1, 0, 1, 3) - u1[1]*Rcb.block(2, 0, 1, 3);
            B(cur_row) = u1[1]*X2[2] - X2[1];
            ++ cur_row;
        }
    }

    Eigen::VectorXd X(PNum + 3*(FrmNum - 1));
    X = A.householderQr().solve(B);

    std::cout << "X ::" << X.transpose() << std::endl;

    Eigen::VectorXd f(2*(FrmNum - 1)*PNum);
    f = A*X - B;

    std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
    std::cout << "Est:" << bg.transpose() << std::endl;
    std::cout << "GT :" << gtbg.transpose() << std::endl;
*/



    Eigen::Vector3d bg;
    bg.setZero();

    Eigen::MatrixXd A(2*(FrmNum - 1)*PNum, PNum + 3*(FrmNum - 1));
    Eigen::VectorXd B(2*(FrmNum - 1)*PNum);
    A.setZero();
    B.setZero();

    int cur_row = 0;
    for(int i = 0; i < FrmNum - 1; i++)
    {
        Eigen::Matrix3d DR = pInts[i].GetDeltaRotation(gtbg);
        Eigen::Matrix3d PreX1 = Rcb*DR*Rbc;
        Eigen::Vector3d X2 = Rcb*DR*tbc + tcb;

        for(int j = 0; j < PNum; j++)
        {
            Eigen::Vector3d u1 = AlignedPts[i][j];

            Eigen::Vector3d X1 = PreX1*AlignedPts.back()[j];
            A(cur_row, j) = X1[0] - u1[0]*X1[2];
            A.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(0, 0, 1, 3) - u1[0]*Rcb.block(2, 0, 1, 3);
            B(cur_row) = u1[0]*X2[2] - X2[0];

            ++ cur_row;

            A(cur_row, j) = X1[1] - u1[1]*X1[2];
            A.block(cur_row, PNum + 3*i, 1, 3) = Rcb.block(1, 0, 1, 3) - u1[1]*Rcb.block(2, 0, 1, 3);
            B(cur_row) = u1[1]*X2[2] - X2[1];
            ++ cur_row;
        }
    }

    Eigen::VectorXd X(PNum + 3*(FrmNum - 1));
    X = A.householderQr().solve(B);

    std::cout << "X ::" << X.transpose() << std::endl;

    Eigen::VectorXd f(2*(FrmNum - 1)*PNum);
    f = A*X - B;

    std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
    std::cout << "Est:" << bg.transpose() << std::endl;
    std::cout << "GT :" << gtbg.transpose() << std::endl;


}








void Reprojection_3D_Solution(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                        std::vector<IMU::Preintegrated> &pInts,
                        const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                        const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                        const Eigen::Vector3d &gtg, std::vector<Eigen::Vector3d> &gttrans)
{
    int PNum = AlignedPts.front().size();
    int FrmNum = FrameNum + 1;


    // Ceres 优化所有项
    // 优化每一帧到最后一帧的t12


    std::vector<double*> pointers;// 装所有lamda和bg，g

    for(int i = 0; i < PNum; i++)
    {
        double* lamda = new double[3];
        lamda[0] = AlignedPts.back()[i].x();
        lamda[1] = AlignedPts.back()[i].y();
        lamda[2] = 1;
        pointers.push_back(lamda);
    }

    for(int i = 0; i < FrmNum - 1; i++)
    {
        double* t12 = new double[3];
        t12[0] = 0;
        t12[1] = 0;
        t12[2] = 0;
        pointers.push_back(t12);
    }

    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();


    double** parameters = new double*[3];
    parameters[0] = bg_ptr;
    parameters[1] = nullptr;
    parameters[2] = nullptr;

    ceres::Problem problem;

    for(int i = 0; i < PNum; i++)
    {
        parameters[2] = pointers[i];
        for(int j = 0; j < FrmNum - 1; j++)
        {
            parameters[1] = pointers[PNum + j];
            Eigen::Vector3d u1 = AlignedPts[j][i];
            ceres::CostFunction* cost_function = new Reprojection_3D_CostFunction(u1, pInts[j], Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters, 3);
        }
    }


    double** parameters2 = new double*[1];
    for(int i = 0; i < PNum; i++)
    {
        parameters2[0] = pointers[i];
        Eigen::Vector3d u2 = AlignedPts.back()[i];
        ceres::CostFunction* cost_function = new Reprojection_3D_Self_CostFunction(u2);
        problem.AddResidualBlock(cost_function, nullptr, parameters2, 1);
    }

    ceres::Solver::Options options;

//    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.max_solver_time_in_seconds = 0.2;

    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    bool converged = (summary.termination_type == ceres::CONVERGENCE);


//    if(converged)
//    {
//        bool failed = false;
//        for(int i = 0; i < PNum; i++)
//        {
//            double lamda = pointers[i][0];
//            if(lamda < 0)
//            {
//                failed = true;
//                break;
//            }
//        }
//        if(failed)
//        {
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            std::cout << "lamda is lower than 0!!!!!!!!!!!!!" << std::endl;
//            delete[] parameters;
//            for (double* ptr : pointers)
//                delete[] ptr;
//            return;
//        }
//    }


    if(converged)
    {

        for(int i = 0; i < PNum; i++)
        {
            double lamda = pointers[i][2];
            std::cout << 1/lamda << ", ";
        }

        std::cout << std::endl;

        std::cout << "PointsNum: " << PointNum << std::endl;
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "Est:" << bg.transpose() << std::endl;
        std::cout << "GT :" << gtbg.transpose() << std::endl;

//        Eigen::Vector3d esttrans(pointers[PNum][0], pointers[PNum][1], pointers[PNum][2]);
//        std::cout << "EstTrans:::" << esttrans.transpose() << std::endl;
//        std::cout << "gtTrans:::" << gttrans.transpose() << std::endl;


        ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
        StaError += (gtbg - bg).norm();
        StaCount ++;

        int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
        if(idi > 99)
            idi = 99;
        statistical[idi] ++;
    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }
}

/*
void CloseForm_Solution_PAL(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                            const std::vector<std::vector<Eigen::Vector6d>> &AlignedLns,
                            std::vector<IMU::Preintegrated> &pInts,
                            const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                            const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                            const Eigen::Vector3d &gtg,
                            const std::vector<IMU::Preintegrated> &pIntsgt)
{
    const int PNum = PointNum;
    const int LNum = LineNum;
    const int FrmNum = FrameNum + 1;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d Rbc = Tbc.rotation();
    const Eigen::Vector3d tbc = Tbc.translation();
    const double dt = 1.0f / InitRate;
    int cur_row = 0;
    std::cout << std::endl;

    Timer timer;
    timer.Start();

    // 以下为点特征
    // 加上g一起计算·改，用论文中方式计算,时间会更快
/*
    Eigen::Vector3d g;
    Eigen::Vector3d v;

    Eigen::MatrixXd AP(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd XP(6 + PNum*FrmNum);
    Eigen::VectorXd BP(3*(FrmNum - 1)*PNum);
    AP.setZero();
    BP.setZero();

    cur_row = 0;
    std::vector<std::vector<Eigen::Vector3d>> VecU;
    for(int i = 0; i < PNum; i++)
    {
        std::vector<Eigen::Vector3d> Vecu;
        Eigen::Vector3d u1 = Rbc*AlignedPts[0][i];
        Vecu.push_back(u1);
        for(int j = 1; j < FrmNum; j++)
        {
//            Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(gtbg);
//            Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(gtbg, gtba);
//            Eigen::Matrix3d DR = pIntsgt[j - 1].dR;
//            Eigen::Vector3d DP = pIntsgt[j - 1].dP;
            Eigen::Matrix3d DR = pInts[j - 1].dR;
            Eigen::Vector3d DP = pInts[j - 1].dP;
            double delt = double(j)*dt;
            Eigen::Vector3d u2 = Rbc*AlignedPts[j][i];
            AP.block(cur_row, 0, 3, 3) = 0.5*delt*delt*I;
            AP.block(cur_row, 3, 3, 3) = delt*I;
            AP.block(cur_row, i*FrmNum + 6, 3, 1) = - u1;
            AP.block(cur_row, i*FrmNum + j + 6, 3, 1) = DR*u2;
            BP.segment(cur_row, 3) = - DP - (DR - I)*tbc;
            Vecu.push_back(u2);
            cur_row += 3;
        }
        VecU.push_back(Vecu);
    }

    XP = AP.colPivHouseholderQr().solve(BP);


    g = XP.head(3);
    v = XP.segment(3, 3);


//    std::cout << X.transpose() << std::endl;

//    std::cout << "v:" << v.transpose() << std::endl;
//    std::cout << "gtv:" << gtv.transpose() << std::endl;
    std::cout << "Before:::: " << std::endl;
    std::cout << "Errorv:" << (v - gtv).norm() << std::endl;
    std::cout << "gnorm:" << g.norm() << std::endl;
    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;


    // Ceres 优化所有项


    std::vector<double*> pointers;// 装所有lamda和bg，g

    for(int i = 0; i < PNum*FrmNum; i++)
    {
        double* lamda = new double[1];
        lamda[0] = XP[i + 6];
//        lamda[0] = 1;
        pointers.push_back(lamda);
    }

    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    double* v0_ptr = new double[3];
    pointers.push_back(v0_ptr);
    Eigen::Map<Eigen::Vector3d> v0(v0_ptr);
    v0 = v;

//    double* Rwg_ptr = new double[9];
//    pointers.push_back(Rwg_ptr);
//    Eigen::Map<Eigen::Matrix3d> Rwg(Rwg_ptr);
//    Rwg.setIdentity();

//    Eigen::Vector3d dirG = g.normalized();
//    const Eigen::Vector3d gI = IMU::GRAVITY_VECTOR.normalized();
//    const Eigen::Vector3d vI = gI.cross(dirG);
//    const double cos_theta = gI.dot(dirG);
//    const double theta = std::acos(cos_theta);
//    Rwg = ExpSO3(vI*theta/vI.norm());

    double* g_ptr = new double[3];
    pointers.push_back(g_ptr);
    Eigen::Map<Eigen::Vector3d> g0(g_ptr);
//    g0.setZero();
    g0 = g;

    double** parameters = new double*[5];
    parameters[0] = bg_ptr;
//    parameters[1] = Rwg_ptr;
    parameters[1] = g_ptr;
    parameters[2] = v0_ptr;
    parameters[3] = nullptr;
    parameters[4] = nullptr;

    ceres::Problem problem;

    for(int i = 0; i < PNum; i++)
    {
        parameters[3] = pointers[i*FrmNum];//lamda0
        Eigen::Vector3d u1 = VecU[i][0];
        for(int j = 1; j < FrmNum; j++)
        {
            double delt = j*dt;
            Eigen::Vector3d u2 = VecU[i][j];
            parameters[4] = pointers[i*FrmNum + j];//lamda1
            ceres::CostFunction* cost_function = new CloseformCostFunction(u1, u2, pInts[j - 1], delt, gtba, Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters, 5);
//            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), parameters, 5);
//            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(10.0), parameters, 5);
//            problem.SetParameterLowerBound(parameters[3], 0, 1e-5);
//            problem.SetParameterLowerBound(parameters[4], 0, 1e-5);
        }
    }

    // gyro bias = 0的先验，感觉用处不大
//    ceres::CostFunction* prior_cost_function = new BiasPriorCostFunction(1);//prior先验权重10e5
//    problem.AddResidualBlock(prior_cost_function, nullptr, bg_ptr);

//    GravityParameterization* gravity_local_parameterization = new GravityParameterization;
//    problem.SetParameterization(Rwg_ptr, gravity_local_parameterization);


    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);


    bool converged = (summary.termination_type == ceres::CONVERGENCE);

//    const Eigen::Vector3d g0 = Rwg*IMU::GRAVITY_VECTOR;

    if(converged)
    {
        std::cout << "After:::: " << std::endl;
//        std::cout << "PointsNum: " << PointNum << std::endl;
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "Est:" << bg.transpose() << std::endl;
        std::cout << "GT :" << gtbg.transpose() << std::endl;
        std::cout << "Errorv:" << (v0 - gtv).norm() << std::endl;
        std::cout << "gnorm:" << g0.norm() << std::endl;
        std::cout << "DirectionError:" << 180.*std::acos(g0.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;


        StaTime += timer.ElapsedSeconds();
        StaErrorv += (v0 - gtv).norm();
        gravity_error += 180.*std::acos(g0.normalized().dot(gtg.normalized()))/EIGEN_PI;

        ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
        StaError += (gtbg - bg).norm();
        StaCount ++;

        int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
        if(idi > 99)
            idi = 99;
        statistical[idi] ++;

    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }
    delete[] parameters;
    for (double* ptr : pointers)
    {
        delete[] ptr;
    }
*/



    // 以下为线特征
    // 原始方法 0.011s左右


//    std::vector<Eigen::Vector3d> normd;
//    for(int i = 0; i < LNum; i++)
//    {
//        Eigen::MatrixXd AD(3*(FrmNum - 1), 2*FrmNum - 1);
//        Eigen::VectorXd XD(2*FrmNum - 1);
//        Eigen::VectorXd BD(3*(FrmNum - 1));
//        AD.setZero();
//        BD.setZero();
//        Eigen::Vector3d p1 = Rbc*(AlignedLns[0][i].tail(3) - AlignedLns[0][i].head(3)).normalized();
//        Eigen::Vector3d q1 = Rbc*AlignedLns[0][i].head(3).normalized();
//        for(int j = 0; j < FrmNum - 1; j++)
//        {
//            Eigen::Vector3d p2 = Rbc*(AlignedLns[j + 1][i].tail(3) - AlignedLns[j + 1][i].head(3)).normalized();
//            Eigen::Vector3d q2 = Rbc*AlignedLns[j + 1][i].head(3).normalized();
//            Eigen::Matrix3d DR = pInts[j].GetDeltaRotation(gtbg);
//            AD.block(3*j, 0, 3, 1) = q1;
//            AD.block(3*j, 2*j + 1, 3, 1) = DR*p2;
//            AD.block(3*j, 2*j + 2, 3, 1) = DR*q2;
//            BD.segment(3*j, 3) = - p1;
//        }
//        XD = AD.colPivHouseholderQr().solve(BD);
//        Eigen::Vector3d d1 = (p1 + XD[0]*q1).normalized();
//        normd.push_back(d1);
//    }


    // 消元之后的方法
/*
    std::vector<Eigen::Vector3d> normd;
    std::vector<double> Vecbeta;
    std::vector<Eigen::Vector3d> Vecq1;
    std::vector<Eigen::Vector3d> Vecp1;
    std::vector<std::vector<Eigen::Matrix3d>> VecM;
    for(int i = 0; i < LNum; i++)
    {
        Eigen::VectorXd V(3*(FrmNum - 1));
        Eigen::MatrixXd W(3*(FrmNum - 1), FrmNum - 1);
        Eigen::MatrixXd Q(3*(FrmNum - 1), FrmNum - 1);
        Eigen::VectorXd BD(3*(FrmNum - 1));
        V.setZero();
        W.setZero();
        Q.setZero();
        BD.setZero();
        Eigen::Vector3d p1 = Rbc*(AlignedLns[0][i].tail(3) - AlignedLns[0][i].head(3)).normalized();
        Eigen::Vector3d q1 = Rbc*AlignedLns[0][i].head(3).normalized();
        for(int j = 0; j < FrmNum - 1; j++)
        {
            Eigen::Vector3d p2 = Rbc*(AlignedLns[j + 1][i].tail(3) - AlignedLns[j + 1][i].head(3)).normalized();
            Eigen::Vector3d q2 = Rbc*AlignedLns[j + 1][i].head(3).normalized();
//            Eigen::Matrix3d DR = pInts[j].GetDeltaRotation(gtbg);
            Eigen::Matrix3d DR = pInts[j].dR;
            V.segment(3*j, 3) = DR.transpose()*q1;
            W.block(3*j, j, 3, 1) = p2;
            Q.block(3*j, j, 3, 1) = q2;
            BD.segment(3*j, 3) = - DR.transpose()*p1;
        }

        Eigen::MatrixXd P(3*(FrmNum - 1), 3*(FrmNum - 1));
        P.setIdentity();
        P -= Q*Q.transpose();
        Eigen::MatrixXd invH(FrmNum - 1, FrmNum - 1);
        invH = W.transpose()*Q;
        for(int j = 0; j < FrmNum - 1; j++)
        {
            double diag = invH(j, j);
            invH(j, j) = 1.0f/(1.0f - diag*diag);
        }
        Eigen::MatrixXd LeftMulti(3*(FrmNum - 1), 3*(FrmNum - 1));
        LeftMulti = P - P*W*invH*W.transpose()*P;

        std::vector<Eigen::Matrix3d> Vecm;
        for(int j = 0; j < FrmNum - 1; j++)
        {
            Eigen::Matrix3d SubMulti = LeftMulti.block(3*j, 3*j, 3, 3);
            V.segment(3*j, 3) = SubMulti*V.segment(3*j, 3);
            BD.segment(3*j, 3) = SubMulti*BD.segment(3*j, 3);
            Vecm.push_back(SubMulti);
        }

        Eigen::VectorXd XD(1);
        XD = V.householderQr().solve(BD);
//        XD = V.colPivHouseholderQr().solve(BD);

        Eigen::Vector3d d1 = (p1 + XD[0]*q1).normalized();
        normd.push_back(d1);
        Vecbeta.push_back(XD[0]);
        Vecp1.push_back(p1);
        Vecq1.push_back(q1);
        VecM.push_back(Vecm);
//        std::cout << "XD: ";
//        std::cout << XD[0] << ", ";
    }



    std::vector<std::vector<Eigen::Vector3d>> VecN;

    Eigen::MatrixXd A(3*LNum*(FrmNum - 1), 6 + PNum*FrmNum);
    Eigen::VectorXd X(6 + PNum*FrmNum);
    Eigen::VectorXd B(3*LNum*(FrmNum - 1));
    A.setZero();
    B.setZero();
    cur_row = 0;
    for(int i = 0; i < LNum; i++)
    {
        std::vector<Eigen::Vector3d> Vecn;
        Eigen::Vector3d n1 = Skew(AlignedLns[0][i].head(3))*AlignedLns[0][i].tail(3);
        n1 = Rbc*n1.normalized();
        Vecn.push_back(n1);
//        Eigen::Matrix3d SkewRbcd1 = Skew(Vecp1[i] + Vecbeta[i]*Vecq1[i]);// 前面已经乘了Rbc
        Eigen::Matrix3d SkewRbcd1 = Skew(normd[i]);// 前面已经乘了Rbc
        for(int j = 1; j < FrmNum; j++)
        {
            double delt = j*dt;
            Eigen::Vector3d n2 = Skew(AlignedLns[j][i].head(3))*AlignedLns[j][i].tail(3);
            n2 = Rbc*n2.normalized();
//            Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(gtbg);
//            Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(gtbg, gtba);
            Eigen::Matrix3d DR = pInts[j - 1].dR;
            Eigen::Vector3d DP = pInts[j - 1].dP;
            A.block(cur_row, 0, 3, 3) = SkewRbcd1*delt;
            A.block(cur_row, 3, 3, 3) = 0.5*SkewRbcd1*delt*delt;
            A.block(cur_row, 6 + i, 3, 1) = n1;
            A.block(cur_row, 6 + j*LNum + i, 3, 1) = - DR*n2;
            B.segment(cur_row, 3) = - SkewRbcd1*((DR - I)*tbc + DP);
            cur_row += 3;
            Vecn.push_back(n2);
        }
        VecN.push_back(Vecn);
    }

    Eigen::Vector3d g;
    Eigen::Vector3d v;

    // 这里用householderQr解不出来？？？
//    X = A.householderQr().solve(B);
    X = A.colPivHouseholderQr().solve(B);
    v = X.head(3);
    g = X.segment(3, 3);

//    std::cout << "v:" << v.transpose() << std::endl;
//    std::cout << "gtv:" << gtv.transpose() << std::endl;
//    std::cout << "Errorv:" << (v - gtv).norm() << std::endl;
//    std::cout << "gnorm:" << g.norm() << std::endl;
//    std::cout << "DirectionError:" << 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;
//    StaTime += timer.ElapsedSeconds();
//    StaErrorv += (v - gtv).norm();
//    gravity_error += 180.*std::acos(g.normalized().dot(gtg.normalized()))/EIGEN_PI;
//    StaCount ++;


    // ceres线特征优化

    std::vector<double*> pointers;// 装所有lamda，beta和bg，g，v

    for(int i = 0; i < LNum*FrmNum; i++)
    {
        double* lamda = new double[1];
        lamda[0] = X[i + 6];
//        lamda[0] = 1;
        pointers.push_back(lamda);
    }

    for(int i = 0; i < LNum; i++)
    {
        double* beta = new double[1];
        beta[0] = Vecbeta[i];
//        beta[0] = 1;
        pointers.push_back(beta);
    }

    double* bg_ptr = new double[3];
    pointers.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    double* v0_ptr = new double[3];
    pointers.push_back(v0_ptr);
    Eigen::Map<Eigen::Vector3d> v0(v0_ptr);
    v0 = v;

//    double* Rwg_ptr = new double[9];
//    pointers.push_back(Rwg_ptr);
//    Eigen::Map<Eigen::Matrix3d> Rwg(Rwg_ptr);
//    Rwg.setIdentity();

//    Eigen::Vector3d dirG = g.normalized();
//    const Eigen::Vector3d gI = IMU::GRAVITY_VECTOR.normalized();
//    const Eigen::Vector3d vI = gI.cross(dirG);
//    const double cos_theta = gI.dot(dirG);
//    const double theta = std::acos(cos_theta);
//    Rwg = ExpSO3(vI*theta/vI.norm());

    double* g_ptr = new double[3];
    pointers.push_back(g_ptr);
    Eigen::Map<Eigen::Vector3d> g0(g_ptr);
//    g0.setZero();
    g0 = g;

    ceres::Problem problem;

    double** parameters = new double*[2]; // 首先是线的beta残差
    parameters[0] = bg_ptr;
    parameters[1] = nullptr;

//    for(int i = 0; i < LNum; i++)
//    {
//        parameters[1] = pointers[LNum*FrmNum + i];//beta
//        Eigen::Vector3d p1 = Vecp1[i];
//        Eigen::Vector3d q1 = Vecq1[i];
//        for(int j = 0; j < FrmNum - 1; j++)
//        {
//            Eigen::Matrix3d M = VecM[i][j];
//            ceres::CostFunction* cost_function = new CloseformCostFunctionLinesBeta(p1, q1, pInts[j], M);
//            problem.AddResidualBlock(cost_function, nullptr, parameters, 2);
//        }
//    }

    double** parameters2 = new double*[6];// 然后是线的残差
    parameters2[0] = bg_ptr;
//    parameters2[1] = Rwg_ptr;
    parameters2[1] = g_ptr;
    parameters2[2] = v0_ptr;
    parameters2[3] = nullptr;
    parameters2[4] = nullptr;
    parameters2[5] = nullptr;

    for(int i = 0; i < LNum; i++)
    {
        parameters2[3] = pointers[LNum*FrmNum + i];// beta
        parameters2[4] = pointers[i];// lamda1
        Eigen::Vector3d n1 = VecN[i][0];
        Eigen::Vector3d p1 = Vecp1[i];
        Eigen::Vector3d q1 = Vecq1[i];
        for(int j = 1; j < FrmNum; j++)
        {
            double delt = j*dt;
            Eigen::Vector3d n2 = VecN[i][j];
            parameters2[5] = pointers[j*LNum + i];//lamda2
            ceres::CostFunction* cost_function = new CloseformCostFunctionLines(n1, n2, p1, q1, pInts[j - 1], delt, gtba, Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters2, 6);
//            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), parameters, 5);
//            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(10.0), parameters, 5);
//            problem.SetParameterLowerBound(parameters[3], 0, 1e-5);
//            problem.SetParameterLowerBound(parameters[4], 0, 1e-5);
        }
    }

    // gyro bias = 0的先验，感觉用处不大
//    ceres::CostFunction* prior_cost_function = new BiasPriorCostFunction(1);//prior先验权重10e5
//    problem.AddResidualBlock(prior_cost_function, nullptr, bg_ptr);

//    GravityParameterization* gravity_local_parameterization = new GravityParameterization;
//    problem.SetParameterization(Rwg_ptr, gravity_local_parameterization);


    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);


    bool converged = (summary.termination_type == ceres::CONVERGENCE);

//    const Eigen::Vector3d g0 = Rwg*IMU::GRAVITY_VECTOR;

    if(converged)
    {
        std::cout << "After:::: " << std::endl;
//        std::cout << "PointsNum: " << PointNum << std::endl;
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "Est:" << bg.transpose() << std::endl;
        std::cout << "GT :" << gtbg.transpose() << std::endl;
        std::cout << "Errorv:" << (v0 - gtv).norm() << std::endl;
        std::cout << "gnorm:" << g0.norm() << std::endl;
        std::cout << "DirectionError:" << 180.*std::acos(g0.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;


        StaTime += timer.ElapsedSeconds();
        StaErrorv += (v0 - gtv).norm();
        gravity_error += 180.*std::acos(g0.normalized().dot(gtg.normalized()))/EIGEN_PI;

        ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
        StaError += (gtbg - bg).norm();
        StaCount ++;

        int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
        if(idi > 99)
            idi = 99;
        statistical[idi] ++;

    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }

    delete[] parameters;
    delete[] parameters2;
    for (double* ptr : pointers)
    {
        delete[] ptr;
    }
*/

//}




void CloseForm_Solution_PAL(Result &result, const std::vector<std::vector<Eigen::Vector3d>> &AlignedPts,
                            const std::vector<std::vector<Eigen::Vector6d>> &AlignedLns,
                            std::vector<IMU::Preintegrated> &pInts,
                            const Eigen::Isometry3d &Tbc, const Eigen::Vector3d &gtbg,
                            const Eigen::Vector3d &gtba, const Eigen::Vector3d &gtv,
                            const Eigen::Vector3d &gtg,
                            const std::vector<IMU::Preintegrated> &pIntsgt)
{
    const int PNum = PointNum;
    const int LNum = LineNum;
    const int FrmNum = FrameNum + 1;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d Rbc = Tbc.rotation();
    const Eigen::Vector3d tbc = Tbc.translation();
    const double dt = 1.0f / InitRate;
    int cur_row = 0;
    std::cout << std::endl;

    Timer timer;
    timer.Start();

    // 以下为点特征
    // 加上g一起计算·改，用论文中方式计算,时间会更快
    Eigen::Vector3d g;
    Eigen::Vector3d v;
    Eigen::MatrixXd AP(3*(FrmNum - 1)*PNum, 6 + PNum*FrmNum);
    Eigen::VectorXd XP(6 + PNum*FrmNum);
    Eigen::VectorXd BP(3*(FrmNum - 1)*PNum);
    AP.setZero();
    BP.setZero();
    cur_row = 0;
    std::vector<std::vector<Eigen::Vector3d>> VecU;
    for(int i = 0; i < PNum; i++)
    {
        std::vector<Eigen::Vector3d> Vecu;
        Eigen::Vector3d u1 = Rbc*AlignedPts[0][i];
        Vecu.push_back(u1);
        for(int j = 1; j < FrmNum; j++)
        {
//            Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(gtbg);
//            Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(gtbg, gtba);
            Eigen::Matrix3d DR = pInts[j - 1].dR;
            Eigen::Vector3d DP = pInts[j - 1].dP;
            double delt = double(j)*dt;
            Eigen::Vector3d u2 = Rbc*AlignedPts[j][i];
            AP.block(cur_row, 0, 3, 3) = 0.5*delt*delt*I;
            AP.block(cur_row, 3, 3, 3) = delt*I;
            AP.block(cur_row, i*FrmNum + 6, 3, 1) = - u1;
            AP.block(cur_row, i*FrmNum + j + 6, 3, 1) = DR*u2;
            BP.segment(cur_row, 3) = - DP - (DR - I)*tbc;
            Vecu.push_back(u2);
            cur_row += 3;
        }
        VecU.push_back(Vecu);
    }
    XP = AP.colPivHouseholderQr().solve(BP);
    g = XP.head(3);
    v = XP.segment(3, 3);

    // 线特征

    std::vector<Eigen::Vector3d> normd;
    std::vector<double> Vecbeta;
    std::vector<Eigen::Vector3d> Vecq1;
    std::vector<Eigen::Vector3d> Vecp1;
    std::vector<std::vector<Eigen::Matrix3d>> VecM;
    for(int i = 0; i < LNum; i++)
    {
        Eigen::VectorXd V(3*(FrmNum - 1));
        Eigen::MatrixXd W(3*(FrmNum - 1), FrmNum - 1);
        Eigen::MatrixXd Q(3*(FrmNum - 1), FrmNum - 1);
        Eigen::VectorXd BD(3*(FrmNum - 1));
        V.setZero();
        W.setZero();
        Q.setZero();
        BD.setZero();
        Eigen::Vector3d p1 = Rbc*(AlignedLns[0][i].tail(3) - AlignedLns[0][i].head(3)).normalized();
        Eigen::Vector3d q1 = Rbc*AlignedLns[0][i].head(3).normalized();
        for(int j = 0; j < FrmNum - 1; j++)
        {
            Eigen::Vector3d p2 = Rbc*(AlignedLns[j + 1][i].tail(3) - AlignedLns[j + 1][i].head(3)).normalized();
            Eigen::Vector3d q2 = Rbc*AlignedLns[j + 1][i].head(3).normalized();
//            Eigen::Matrix3d DR = pInts[j].GetDeltaRotation(gtbg);
            Eigen::Matrix3d DR = pInts[j].dR;
            V.segment(3*j, 3) = DR.transpose()*q1;
            W.block(3*j, j, 3, 1) = p2;
            Q.block(3*j, j, 3, 1) = q2;
            BD.segment(3*j, 3) = - DR.transpose()*p1;
        }

        Eigen::MatrixXd P(3*(FrmNum - 1), 3*(FrmNum - 1));
        P.setIdentity();
        P -= Q*Q.transpose();
        Eigen::MatrixXd invH(FrmNum - 1, FrmNum - 1);
        invH = W.transpose()*Q;
        for(int j = 0; j < FrmNum - 1; j++)
        {
            double diag = invH(j, j);
            invH(j, j) = 1.0f/(1.0f - diag*diag);
        }
        Eigen::MatrixXd LeftMulti(3*(FrmNum - 1), 3*(FrmNum - 1));
        LeftMulti = P - P*W*invH*W.transpose()*P;

        std::vector<Eigen::Matrix3d> Vecm;
        for(int j = 0; j < FrmNum - 1; j++)
        {
            Eigen::Matrix3d SubMulti = LeftMulti.block(3*j, 3*j, 3, 3);
            V.segment(3*j, 3) = SubMulti*V.segment(3*j, 3);
            BD.segment(3*j, 3) = SubMulti*BD.segment(3*j, 3);
            Vecm.push_back(SubMulti);
        }

        Eigen::VectorXd XD(1);
        XD = V.householderQr().solve(BD);
//        XD = V.colPivHouseholderQr().solve(BD);

        Eigen::Vector3d d1 = (p1 + XD[0]*q1).normalized();
        normd.push_back(d1);
        Vecbeta.push_back(XD[0]);
        Vecp1.push_back(p1);
        Vecq1.push_back(q1);
        VecM.push_back(Vecm);
    }

    std::vector<std::vector<Eigen::Vector3d>> VecN;
    Eigen::MatrixXd AL(3*LNum*(FrmNum - 1), 6 + PNum*FrmNum);
    Eigen::VectorXd XL(6 + PNum*FrmNum);
    Eigen::VectorXd BL(3*LNum*(FrmNum - 1));
    AL.setZero();
    BL.setZero();
    cur_row = 0;
    for(int i = 0; i < LNum; i++)
    {
        std::vector<Eigen::Vector3d> Vecn;
        Eigen::Vector3d n1 = Skew(AlignedLns[0][i].head(3))*AlignedLns[0][i].tail(3);
        n1 = Rbc*n1.normalized();
        Vecn.push_back(n1);
//        Eigen::Matrix3d SkewRbcd1 = Skew(Vecp1[i] + Vecbeta[i]*Vecq1[i]);// 前面已经乘了Rbc
        Eigen::Matrix3d SkewRbcd1 = Skew(normd[i]);// 前面已经乘了Rbc
        for(int j = 1; j < FrmNum; j++)
        {
            double delt = j*dt;
            Eigen::Vector3d n2 = Skew(AlignedLns[j][i].head(3))*AlignedLns[j][i].tail(3);
            n2 = Rbc*n2.normalized();
//            Eigen::Matrix3d DR = pInts[j - 1].GetDeltaRotation(gtbg);
//            Eigen::Vector3d DP = pInts[j - 1].GetDeltaPosition(gtbg, gtba);
            Eigen::Matrix3d DR = pInts[j - 1].dR;
            Eigen::Vector3d DP = pInts[j - 1].dP;
            AL.block(cur_row, 0, 3, 3) = SkewRbcd1*delt;
            AL.block(cur_row, 3, 3, 3) = 0.5*SkewRbcd1*delt*delt;
            AL.block(cur_row, 6 + i, 3, 1) = n1;
            AL.block(cur_row, 6 + j*LNum + i, 3, 1) = - DR*n2;
            BL.segment(cur_row, 3) = - SkewRbcd1*((DR - I)*tbc + DP);
            cur_row += 3;
            Vecn.push_back(n2);
        }
        VecN.push_back(Vecn);
    }


    // 这里用householderQr解不出来？？？
//    X = A.householderQr().solve(B);
    XL = AL.colPivHouseholderQr().solve(BL);
//    v = XL.head(3);
//    g = XL.segment(3, 3);



    // Ceres 优化所有项

    std::vector<double*> para_pts;
    for(int i = 0; i < PNum*FrmNum; i++)
    {
        double* lamda = new double[1];
        lamda[0] = XP[i + 6];
//        lamda[0] = 1;
        para_pts.push_back(lamda);
    }

    double* bg_ptr = new double[3];
    para_pts.push_back(bg_ptr);
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();

    double* v0_ptr = new double[3];
    para_pts.push_back(v0_ptr);
    Eigen::Map<Eigen::Vector3d> v0(v0_ptr);
    v0 = v;

//    double* Rwg_ptr = new double[9];
//    pointers.push_back(Rwg_ptr);
//    Eigen::Map<Eigen::Matrix3d> Rwg(Rwg_ptr);
//    Rwg.setIdentity();

//    Eigen::Vector3d dirG = g.normalized();
//    const Eigen::Vector3d gI = IMU::GRAVITY_VECTOR.normalized();
//    const Eigen::Vector3d vI = gI.cross(dirG);
//    const double cos_theta = gI.dot(dirG);
//    const double theta = std::acos(cos_theta);
//    Rwg = ExpSO3(vI*theta/vI.norm());

    double* g_ptr = new double[3];
    para_pts.push_back(g_ptr);
    Eigen::Map<Eigen::Vector3d> g0(g_ptr);
//    g0.setZero();
    g0 = g;

    double** parameters1 = new double*[5];
    parameters1[0] = bg_ptr;
//    parameters[1] = Rwg_ptr;
    parameters1[1] = g_ptr;
    parameters1[2] = v0_ptr;
    parameters1[3] = nullptr;
    parameters1[4] = nullptr;

    ceres::Problem problem;

    for(int i = 0; i < PNum; i++)
    {
        parameters1[3] = para_pts[i*FrmNum];//lamda0
        Eigen::Vector3d u1 = VecU[i][0];
        for(int j = 1; j < FrmNum; j++)
        {
            double delt = j*dt;
            Eigen::Vector3d u2 = VecU[i][j];
            parameters1[4] = para_pts[i*FrmNum + j];//lamda1
            ceres::CostFunction* cost_function = new CloseformCostFunction(u1, u2, pInts[j - 1], delt, gtba, Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters1, 5);
//            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), parameters, 5);
//            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(10.0), parameters, 5);
//            problem.SetParameterLowerBound(parameters[3], 0, 1e-5);
//            problem.SetParameterLowerBound(parameters[4], 0, 1e-5);
        }
    }

    // ceres线特征优化
    std::vector<double*> para_beta;
    for(int i = 0; i < LNum; i++)
    {
        double* beta = new double[1];
        beta[0] = Vecbeta[i];
//        beta[0] = 1;
        para_beta.push_back(beta);
    }

    double** parameters2 = new double*[2]; // 首先是线的beta残差
    parameters2[0] = bg_ptr;
    parameters2[1] = nullptr;
    /*
    for(int i = 0; i < LNum; i++)
    {
        parameters2[1] = para_beta[i];//beta
        Eigen::Vector3d p1 = Vecp1[i];
        Eigen::Vector3d q1 = Vecq1[i];
        for(int j = 0; j < FrmNum - 1; j++)
        {
            Eigen::Matrix3d M = VecM[i][j];
            ceres::CostFunction* cost_function = new CloseformCostFunctionLinesBeta(p1, q1, pInts[j], M);
            problem.AddResidualBlock(cost_function, nullptr, parameters2, 2);
//            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), parameters, 5);
//            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(10.0), parameters, 5);
//            problem.SetParameterLowerBound(parameters[3], 0, 1e-5);
//            problem.SetParameterLowerBound(parameters[4], 0, 1e-5);
        }
    }
    */

    std::vector<double*> para_lns;
    for(int i = 0; i < LNum*FrmNum; i++)
    {
        double* lamda = new double[1];
        lamda[0] = XL[i + 6];
//        lamda[0] = 1;
        para_lns.push_back(lamda);
    }
    double** parameters3 = new double*[6];// 然后是线的残差
    parameters3[0] = bg_ptr;
//    parameters3[1] = Rwg_ptr;
    parameters3[1] = g_ptr;
    parameters3[2] = v0_ptr;
    parameters3[3] = nullptr;
    parameters3[4] = nullptr;
    parameters3[5] = nullptr;
    for(int i = 0; i < LNum; i++)
    {
        parameters3[3] = para_beta[i];// beta
        parameters3[4] = para_lns[i];// lamda1
        Eigen::Vector3d n1 = VecN[i][0];
        Eigen::Vector3d p1 = Vecp1[i];
        Eigen::Vector3d q1 = Vecq1[i];
        for(int j = 1; j < FrmNum; j++)
        {
            double delt = j*dt;
            Eigen::Vector3d n2 = VecN[i][j];
            parameters3[5] = para_lns[j*LNum + i];//lamda2
            ceres::CostFunction* cost_function = new CloseformCostFunctionLines(n1, n2, p1, q1, pInts[j - 1], delt, gtba, Tbc);
            problem.AddResidualBlock(cost_function, nullptr, parameters3, 6);
//            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), parameters, 5);
//            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(10.0), parameters, 5);
//            problem.SetParameterLowerBound(parameters[3], 0, 1e-5);
//            problem.SetParameterLowerBound(parameters[4], 0, 1e-5);
        }
    }


    // gyro bias = 0的先验，感觉用处不大
//    ceres::CostFunction* prior_cost_function = new BiasPriorCostFunction(1);//prior先验权重10e5
//    problem.AddResidualBlock(prior_cost_function, nullptr, bg_ptr);

//    GravityParameterization* gravity_local_parameterization = new GravityParameterization;
//    problem.SetParameterization(Rwg_ptr, gravity_local_parameterization);

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    bool converged = (summary.termination_type == ceres::CONVERGENCE);

//    const Eigen::Vector3d g0 = Rwg*IMU::GRAVITY_VECTOR;

    if(converged)
    {
        std::cout << "Error:" << (gtbg - bg).norm()/gtbg.norm() << std::endl;
        std::cout << "Est:" << bg.transpose() << std::endl;
        std::cout << "GT :" << gtbg.transpose() << std::endl;
        std::cout << "Errorv:" << (v0 - gtv).norm() << std::endl;
        std::cout << "gnorm:" << g0.norm() << std::endl;
        std::cout << "DirectionError:" << 180.*std::acos(g0.normalized().dot(gtg.normalized()))/EIGEN_PI << std::endl;


        StaTime += timer.ElapsedSeconds();
        StaErrorv += (v0 - gtv).norm();
        gravity_error += 180.*std::acos(g0.normalized().dot(gtg.normalized()))/EIGEN_PI;

        ErrorPercen += (gtbg - bg).norm()/gtbg.norm();
        StaError += (gtbg - bg).norm();
        StaCount ++;

        int idi = ceil(((gtbg - bg).norm()/gtbg.norm())*100);
        if(idi > 99)
            idi = 99;
        statistical[idi] ++;

    }
    else
    {
        std::cout << "Can't Converged!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }



    delete[] parameters1;
    delete[] parameters2;
    delete[] parameters3;
    for(auto ptr : para_pts)
        delete[] ptr;
    for(auto ptr : para_beta)
        delete[] ptr;
    for(auto ptr : para_lns)
        delete[] ptr;

}



}
