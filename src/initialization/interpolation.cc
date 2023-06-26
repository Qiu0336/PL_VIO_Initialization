
#include "interpolation.h"

BSpline_Cubic::BSpline_Cubic(int frames):frames(frames) // 只需要给待插值的帧数即可
{
    A.resize(frames + 2, frames + 2);
    A.setZero();

    A(1, 0) = 6;
    A(frames, frames + 1) = 6;
    for(int i = 2; i < frames; i++)
    {
        A(i, i-1) = 1;
        A(i, i)   = 4;
        A(i, i+1) = 1;
    }

    dt = 1.0f/(frames - 1);


    // 非扭结边界条件：第一个点和第二个点的三阶导相等，倒数第一和倒数第二个点的三阶导相等
    const double inv_dt3 = 1.0f/(dt*dt*dt);
    A(0, 0) = -6*inv_dt3;//a3
    A(0, 1) = 12*inv_dt3;//b3 - a4
    A(0, 2) = -9*inv_dt3;//c3 - b4
    A(0, 3) =  4*inv_dt3;//d3 - c4
    A(0, 4) =   -inv_dt3;//-d4

    A(frames + 1, frames - 3) =   -inv_dt3;//al+1
    A(frames + 1, frames - 2) =  4*inv_dt3;//bl+1 - al+2
    A(frames + 1, frames - 1) = -9*inv_dt3;//cl+1 - bl+2
    A(frames + 1, frames)     = 12*inv_dt3;//dl+1 - cl+2
    A(frames + 1, frames + 1) = -6*inv_dt3;//-dl+2


    // 自然边界条件：第一个点和最后一个点的二阶导都为0
//        const double inv_dt2 = 1.0f/(dt*dt);
//        A(0, 0) =  6*inv_dt2;
//        A(0, 1) = -9*inv_dt2;
//        A(0, 2) =  3*inv_dt2;

//        A(frames + 1, frames - 1) =  3*inv_dt2;
//        A(frames + 1, frames)     = -9*inv_dt2;
//        A(frames + 1, frames + 1) =  6*inv_dt2;

    Knots.resize(frames + 6);// 节点表计算
    Knots.setZero();
    for(int i = 4; i < frames + 6; i++)
    {
        if(i < frames + 2)
            Knots(i) = (i - 3)*dt;
        else
            Knots(i) = 1;
    }

    // 矩阵Z和控制点在传入每一帧位置后计算
    ControlPoints.resize(frames + 2, 3);
    Z.resize(frames + 2,3);
    Z.setZero();

}

bool BSpline_Cubic::Set_ControlPoints(std::vector<Eigen::Vector3d> Points)// 设置新的插值点，同时更新对应的控制点
{
    if(int(Points.size()) != frames)// 检查传入的点是否等于设置的frames数
    {
        std::cout << "B_Spline Set_ControlPoints numbers Error!!!";
        return false;
    }

    for(int i = 0; i < frames; i++)
        Z.row(i+1) = Points[i];

    ControlPoints.setZero();
    ControlPoints = 6*A.inverse()*Z;
    return true;
}

// pj应该 ∈ [5, nframes - 1]
// 所以取t的时候应该保证  t ∈ [2*dt, 1 - 3*dt],dt为节点表间隔
Eigen::Vector3d BSpline_Cubic::Get_Interpolation(double t)// 获取插值，传入变量t即可，t∈ [0, 1)
{
    int pj;
    for(int i = 3; i < frames + 2; i++)//Knots(3) = 0;  Knots(nframes + 2) = 1
    {
        if(Knots(i) <= t)
            pj = i;   //pj ∈ [3, nframes + 1]
        else
            break;
    }

    double invbj = 1.0f / (6*dt*dt*dt);
    double delt = t - Knots(pj);

    double Bjt = delt*delt*delt*invbj;
    double Bj_1t = ( - 3*delt*delt*delt + 3*dt*delt*delt + 3*dt*dt*delt + dt*dt*dt)*invbj;
    double Bj_2t = (3*delt*delt*delt - 6*dt*delt*delt + 4*dt*dt*dt)*invbj;
    double Bj_3t = ( - delt*delt*delt + 3*dt*delt*delt - 3*dt*dt*delt + dt*dt*dt)*invbj;

    Eigen::Vector3d res;
    res = Bj_3t*ControlPoints.row(pj-3) + Bj_2t*ControlPoints.row(pj-2)
            + Bj_1t*ControlPoints.row(pj-1) + Bjt*ControlPoints.row(pj);
    return res;
}

Eigen::Vector3d BSpline_Cubic::Get_Interpolation_FirstOrder(double t)// 获取插值一阶导数，传入变量t即可，t∈ [0, 1)
{
    int pj;
    for(int i = 3; i < frames + 2; i++)//Knots(3) = 0;  Knots(nframes + 2) = 1
    {
        if(Knots(i) <= t)
            pj = i;   //paraJ ∈ [3, nframes + 1]
        else
            break;
    }

    double invbj = 1.0f / (6*dt*dt*dt);
    double delt = t - Knots(pj);

    double Bjt = 3*delt*delt*invbj;
    double Bj_1t = ( - 9*delt*delt + 6*dt*delt + 3*dt*dt)*invbj;
    double Bj_2t = (9*delt*delt - 12*dt*delt)*invbj;
    double Bj_3t = ( - 3*delt*delt + 6*dt*delt - 3*dt*dt)*invbj;

    Eigen::Vector3d res;
    res = Bj_3t*ControlPoints.row(pj-3) + Bj_2t*ControlPoints.row(pj-2)
            + Bj_1t*ControlPoints.row(pj-1) + Bjt*ControlPoints.row(pj);
    return res;
}

Eigen::Vector3d BSpline_Cubic::Get_Interpolation_SecondOrder(double t)// 获取插值二阶导数，传入变量t即可，t∈ [0, 1)
{
    int pj;
    for(int i = 3; i < frames + 2; i++)//Knots(3) = 0;  Knots(nframes + 2) = 1
    {
        if(Knots(i) <= t)
            pj = i;   //paraJ ∈ [3, nframes + 1]
        else
            break;
    }

    double invbj = 1.0f / (6*dt*dt*dt);
    double delt = t - Knots(pj);

    double Bjt = 6*delt*invbj;
    double Bj_1t = ( - 18*delt + 6*dt)*invbj;
    double Bj_2t = (18*delt - 12*dt)*invbj;
    double Bj_3t = ( - 6*delt + 6*dt)*invbj;

    Eigen::Vector3d res;
    res = Bj_3t*ControlPoints.row(pj-3) + Bj_2t*ControlPoints.row(pj-2)
            + Bj_1t*ControlPoints.row(pj-1) + Bjt*ControlPoints.row(pj);
    return res;
}

// 旋转插值获取，输入参数：R0，R10以及R0到R10之间imu测量的w：w0——w10; 返回插值后的R0——R10
std::vector<Eigen::Matrix3d> Rot_Interpolation(Eigen::Matrix3d R0, Eigen::Matrix3d R10,
                                               std::vector<Eigen::Vector3d> w, const double dt)
{
    std::vector<Eigen::Vector3d> delta_wt;
    Eigen::Vector3d sum_wt;
    sum_wt.setZero();
    Eigen::Vector3d div2_w01 = 0.5*(w[0] + w[1]);
    int internums = w.size() - 1;
    for(int i = 1; i < internums; i++)
    {
        Eigen::Vector3d dwt;
        dwt = (0.5*(w[i] + w[i + 1]) - div2_w01)*dt;
        delta_wt.push_back(dwt);
        sum_wt += dwt;
    }

    Eigen::Vector3d fai10 = LogSO3(R0.transpose()*R10);

    std::vector<Eigen::Vector3d> fai;// fai0 - fai9
    std::vector<Eigen::Matrix3d> Ri;// R0 - R9

    fai.push_back(Eigen::Vector3d::Zero());// fai0
    Ri.push_back(R0);// R0
    double div = 1.0f/internums;
    fai.push_back(div*(fai10 - sum_wt));// fai1
    Ri.push_back(R0*ExpSO3(fai[1]));// R1
    for(int j = 1; j < internums - 1; j++)
    {
        Eigen::Vector3d faii = fai[j] + fai[1] + delta_wt[j - 1];
        fai.push_back(faii);
        Ri.push_back(R0*ExpSO3(fai[j + 1]));
    }
    Ri.push_back(R10);
    return Ri;



//    Eigen::Vector3d sigma_w1;
//    Eigen::Vector3d sigma_w2;
//    sigma_w1.setZero();
//    sigma_w2.setZero();
//    std::vector<Eigen::Vector3d> mid_w;// 中值积分
//    for(int i = 0; i < 10; i++)
//    {
//        mid_w.push_back(0.5*(w[i] + w[i + 1])*dt);
//        sigma_w2 += mid_w[i];
//    }

//    std::vector<Eigen::Matrix3d> Ri;// R0 - R9
//    Ri.push_back(R0);

//    Eigen::Vector3d fai10 = LogSO3(R0.transpose()*R10);
//    Eigen::Matrix3d skew10_div2 = 0.5*Skew(fai10);

//    Eigen::Matrix3d A;
//    Eigen::Vector3d b;
//    for(int i = 1; i < 10; i++)
//    {
//        sigma_w2 -= mid_w[i - 1];
//        sigma_w1 += mid_w[i - 1];

//        b = i*(sigma_w1 + sigma_w2 - fai10) - 10*sigma_w1;
//        A = i*skew10_div2 - 10*Eigen::Matrix3d::Identity();

//        Ri.push_back(R0*ExpSO3(A.inverse()*b));
//    }
//    Ri.push_back(R10);
//    return Ri;

}



