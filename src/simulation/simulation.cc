
#include <iostream>
#include <fstream>
#include <string>
//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>
#include <thread>
#include <pangolin/pangolin.h>
#include <unistd.h>

#include "util/csv.h"
#include "util/io.h"
#include "imu.h"
#include "utilities.h"

namespace Eigen {
typedef Eigen::Matrix<double, 6, 1> Vector6d;
} // namespace Eigen


bool IsinCamView(Param &params, Eigen::Vector3d &pc)
{
    if(pc(2) < 0) return false; // z必须大于０,在摄像机坐标系前方
    Eigen::Vector3d uv = params.K*(pc/pc[2]);
    if(uv[0] > 0 && uv[0] < params.image_h && uv[1] > 0 && uv[1] < params.image_w)
        return true;
    return false;
}

void CreatePoints(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &Points)
{
    // 5*5*2为八分之一个square墙壁
    const double x = 6;
    const double y = 6;
    const double z = 3;
    const int PNumPerWall = 600;//每一面墙的点数

    std::random_device rd;// 随机数生成设备
    std::default_random_engine e(rd());// 随机数生成器
    std::uniform_real_distribution<double> u(-1, 1);
    /*
    for(int i = 0; i < PNumPerWall; i++)
    {
        Eigen::Vector3d pt1(x, u(e)*y, u(e)*z);
        Eigen::Vector3d pt2( - x, u(e)*y, u(e)*z);
        Eigen::Vector3d pt3(u(e)*x, y, u(e)*z);
        Eigen::Vector3d pt4(u(e)*x, - y, u(e)*z);
        Points.push_back(pt1);
        Points.push_back(pt2);
        Points.push_back(pt3);
        Points.push_back(pt4);
    }
    */
    for(int i = 0; i < PNumPerWall; i++)
    {
        Eigen::Vector3d pt(x, u(e)*y, u(e)*z);
        Points.push_back(pt);
    }
    for(int i = 0; i < PNumPerWall; i++)
    {
        Eigen::Vector3d pt( - x, u(e)*y, u(e)*z);
        Points.push_back(pt);
    }
    for(int i = 0; i < PNumPerWall; i++)
    {
        Eigen::Vector3d pt(u(e)*x, y, u(e)*z);
        Points.push_back(pt);
    }
    for(int i = 0; i < PNumPerWall; i++)
    {
        Eigen::Vector3d pt(u(e)*x, - y, u(e)*z);
        Points.push_back(pt);
    }
}



void CreatePoints2(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &Points)
{
    const int PNumPerWall = 400;//每一面墙的点数

    std::random_device rd;// 随机数生成设备
    std::default_random_engine e(rd());// 随机数生成器
    std::uniform_real_distribution<double> rand_fai(0, EIGEN_PI/2);// xoy角度
    std::uniform_real_distribution<double> rand_theta(EIGEN_PI/6, 5*EIGEN_PI/6);// z角度
    std::uniform_real_distribution<double> rand_r(4, 10);// 深度

    for(int i = 0; i < PNumPerWall; i++)
    {
        double fai = rand_fai(e);
        double theta = rand_theta(e);
        double r = rand_r(e);
        Eigen::Vector3d pt(r*sin(theta)*cos(fai), r*sin(theta)*sin(fai), r*cos(theta));
        Points.push_back(pt);
    }
    for(int i = 0; i < PNumPerWall; i++)
    {
        double fai = rand_fai(e) + EIGEN_PI/2;
        double theta = rand_theta(e);
        double r = rand_r(e);
        Eigen::Vector3d pt(r*sin(theta)*cos(fai), r*sin(theta)*sin(fai), r*cos(theta));
        Points.push_back(pt);
    }
    for(int i = 0; i < PNumPerWall; i++)
    {
        double fai = rand_fai(e) + EIGEN_PI;
        double theta = rand_theta(e);
        double r = rand_r(e);
        Eigen::Vector3d pt(r*sin(theta)*cos(fai), r*sin(theta)*sin(fai), r*cos(theta));
        Points.push_back(pt);
    }
    for(int i = 0; i < PNumPerWall; i++)
    {
        double fai = rand_fai(e) + 3*EIGEN_PI/2;
        double theta = rand_theta(e);
        double r = rand_r(e);
        Eigen::Vector3d pt(r*sin(theta)*cos(fai), r*sin(theta)*sin(fai), r*cos(theta));
        Points.push_back(pt);
    }

}



void CreateLines(std::vector<Eigen::Vector6d, Eigen::aligned_allocator<Eigen::Vector6d>> &Lines)
{
    // 5*5*2为八分之一个square墙壁
    const double x = 8;
    const double y = 8;
    const double z = 4;
    const int LNumPerWall = 300;//每一面墙的线数
    const double Linemin = 1;//线长最小值
    const double Linemax = 3;//线长最大值

    std::random_device rd;// 随机数生成设备
    std::default_random_engine e(rd());// 随机数生成器
    std::uniform_real_distribution<double> u(-1, 1);
    std::uniform_real_distribution<double> u2(Linemin, Linemax);
    for(int i = 0; i < LNumPerWall; i++)
    {
        Eigen::Vector6d ln;
        ln.head(3) = Eigen::Vector3d(x, u(e)*y, u(e)*z);// 先随机生成一个点
        double angle = u(e)*EIGEN_PI;
        double length = u2(e);
        double p = ln[1] + sin(angle)*length;
        double q = ln[2] + cos(angle)*length;
        if(std::fabs(p) > y || std::fabs(q) > z)
        {
            i--;
            continue;
        }
        ln.tail(3) = Eigen::Vector3d(x, p, q);// 先随机生成一个点
        Lines.push_back(ln);
    }
    for(int i = 0; i < LNumPerWall; i++)
    {
        Eigen::Vector6d ln;
        ln.head(3) = Eigen::Vector3d(- x, u(e)*y, u(e)*z);// 先随机生成一个点
        double angle = u(e)*EIGEN_PI;
        double length = u2(e);
        double p = ln[1] + sin(angle)*length;
        double q = ln[2] + cos(angle)*length;
        if(std::fabs(p) > y || std::fabs(q) > z)
        {
            i--;
            continue;
        }
        ln.tail(3) = Eigen::Vector3d(- x, p, q);// 先随机生成一个点
        Lines.push_back(ln);
    }
    for(int i = 0; i < LNumPerWall; i++)
    {
        Eigen::Vector6d ln;
        ln.head(3) = Eigen::Vector3d(u(e)*x, y, u(e)*z);// 先随机生成一个点
        double angle = u(e)*EIGEN_PI;
        double length = u2(e);
        double p = ln[0] + sin(angle)*length;
        double q = ln[2] + cos(angle)*length;
        if(std::fabs(p) > x || std::fabs(q) > z)
        {
            i--;
            continue;
        }
        ln.tail(3) = Eigen::Vector3d(p, y, q);// 先随机生成一个点
        Lines.push_back(ln);
    }
    for(int i = 0; i < LNumPerWall; i++)
    {
        Eigen::Vector6d ln;
        ln.head(3) = Eigen::Vector3d(u(e)*x, - y, u(e)*z);// 先随机生成一个点
        double angle = u(e)*EIGEN_PI;
        double length = u2(e);
        double p = ln[0] + sin(angle)*length;
        double q = ln[2] + cos(angle)*length;
        if(std::fabs(p) > x || std::fabs(q) > z)
        {
            i--;
            continue;
        }
        ln.tail(3) = Eigen::Vector3d(p, - y, q);// 先随机生成一个点
        Lines.push_back(ln);
    }
}


void ViewerSimulation(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pts,
                      std::vector<IMUData> &imudata, std::vector<CamData> &camdata,
                      Eigen::Matrix3d &Kinv, int &t)
{
    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);// 启动深度测试
    glEnable(GL_BLEND);// 启动颜色混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);// 颜色混合的方式

    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 320, 0.2, 500),
                // 相机参数配置，高度，宽度，4个内参，最近/最远视距
                pangolin::ModelViewLookAt(2,0,2, 0,0,0, pangolin::AxisY)
                // 相机所在位置，相机所看点的位置，最后是相机轴方向
                );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
    // 显示视图在窗口中的范围（下上左右），最后一个参数为视窗长宽比

    const int wx = 6;
    const int wy = 6;
    const int wz = 3;
    while(!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// 清空颜色和深度缓存,刷新显示
        d_cam.Activate(s_cam);// 激活并设置状态矩阵
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);// 画背景

        glPointSize(3.0);//点的大小
        glColor3f(0.0 ,0.0, 0.0);//颜色
        glBegin(GL_LINE_LOOP);
        glVertex3f(wx, wy, wz);
        glVertex3f(wx, -wy, wz);
        glVertex3f(wx, -wy, -wz);
        glVertex3f(wx, wy, -wz);
        glEnd();

        glBegin(GL_LINE_LOOP);
        glVertex3f(-wx, wy, wz);
        glVertex3f(-wx, -wy, wz);
        glVertex3f(-wx, -wy, -wz);
        glVertex3f(-wx, wy, -wz);
        glEnd();

        glBegin(GL_LINE_LOOP);
        glVertex3f(wx, wy, wz);
        glVertex3f(-wx, wy, wz);
        glVertex3f(-wx, wy, -wz);
        glVertex3f(wx, wy, -wz);
        glEnd();

        glBegin(GL_LINE_LOOP);
        glVertex3f(wx, -wy, wz);
        glVertex3f(-wx, -wy, wz);
        glVertex3f(-wx, -wy, -wz);
        glVertex3f(wx, -wy, -wz);
        glEnd();

        glPointSize(3.0);//点的大小
        glColor3f(1.0 ,0.0, 0.0);//点的颜色
        glBegin(GL_POINTS);
        for(auto &pt : pts)
            glVertex3f(pt[0], pt[1], pt[2]);
        glEnd();

//        glColor3f(0.0 ,0.0, 1.0);//颜色
//        glBegin(GL_LINE_LOOP);
//        for(auto &im : imudata)
//            glVertex3f(im.twb[0], im.twb[1], im.twb[2]);
//        glEnd();

        glColor3f(0.0 ,1.0, 0.0);//颜色
        glBegin(GL_LINE_LOOP);
        for(auto &cam : camdata)
            glVertex3f(cam.twc[0], cam.twc[1], cam.twc[2]);
        glEnd();


        Eigen::Vector3d vertex0 = camdata[t].twc;
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vertices;
        vertices.push_back(Kinv*Eigen::Vector3d(0, 0, 1));
        vertices.push_back(Kinv*Eigen::Vector3d(0, 640, 1));
        vertices.push_back(Kinv*Eigen::Vector3d(640, 640, 1));
        vertices.push_back(Kinv*Eigen::Vector3d(640, 0, 1));
        for(auto &vertex : vertices)
            vertex = camdata[t].Rwc*vertex + camdata[t].twc;

        // 投影到墙壁点的计算，若不加此处代码，则为只画出归一化平面
        for(auto &vertex : vertices)
        {
            Eigen::Vector3d dir = (vertex - vertex0).normalized();
            double ax, ay;
            if(dir[0] > 0)
                ax = (wx - vertex0[0])/dir[0];
            else
                ax = (- wx - vertex0[0])/dir[0];
            if(dir[1] > 0)
                ay = (wy - vertex0[1])/dir[1];
            else
                ay = (- wy - vertex0[1])/dir[1];
            double a = std::min(ax, ay);
            vertex = vertex0 + a*dir;
        }


        // 画出相机投影
        glColor3f(0.0 ,1.0, 1.0);//颜色
        glBegin(GL_LINE_LOOP);
        for(auto &vertex : vertices)
            glVertex3f(vertex[0], vertex[1], vertex[2]);
        glEnd();

        glBegin(GL_LINES);
        for(auto &vertex : vertices)
        {
            glVertex3f(vertex0[0], vertex0[1], vertex0[2]);
            glVertex3f(vertex[0], vertex[1], vertex[2]);
        }
        glEnd();


        pangolin::FinishFrame();// 最终显示
    }
}


void ViewerSimulationLines(std::vector<Eigen::Vector6d, Eigen::aligned_allocator<Eigen::Vector6d>> &lns,
                      std::vector<IMUData> &imudata, std::vector<CamData> &camdata,
                      Eigen::Matrix3d &Kinv, int &t)
{
    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);// 启动深度测试
    glEnable(GL_BLEND);// 启动颜色混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);// 颜色混合的方式

    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 320, 0.2, 500),
                // 相机参数配置，高度，宽度，4个内参，最近/最远视距
                pangolin::ModelViewLookAt(2,0,2, 0,0,0, pangolin::AxisY)
                // 相机所在位置，相机所看点的位置，最后是相机轴方向
                );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
    // 显示视图在窗口中的范围（下上左右），最后一个参数为视窗长宽比

    const int wx = 8;
    const int wy = 8;
    const int wz = 4;
    while(!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// 清空颜色和深度缓存,刷新显示
        d_cam.Activate(s_cam);// 激活并设置状态矩阵
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);// 画背景

        glPointSize(3.0);//点的大小
        glColor3f(0.0 ,0.0, 0.0);//颜色
        glBegin(GL_LINE_LOOP);
        glVertex3f(wx, wy, wz);
        glVertex3f(wx, -wy, wz);
        glVertex3f(wx, -wy, -wz);
        glVertex3f(wx, wy, -wz);
        glEnd();

        glBegin(GL_LINE_LOOP);
        glVertex3f(-wx, wy, wz);
        glVertex3f(-wx, -wy, wz);
        glVertex3f(-wx, -wy, -wz);
        glVertex3f(-wx, wy, -wz);
        glEnd();

        glBegin(GL_LINE_LOOP);
        glVertex3f(wx, wy, wz);
        glVertex3f(-wx, wy, wz);
        glVertex3f(-wx, wy, -wz);
        glVertex3f(wx, wy, -wz);
        glEnd();

        glBegin(GL_LINE_LOOP);
        glVertex3f(wx, -wy, wz);
        glVertex3f(-wx, -wy, wz);
        glVertex3f(-wx, -wy, -wz);
        glVertex3f(wx, -wy, -wz);
        glEnd();

        glPointSize(3.0);//点的大小
        glColor3f(1.0 ,0.0, 0.0);//点的颜色
        glBegin(GL_LINES);
        for(auto &ln : lns)
        {
            glVertex3f(ln[0], ln[1], ln[2]);
            glVertex3f(ln[3], ln[4], ln[5]);
        }
        glEnd();

//        glColor3f(0.0 ,0.0, 1.0);//颜色
//        glBegin(GL_LINE_LOOP);
//        for(auto &im : imudata)
//            glVertex3f(im.twb[0], im.twb[1], im.twb[2]);
//        glEnd();

        glColor3f(0.0 ,1.0, 0.0);//颜色
        glBegin(GL_LINE_LOOP);
        for(auto &cam : camdata)
            glVertex3f(cam.twc[0], cam.twc[1], cam.twc[2]);
        glEnd();


        Eigen::Vector3d vertex0 = camdata[t].twc;
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vertices;
        vertices.push_back(Kinv*Eigen::Vector3d(0, 0, 1));
        vertices.push_back(Kinv*Eigen::Vector3d(0, 640, 1));
        vertices.push_back(Kinv*Eigen::Vector3d(640, 640, 1));
        vertices.push_back(Kinv*Eigen::Vector3d(640, 0, 1));
        for(auto &vertex : vertices)
            vertex = camdata[t].Rwc*vertex + camdata[t].twc;

        // 投影到墙壁点的计算，若不加此处代码，则为只画出归一化平面
        for(auto &vertex : vertices)
        {
            Eigen::Vector3d dir = (vertex - vertex0).normalized();
            double ax, ay;
            if(dir[0] > 0)
                ax = (wx - vertex0[0])/dir[0];
            else
                ax = (- wx - vertex0[0])/dir[0];
            if(dir[1] > 0)
                ay = (wy - vertex0[1])/dir[1];
            else
                ay = (- wy - vertex0[1])/dir[1];
            double a = std::min(ax, ay);
            vertex = vertex0 + a*dir;
        }


        // 画出相机投影
        glColor3f(0.0 ,1.0, 1.0);//颜色
        glBegin(GL_LINE_LOOP);
        for(auto &vertex : vertices)
            glVertex3f(vertex[0], vertex[1], vertex[2]);
        glEnd();

        glBegin(GL_LINES);
        for(auto &vertex : vertices)
        {
            glVertex3f(vertex0[0], vertex0[1], vertex0[2]);
            glVertex3f(vertex[0], vertex[1], vertex[2]);
        }
        glEnd();


        pangolin::FinishFrame();// 最终显示
    }
}


int main(int argc, char* argv[])
{

    std::string OutputImu = "/home/qjy/Research/VISLAM/SimulData/Imu/";
    std::string OutputGroundTruth = "/home/qjy/Research/VISLAM/SimulData/GroundTruth/";
    std::string OutputPoints = "/home/qjy/Research/VISLAM/SimulData/Points/";
    std::string OutputLines = "/home/qjy/Research/VISLAM/SimulData/Lines/";
    std::string OutputTimestamps = "/home/qjy/Research/VISLAM/SimulData/timestamps.txt";

    Param params;
    IMU imuGener(params);

    // IMU数据生成
    std::vector<IMUData> imudata;
    std::vector<IMUData> imudata_noise;
    for(float t = params.t_start; t < params.t_end; t += params.imu_timestep)
    {
        IMUData data = imuGener.MotionModel(t);
        imudata.push_back(data);// IMU真实数据无噪声

        // 添加噪声
        IMUData data_noise = data;// IMU数据添加bias和高斯白噪声
        imuGener.addIMUnoise(data_noise);
        imudata_noise.push_back(data_noise);

    }
    imuGener.init_velocity_ = imudata[0].imu_velocity;
    imuGener.init_twb_ = imudata.at(0).twb;
    imuGener.init_Rwb_ = imudata.at(0).Rwb;

    io::ImuData ImuData;
    io::State GroundTruth;
    for(auto &imu : imudata_noise)
    {
        uint64_t timestamp = imu.timestamp*1e9;
        ImuData.emplace_back(timestamp, imu.imu_gyro[0], imu.imu_gyro[1], imu.imu_gyro[2],
                imu.imu_acc[0], imu.imu_acc[1], imu.imu_acc[2]);
        Eigen::Isometry3d pose;
        pose.linear() = imu.Rwb;
        pose.translation() = imu.twb;
        GroundTruth.emplace_back(timestamp, pose,
                imu.imu_velocity[0], imu.imu_velocity[1], imu.imu_velocity[2],
                imu.imu_gyro_bias[0], imu.imu_gyro_bias[1], imu.imu_gyro_bias[2],
                imu.imu_acc_bias[0], imu.imu_acc_bias[1], imu.imu_acc_bias[2]);
    }
    io::Write_File<io::ImuData::value_type>(ImuData, OutputImu + "imudata.csv");
    io::Write_File<io::State::value_type>(GroundTruth, OutputGroundTruth + "groundtruth.csv");


    // 相机数据生成
    std::vector<CamData> camdata;

    for(float t = params.t_start; t < params.t_end; t += params.cam_timestep)
    {
        IMUData imu = imuGener.MotionModel(t);// 借用IMU运动生成模型生成相机的运动
        CamData cam;
        cam.timestamp = imu.timestamp;
        cam.Rwc = imu.Rwb*params.R_bc;
        cam.twc = imu.Rwb*params.t_bc + imu.twb;

        camdata.push_back(cam);
    }

    // 路标点生成
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> Points;
    CreatePoints(Points);

    std::vector<Eigen::Vector6d, Eigen::aligned_allocator<Eigen::Vector6d>> Lines;
    CreateLines(Lines);

    // 显示

//    int time = 0;
//    std::thread Viewer = std::thread(&ViewerSimulation, std::ref(Points), std::ref(imudata),
//                                     std::ref(camdata), std::ref(params.Kinv), std::ref(time));

//    std::thread Viewer = std::thread(&ViewerSimulationLines, std::ref(Lines), std::ref(imudata),
//                                     std::ref(camdata), std::ref(params.Kinv), std::ref(time));
//    while(1)
//    {
//        time = 0;
//        for(float t = params.t_start; t < params.t_end; t += params.cam_timestep)
//        {
//            usleep(params.cam_timestep*1e6);
//            time ++;
//        }
//    }
//    Viewer.join();




    // Tracking数据生成

    std::vector<io::timestamp_t> timestamps;
    std::vector<int> pt_pre_ids;
    std::vector<int> pt_pre_cnts;
    std::vector<int> pt_cur_ids;
    std::vector<int> pt_cur_cnts;
    std::vector<int> pt_pre_ind;
    std::vector<int> pt_cur_ind;// 记录当前点在Points中的索引，用来判断前后帧匹配点

    std::vector<int> ln_pre_ids;
    std::vector<int> ln_pre_cnts;
    std::vector<int> ln_cur_ids;
    std::vector<int> ln_cur_cnts;
    std::vector<int> ln_pre_ind;
    std::vector<int> ln_cur_ind;// 记录当前点在Points中的索引，用来判断前后帧匹配点


    int img_id = 0;
    int pt_id = 0;
    for(auto &cam : camdata)
    {
        io::timestamp_t timestamp = cam.timestamp*1e9;
        timestamps.push_back(timestamp);
        Eigen::Matrix3d Rcw = cam.Rwc.transpose();
        Eigen::Vector3d tcw = - Rcw*cam.twc;

        //以下是点的保存
        pt_cur_ind.clear();
        pt_cur_ids.clear();
        pt_cur_cnts.clear();
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> p2d;
        for(int i = 0; i < (int)Points.size(); i++)
        {
            Eigen::Vector3d pc = Rcw*Points[i] + tcw;
            if(IsinCamView(params, pc))
            {
                // 真值
                Eigen::Vector2d obs(pc[0]/pc[2], pc[1]/pc[2]);
                /*
                std::random_device rd;// 随机数生成设备
                std::default_random_engine generator_(rd());// 随机数生成器
                std::normal_distribution<double> noise(0.0, 0.5);// 高斯白噪声
//                // 投影到相机像素取整
                int u = round(uv[0]) + noise(generator_);
                int v = round(uv[1]) + noise(generator_);

//                int u = round(uv[0]);
//                int v = round(uv[1]);
                Eigen::Vector2d obs;
                obs[0] = (u - params.cx)/params.fx;
                obs[1] = (v - params.cy)/params.fy;
                */

                p2d.push_back(obs);
                pt_cur_ind.push_back(i);

                if(img_id == 0)
                {
                    pt_cur_ids.push_back(pt_id++);
                    pt_cur_cnts.push_back(1);
                }
                else
                {
                    auto it = std::find(pt_pre_ind.begin(), pt_pre_ind.end(), i);
                    if(it == pt_pre_ind.end())// 如果上一帧中没有这个点
                    {
                        pt_cur_ids.push_back(pt_id++);
                        pt_cur_cnts.push_back(1);
                    }
                    else// 如果上一帧中有这个点
                    {
                        int dist = std::distance(pt_pre_ind.begin(), it);
                        pt_cur_ids.push_back(pt_pre_ids[dist]);
                        pt_cur_cnts.push_back(pt_pre_cnts[dist] + 1);
                    }
                }

            }
        }
        io::Points TrackingPoints;
        for(int i = 0; i < (int)pt_cur_ids.size(); i++)
        {
            TrackingPoints.emplace_back(pt_cur_ids[i], pt_cur_cnts[i], p2d[i][0], p2d[i][1]);
        }
        std::sort(TrackingPoints.begin(), TrackingPoints.end(), [](io::point_t &a, io::point_t &b)
        {
            if(a.cnt == b.cnt)
                return a.id < b.id;
            return a.cnt > b.cnt;
        });
        io::Write_File<io::Points::value_type>(TrackingPoints, OutputPoints + std::to_string(timestamp) + ".csv");
        pt_pre_ids = pt_cur_ids;
        pt_pre_cnts = pt_cur_cnts;
        pt_pre_ind = pt_cur_ind;








        ln_cur_ind.clear();
        ln_cur_ids.clear();
        ln_cur_cnts.clear();
        std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> l2d;
        for(int i = 0; i < (int)Lines.size(); i++)
        {
            Eigen::Vector3d p1 = Rcw*Lines[i].head(3) + tcw;
            Eigen::Vector3d p2 = Rcw*Lines[i].tail(3) + tcw;
            if(IsinCamView(params, p1) && IsinCamView(params, p2))
            {
                // 真值
                Eigen::Vector4d obs;
                obs.head(2) = Eigen::Vector2d(p1[0]/p1[2], p1[1]/p1[2]);
                obs.tail(2) = Eigen::Vector2d(p2[0]/p2[2], p2[1]/p2[2]);
                /*
                std::random_device rd;// 随机数生成设备
                std::default_random_engine generator_(rd());// 随机数生成器
                std::normal_distribution<double> noise(0.0, 0.5);// 高斯白噪声
//                // 投影到相机像素取整
                int u = round(uv[0]) + noise(generator_);
                int v = round(uv[1]) + noise(generator_);

//                int u = round(uv[0]);
//                int v = round(uv[1]);
                Eigen::Vector2d obs;
                obs[0] = (u - params.cx)/params.fx;
                obs[1] = (v - params.cy)/params.fy;
                */

                l2d.push_back(obs);
                ln_cur_ind.push_back(i);

                if(img_id == 0)
                {
                    ln_cur_ids.push_back(pt_id++);
                    ln_cur_cnts.push_back(1);
                }
                else
                {
                    auto it = std::find(ln_pre_ind.begin(), ln_pre_ind.end(), i);
                    if(it == ln_pre_ind.end())// 如果上一帧中没有这个线
                    {
                        ln_cur_ids.push_back(pt_id++);
                        ln_cur_cnts.push_back(1);
                    }
                    else// 如果上一帧中有这个线
                    {
                        int dist = std::distance(ln_pre_ind.begin(), it);
                        ln_cur_ids.push_back(ln_pre_ids[dist]);
                        ln_cur_cnts.push_back(ln_pre_cnts[dist] + 1);
                    }
                }

            }
        }
        io::Lines TrackingLines;
        for(int i = 0; i < (int)ln_cur_ids.size(); i++)
        {
            TrackingLines.emplace_back(ln_cur_ids[i], ln_cur_cnts[i], l2d[i][0], l2d[i][1], l2d[i][2], l2d[i][3]);
        }
        std::sort(TrackingLines.begin(), TrackingLines.end(), [](io::line_t &a, io::line_t &b)
        {
            if(a.cnt == b.cnt)
                return a.id < b.id;
            return a.cnt > b.cnt;
        });
        io::Write_File<io::Lines::value_type>(TrackingLines, OutputLines + std::to_string(timestamp) + ".csv");
        ln_pre_ids = ln_cur_ids;
        ln_pre_cnts = ln_cur_cnts;
        ln_pre_ind = ln_cur_ind;




        img_id ++;
    }

    // 保存timestamps，用于Points文件的读取
    std::ofstream outfile;
    outfile.open(OutputTimestamps);
    for(auto &t : timestamps)
        outfile << t << std::endl;
    outfile.close();


    std::cout << "done";

    return 0;

}
