
#ifndef IMUSIM_PARAM_H
#define IMUSIM_PARAM_H

#include <eigen3/Eigen/Core>

class Param{

public:

    Param();

    // time
    int imu_frequency = 200;
    int cam_frequency = 20;
    double imu_timestep = 1./imu_frequency;
    double cam_timestep = 1./cam_frequency;
    double t_start = 0.;
    double t_end = 20;  //  20 s

    // bias随机游走噪声
    double gyro_bias_sigma = 2.0e-05;
    double acc_bias_sigma = 3.0e-3;

    // 测量值的高斯白噪声
    double gyro_noise_sigma = 1.7e-04;    // rad/s * 1/sqrt(hz)  设定gyro高斯白噪声的δ
    double acc_noise_sigma = 2.0e-3;      //　m/(s^2) * 1/sqrt(hz)设定acc高斯白噪声的δ

    double pixel_noise = 1;              // 1 pixel noise

    // 相机内参
    double fx = 460;
    double fy = 460;
    double cx = 255;
    double cy = 255;
    double image_w = 640;
    double image_h = 640;

    Eigen::Matrix3d K;
    Eigen::Matrix3d Kinv;

    Eigen::Matrix3d R_bc;   // cam to body
    Eigen::Vector3d t_bc;     // cam to body

    Eigen::Vector3d Init_bg;
    Eigen::Vector3d Init_ba;

};


#endif //IMUSIM_PARAM_H
