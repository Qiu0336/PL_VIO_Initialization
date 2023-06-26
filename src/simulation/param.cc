
#include "param.h"

// 相机-IMU外参以及初始bias在这里修改
Param::Param()
{
    K << fx,  0, cx,
          0, fy, cy,
          0,  0,  1;

    Kinv << 1.0/fx, 0, -cx/fx,
            0, 1.0/fy, -cy/fy,
            0,  0,  1;

    Eigen::Matrix3d R;   // 把body坐标系朝向旋转一下,得到相机坐标系，好让它看到landmark,  相机坐标系的轴在body坐标系中的表示
    // 相机朝着轨迹里面看， 特征点在轨迹外部， 这里我们采用这个
    R << 0, 0, -1,
        -1, 0, 0,
         0, 1, 0;
    R_bc = R;
    t_bc = Eigen::Vector3d(0.05,0.04,0.03);

    Init_bg = Eigen::Vector3d(- 0.0023, 0.0249, 0.0817);
    Init_ba = Eigen::Vector3d(- 0.0236, 0.1210, 0.0748);

}
