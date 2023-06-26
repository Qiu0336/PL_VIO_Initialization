
#include <random>
#include "imu.h"
#include "utilities.h"

// 欧拉角表示的旋转转化为旋转矩阵：见第二课课件
Eigen::Matrix3d euler2Rotation( Eigen::Vector3d  eulerAngles)
{
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);
    double yaw = eulerAngles(2);

    double cr = cos(roll);  double sr = sin(roll);
    double cp = cos(pitch); double sp = sin(pitch);
    double cy = cos(yaw);   double sy = sin(yaw);

    Eigen::Matrix3d RIb;
    RIb << cy*cp, cy*sp*sr - sy*cr, sy*sr + cy* cr*sp,
            sy*cp, cy *cr + sy*sr*sp, sp*sy*cr - cy*sr,
            - sp, cp*sr, cp*cr;
    return RIb;
}

// w系下欧拉角表示的角速度转化为b系下角速度变换矩阵：见第二课课件
Eigen::Matrix3d eulerRates2bodyRates(Eigen::Vector3d eulerAngles)
{
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);

    double cr = cos(roll);  double sr = sin(roll);
    double cp = cos(pitch); double sp = sin(pitch);

    Eigen::Matrix3d R;
    R << 1, 0, - sp,
         0, cr, sr*cp,
         0, - sr, cr*cp;
    return R;
}


IMU::IMU(Param p): param_(p)
{
    gyro_bias_ = p.Init_bg;
    acc_bias_ = p.Init_ba;
}

void IMU::addIMUnoise(IMUData& data)// IMU数据添加噪声
{
    std::random_device rd;// 随机数生成设备
    std::default_random_engine generator_(rd());// 随机数生成器
    std::normal_distribution<double> noise(0.0, 1.0);// 高斯白噪声

    Eigen::Vector3d noise_gyro(noise(generator_), noise(generator_), noise(generator_));
    Eigen::Matrix3d gyro_sqrt_cov = param_.gyro_noise_sigma * Eigen::Matrix3d::Identity();
    data.imu_gyro = data.imu_gyro + gyro_sqrt_cov * noise_gyro / sqrt(param_.imu_timestep) + gyro_bias_;
    // ↑↑↑ gyro高斯白噪声的离散化：nd[k]=δ'ω[k]，其中δ'=δ/sqrt(△t),δ为连续高斯的sigma，ω[k]～N(0,1)

    Eigen::Vector3d noise_acc(noise(generator_), noise(generator_), noise(generator_));
    Eigen::Matrix3d acc_sqrt_cov = param_.acc_noise_sigma * Eigen::Matrix3d::Identity();
    data.imu_acc = data.imu_acc + acc_sqrt_cov * noise_acc / sqrt(param_.imu_timestep) + acc_bias_;
//    data.imu_acc = data.imu_acc + acc_sqrt_cov * noise_acc / sqrt(param_.imu_timestep);
    // ↑↑↑  acc高斯白噪声的离散化：nd[k]=δ'ω[k]，其中δ'=δ/sqrt(△t),δ为连续高斯的sigma，ω[k]～N(0,1)

    // gyro_bias update
    Eigen::Vector3d noise_gyro_bias(noise(generator_), noise(generator_), noise(generator_));
    gyro_bias_ += param_.gyro_bias_sigma * sqrt(param_.imu_timestep ) * noise_gyro_bias;
    data.imu_gyro_bias = gyro_bias_;
    // ↑↑↑ gyro bias的离散化：bd[k]=bd[k-1]+δ'ω[k]，其中δ'=δ*sqrt(△t),δ为连续bias的sigma，ω[k]～N(0,1)

    // acc_bias update
    Eigen::Vector3d noise_acc_bias(noise(generator_), noise(generator_), noise(generator_));
    acc_bias_ += param_.acc_bias_sigma * sqrt(param_.imu_timestep ) * noise_acc_bias;
    data.imu_acc_bias = acc_bias_;
    // ↑↑↑  acc bias的离散化：bd[k]=bd[k-1]+δ'ω[k]，其中δ'=δ*sqrt(△t),δ为连续bias的sigma，ω[k]～N(0,1)
}

// 得到IMU运动模型，这里的轨迹自己设
IMUData IMU::MotionModel(double t)
{
    IMUData data;
    // param
    float x = 3;
    float y = 3;
    float z = 1;           //控制z振幅
    float K1 = 10;          // z轴的正弦频率是x，y的k1倍
    float K = M_PI/10;    // 20 * K = 2pi ，即K=2pi/一周总时间,由于我们采取的是时间是20s, 系数K控制yaw正好旋转一圈，运动一周

    // translation平移模拟加速度
    // twb:  body frame in world frame
    // position决定了点运动的轨迹
    Eigen::Vector3d position(x*cos(K*t), y*sin(K*t), z*sin(K1*K*t));
    // 位置的导数dp为速度
    Eigen::Vector3d dp( - K*x*sin(K*t), K*y*cos(K*t), z*K1*K*cos(K1*K*t));
    // 速度的导数ddp为加速度
    Eigen::Vector3d ddp( - K*K*x*cos(K*t), - K*K*y*sin(K*t), - z*K1*K1*K*K*sin(K1*K*t));
    // Rotation旋转模拟角速度，yaw轴绕着中轴线转1圈0——2π，roll和pich自己设大小
    double k_roll = 0.0;
    double k_pitch = 0.0;
//    double k_roll = 0.1;
//    double k_pitch = 0.2;
    Eigen::Vector3d eulerAngles(k_roll*cos(t) , k_pitch*sin(t) , K*t + M_PI);// roll ~ [-0.2, 0.2], pitch ~ [-0.3, 0.3], yaw ~ [0,2pi]
    Eigen::Vector3d eulerAnglesRates(-k_roll*sin(t), k_pitch*cos(t), K);// euler angles 的导数

    // 欧拉角到旋转矩阵的转化，得到Rwb
    Eigen::Matrix3d Rwb = euler2Rotation(eulerAngles);
    // W系角速度转变为body系
    Eigen::Vector3d imu_gyro = eulerRates2bodyRates(eulerAngles)*eulerAnglesRates;

    Eigen::Vector3d gn(0,0,-9.81);
    Eigen::Vector3d imu_acc = Rwb.transpose()*(ddp - gn);//  Rbw * (a-g)

    data.imu_gyro = imu_gyro;// body系下角速度
    data.imu_acc = imu_acc;// body系下加速度
    data.Rwb = Rwb;
    data.twb = position;
    data.imu_velocity = dp;// W系下线速度
    data.timestamp = t;
    return data;
}

//读取生成的imu数据并用imu动力学模型对数据进行计算，最后保存imu积分以后的轨迹，用来验证数据以及模型的有效性。
//即利用初始位姿和速度以及IMU数据，积分计算出整个轨迹（这里不考虑noise和bias），分欧拉和中值两种积分法
//src源数据文件，dist积分后得到的位姿文件
void IMU::testImu(std::string src, std::string dist)
{
    std::vector<IMUData>imudata;
    LoadPose(src,imudata);

    std::ofstream save_points;
    save_points.open(dist);

    double dt = param_.imu_timestep;
    Eigen::Vector3d Pwb = init_twb_;              // position :    from  imu measurements
    Eigen::Quaterniond Qwb(init_Rwb_);            // quaterniond:  from imu measurements
    Eigen::Vector3d Vw = init_velocity_;          // velocity  :   from imu measurements
    Eigen::Vector3d gw(0,0,-9.81);    // ENU frame
    Eigen::Vector3d temp_a;
    Eigen::Vector3d theta;
    for (int i = 1; i < imudata.size(); ++i) {

//        /// imu 动力学模型 欧拉积分
//        ///

//        IMUData imupose = imudata[i];

//        Eigen::Quaterniond dq; // 四元数表示更新的增量
//        Eigen::Vector3d dtheta_half =  imupose.imu_gyro * dt /2.0;// 0.5ω△t
//        dq.w() = 1;
//        dq.x() = dtheta_half.x();
//        dq.y() = dtheta_half.y();
//        dq.z() = dtheta_half.z();
//        dq.normalize();
        
//        Eigen::Vector3d acc_w = Qwb * (imupose.imu_acc) + gw;  // aw = Rwb * ( acc_body - acc_bias ) + gw
//        Qwb = Qwb * dq;
//        Pwb = Pwb + Vw * dt + 0.5 * dt * dt * acc_w;
//        Vw = Vw + acc_w * dt;
        
        /// imu 动力学模型  中值积分
        ///

        IMUData imupose = imudata[i];
        IMUData imupose_last = imudata[i-1];

        Eigen::Quaterniond dq; // 四元数表示更新的增量
        Eigen::Vector3d dtheta_half = (imupose.imu_gyro+imupose_last.imu_gyro)*dt/4.0;
        // 0.5*0.5(ω1+ω2)△t
        dq.w() = 1;
        dq.x() = dtheta_half.x();
        dq.y() = dtheta_half.y();
        dq.z() = dtheta_half.z();
        dq.normalize();

        Eigen::Vector3d acc_w = (Qwb*imupose.imu_acc + Qwb*imupose_last.imu_acc)/2.0 + gw;
        // aw = (Rwb * acc_body1 + Rwb * acc_body2) / 2.0 + gw
        Qwb = Qwb * dq;
        Pwb = Pwb + Vw * dt + 0.5 * dt * dt * acc_w;
        Vw = Vw + acc_w * dt;


        //　按着imu postion, imu quaternion , cam postion, cam quaternion 的格式存储，由于没有cam，所以imu存了两次
        save_points<<imupose.timestamp<<" "
                   <<Qwb.w()<<" "
                   <<Qwb.x()<<" "
                   <<Qwb.y()<<" "
                   <<Qwb.z()<<" "
                   <<Pwb(0)<<" "
                   <<Pwb(1)<<" "
                   <<Pwb(2)<<" "
                   <<Qwb.w()<<" "
                   <<Qwb.x()<<" "
                   <<Qwb.y()<<" "
                   <<Qwb.z()<<" "
                   <<Pwb(0)<<" "
                   <<Pwb(1)<<" "
                   <<Pwb(2)<<" "
                   <<std::endl;

    }

    std::cout<<"test　end"<<std::endl;

}
