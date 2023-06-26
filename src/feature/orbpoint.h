
#ifndef ORBPOINT_H_
#define ORBPOINT_H_
#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "util/io.h"
#include "camera/CameraFactory.h"
#include "camera/PinholeCamera.h"
#include "ORBextractor.h"

using namespace std;
using namespace Eigen;
using namespace io;
using namespace cv;

extern const int HEIGHT;
extern const int WIDTH;
extern const int FOCAL_LENGTH;//VINS中是固定的

extern const double F_THRESHOLD;
extern const int ORB_MAX_CNT;// 提取的最大特征点数
extern const int ORB_TH_LOW;// 匹配的distance小于该值的都舍掉
extern const double ORB_mfNNratio;// 最优匹配与次匹配的距离比值必须小于该值
extern const int ORB_WINDOWSIZE;// 前后帧匹配点的窗口大小
extern const int HISTO_LENGTH; // 直方图旋转一致性检测，360度平均分成30个部分；

int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

class ORBFeature
{
    public:
    ORBFeature(ORBextractor* ORBExt);
    void PrecessImage(const Mat &_img, bool CheckOrientation = true, bool CheckFundamental = true);
    vector<int> GetFeaturesInArea(Point2f pt, int windowsize);
    void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
    void UndistortedPoints();
    Mat GetPointsImg();
    void GetTrackingPoints(Points &TrackingPoints);
    void ReadIntrinsicParameter(const string &calib_file);

    ORBextractor* ORBExt;

    Mat cur_img;
    Mat pre_des, cur_des;//描述子
    vector<KeyPoint> pre_pts, cur_pts;

    vector<Point2f> cur_un_pts;//去畸变后归一化相机坐标系下的坐标
    vector<int> pre_ids, cur_ids;//对应的id
    vector<int> pre_track_cnt, cur_track_cnt;//对应的已追踪次数
    camodocal::CameraPtr m_camera;

    static int pt_id;//用来作为特征点id，每检测到一个新的特征点，就将++n_id作为该特征点
    static int img_id;//图像id
};

#endif
