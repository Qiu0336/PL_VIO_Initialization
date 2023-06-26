
#ifndef POINT_FEATURE_H_
#define POINT_FEATURE_H_
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

using namespace std;
using namespace Eigen;
using namespace io;

extern const int HEIGHT;
extern const int WIDTH;
extern const int MIN_DIST; // 两个feature之间的最小距离
extern const double F_THRESHOLD;
extern const int MAX_CNT;// 提取的最大特征点数
extern const int FOCAL_LENGTH;//VINS中是固定的


bool inBorder(const cv::Point2f &pt);
template<typename T>
void reduceVector(vector<T> &v, vector<uchar> status);

template<typename T>
void reduceVector(vector<T> &v, vector<uchar> status, vector<float> err);

class PointFeature
{
  public:
    PointFeature();
    void PrecessImage(const cv::Mat &_img);
    void MaskReject();
    void FundamentalReject();
    void UndistortedPoints();
    cv::Mat GetUndistortionImg();
    cv::Mat GetPointsImg();
    void GetTrackingPoints(Points &TrackingPoints);
    void GetOriginPoints(Points &OriginPoints);
    void ReadIntrinsicParameter(const string &calib_file);

    cv::Mat mask;
    cv::Mat pre_img, cur_img;
    vector<cv::Point2f> new_pts;//每一帧中新提取的特征点
    vector<cv::Point2f> pre_pts, cur_pts;//对应的图像特征点
    vector<cv::Point2f> cur_un_pts;//去畸变后归一化相机坐标系下的坐标
    vector<int> ids;//能够被跟踪到的特征点的id
    vector<int> track_cnt;  // 表示每个特征点被追踪到的次数
    camodocal::CameraPtr m_camera;

    static int pt_id;//用来作为特征点id，每检测到一个新的特征点，就将++n_id作为该特征点
    static int img_id;//图像id
};

#endif
