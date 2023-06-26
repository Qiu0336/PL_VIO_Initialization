
#ifndef LINE_FEATURE_H_
#define LINE_FEATURE_H_
#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "util/io.h"
#include "util/timer.h"
#include "camera/CameraFactory.h"
#include "camera/PinholeCamera.h"

#include <opencv2/line_descriptor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace Eigen;
using namespace io;
using namespace cv;

class LineFeature
{
  public:
    LineFeature();
    void PrecessImage(const cv::Mat &_img);
    void UndistortedEndPoints();
    Mat GetLinesImg();
    Mat GetLinesImg(Lines &TrackingLines);
    void GetTrackingLines(Lines &TrackingLines);
    void ReadIntrinsicParameter(const string &calib_file);

    Mat cur_img;

    vector<line_descriptor::KeyLine> pre_lines, cur_lines;//提取的线
    vector<int> pre_ids, cur_ids;//对应的id
    vector<int> pre_track_cnt, cur_track_cnt;//对应的已追踪次数

    Mat pre_des, cur_des;//描述子

    vector<pair<Point2d, Point2d>> cur_un_lines;//去畸变后归一化相机坐标系下的坐标

    camodocal::CameraPtr m_camera;

    static int line_id;//用来作为线id，每检测到一个新线，就将++line_id作为该线
    static int img_id;//图像id
};

#endif
