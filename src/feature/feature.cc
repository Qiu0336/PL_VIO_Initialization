#include <iostream>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <sstream>

#include <unistd.h>

#include "util/csv.h"
#include "util/timer.h"
#include "point_feature.h"
#include "line_feature.h"
#include "orbpoint.h"
#include "ORBextractor.h"

using namespace std;

const int HEIGHT = 480;
const int WIDTH = 752;
const int FOCAL_LENGTH = 460;//VINS中是固定的
const double F_THRESHOLD = 1.0f;

// 光流追踪参数
//const int MIN_DIST = 50; // 两个feature之间的最小距离
//const int MAX_CNT = 60;// 提取的最大特征点数
const int MIN_DIST = 40; // 两个feature之间的最小距离
const int MAX_CNT = 100;// 提取的最大特征点数

// ORB匹配参数
const int ORB_MAX_CNT = 1000;// 提取的最大特征点数
const int ORB_TH_LOW = 50;// 匹配的distance小于该值的都舍掉
const double ORB_mfNNratio = 0.8f;// 最优匹配与次匹配的距离比值必须小于该值
const int ORB_WINDOWSIZE = 50;// 前后帧匹配点的窗口大小
const int HISTO_LENGTH = 30; // 直方图旋转一致性检测，360度平均分成30个部分；

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<uint64_t> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            uint64_t t;
            ss >> t;
            vTimeStamps.push_back(t);

        }
    }
}

int main(int argc, char **argv)
{

    string ConfigFile = "/home/qjy/Research/VISLAM/config/EuRoC.yaml";
    string ImagePath = "/home/qjy/Dataset/EuRoC/MH_03_medium/mav0/cam0/data";
    string TimestampPath = "/home/qjy/Dataset/EuRoC/EuRoC_TimeStamps/MH03.txt";

    string OutputPoints = "/home/qjy/Research/VISLAM/trackingpoints/MH03/";
    string OutputLines = "/home/qjy/Research/VISLAM/trackinglines/MH03/";
    string OutputORBPoints = "/home/qjy/Research/VISLAM/trackingORBpoints/MH03/";
    string OutputPicPt = "/home/qjy/Research/VISLAM/pic/";
    string OutputPic = "/home/qjy/Research/VISLAM/pic2/";

    string OutputOriginPoints = "/home/qjy/Research/VISLAM/OriginPts/";

    vector<string> ImageFilenames;
    vector<uint64_t> ImageTimestamps;

    LoadImages(ImagePath, TimestampPath, ImageFilenames, ImageTimestamps);


    int img_size = ImageFilenames.size();

    PointFeature PointF;
    PointF.ReadIntrinsicParameter(ConfigFile);

    LineFeature LineF;
    LineF.ReadIntrinsicParameter(ConfigFile);

    ORBextractor* ORBExt = new ORBextractor(ORB_MAX_CNT, 1.2, 8, 20, 7);
    ORBFeature ORBPF(ORBExt);
    ORBPF.ReadIntrinsicParameter(ConfigFile);

    cv::Mat im;
    for(int ni = 0; ni < img_size; ni ++)
    {
        Points TrackingPoints;
        Points OriginPoints;
        Lines TrackingLines;
        // Read image from file
        im = cv::imread(ImageFilenames[ni], cv::IMREAD_UNCHANGED);

        double timestamp = ImageTimestamps[ni]*1e-9;

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 <<  ImageFilenames[ni] << endl;
            return 1;
        }
        Timer timer;
        timer.Start();


        // 点处理

        PointF.PrecessImage(im);
        PointF.GetTrackingPoints(TrackingPoints);
//        PointF.GetOriginPoints(OriginPoints);
        if(TrackingPoints.size() > 0)
        {
            io::Write_File<io::Points::value_type>(TrackingPoints, OutputPoints + to_string(ImageTimestamps[ni]) + ".csv");
//            io::Write_File<io::Points::value_type>(OriginPoints, OutputOriginPoints + to_string(ImageTimestamps[ni]) + ".csv");
        }
//        if(ni % 2 == 0)
            imwrite(OutputPicPt + to_string(ni) + ".png", PointF.GetPointsImg());
        cv::imshow("img", PointF.GetPointsImg());
        cv::waitKey(1);



        // ORB点处理
/*
        ORBPF.PrecessImage(im, true, false);
        ORBPF.GetTrackingPoints(TrackingPoints);
        if(TrackingPoints.size() > 0)
        {
            io::Write_File<io::Points::value_type>(TrackingPoints, OutputORBPoints + to_string(ImageTimestamps[ni]) + ".csv");
//            io::Write_File<io::Points::value_type>(OriginPoints, OutputOriginPoints + to_string(ImageTimestamps[ni]) + ".csv");
        }
//        if(ni % 2 == 0)
            imwrite(OutputPic + to_string(ni) + ".png", ORBPF.GetPointsImg());

//        cv::imshow("img", ORBPF.GetPointsImg());
//        cv::waitKey(1);
*/



        // 线处理
//        if((ni % 2) == 0)
//        {
//            LineF.PrecessImage(im);
//            LineF.GetTrackingLines(TrackingLines);
//            if(TrackingLines.size() > 0)
//                io::Write_File<io::Lines::value_type>(TrackingLines, OutputLines + to_string(ImageTimestamps[ni]) + ".csv");
//            imwrite(OutputPic + to_string(ni) + ".png", LineF.GetLinesImg(TrackingLines));
//            cv::imshow("img", LineF.GetLinesImg(TrackingLines));
//            cv::waitKey(1);
//        }



        // Wait to load the next frame
        double t_process = timer.ElapsedSeconds();
        double T = 0;
        if(ni < img_size-1)
            T = ImageTimestamps[ni+1]*1e-9 - timestamp;
        if(t_process < T)
            usleep((T-t_process)*1e6);
    }
}
