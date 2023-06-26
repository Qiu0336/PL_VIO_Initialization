
#include "feature/point_feature.h"

int PointFeature::pt_id = 0;//用来作为特征点id，每检测到一个新的特征点，就将++n_id作为该特征点
int PointFeature::img_id = 0;//图像id

bool inBorder(const cv::Point2f &pt)  // 判断点是否在图像边界内
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < WIDTH - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < HEIGHT - BORDER_SIZE;
}

template<typename T>
void reduceVector(vector<T> &v, vector<uchar> status)  // 根据status，剔除vector中status为0的点
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

template<typename T>
void reduceVector(vector<T> &v, vector<uchar> status, vector<float> err)  // 根据status，剔除vector中status为0的点
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
    {
//        if (status[i] && err[i] < 20.0f)
        if (status[i])
        {
            v[j++] = v[i];
        }
    }
    v.resize(j);
}

PointFeature::PointFeature()
{
}

void PointFeature::PrecessImage(const cv::Mat &_img)
{
    cv::Mat img;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(_img, img);// 图像均衡化
    cur_img = img;

    if (img_id == 0)
    {
        cv::goodFeaturesToTrack(cur_img, new_pts, MAX_CNT, 0.05, MIN_DIST, mask);// 特征点提取
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 40, 0.01);
        cv::cornerSubPix(cur_img, new_pts, cv::Size(5, 5), cv::Size(-1, -1), criteria);
        for (auto &p : new_pts)// 将新提取的特征点添加到cur_pts中
        {
            cur_pts.push_back(p);
            ids.push_back(pt_id++);
            track_cnt.push_back(1);
        }
    }
    else
    {
        cur_pts.clear();
        if (pre_pts.size() > 0)// 光流追踪，pre_pts表示待追踪的点，追踪结果在cur_pts
        {
            vector<uchar> status;
            vector<float> err;
//            cv::calcOpticalFlowPyrLK(pre_img, cur_img, pre_pts, cur_pts, status, err);//光流计算


            cv::calcOpticalFlowPyrLK(pre_img, cur_img, pre_pts, cur_pts, status, err,
                                     cv::Size(21,21), 3, cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                                     0, 0.01);

//            for(unsigned int i = 0; i < status.size(); i++)
//            {
//                std::cout << (int)status[i] << ", " << err[i] << std::endl;
//            }

            for (int i = 0; i < int(cur_pts.size()); i++)
                if (status[i] && !inBorder(cur_pts[i])) // 排除边界外的点
                    status[i] = 0;
            reduceVector(cur_pts, status);
            reduceVector(pre_pts, status);// 大家一起减少，所以prev_pts-cur_pts-forw_pts匹配点都是按顺序一一对应的
            reduceVector(ids, status);
            reduceVector(cur_un_pts, status);
            reduceVector(track_cnt, status);
//            reduceVector(cur_pts, status, err);
//            reduceVector(pre_pts, status, err);// 大家一起减少，所以prev_pts-cur_pts-forw_pts匹配点都是按顺序一一对应的
//            reduceVector(ids, status, err);
//            reduceVector(cur_un_pts, status, err);
//            reduceVector(track_cnt, status, err);
        }

        FundamentalReject(); // 通过F矩阵去除outliers

        for (auto &n : track_cnt)// 剩下的点追踪次数+1
            n++;

        MaskReject();

        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());//n_max_cnt表示还需要提取的特征点数目
        if (n_max_cnt > 0)
        {
            new_pts.clear();
            cv::goodFeaturesToTrack(cur_img, new_pts, n_max_cnt, 0.05, MIN_DIST, mask);// 特征点提取
            cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 40, 0.01);
            cv::cornerSubPix(cur_img, new_pts, cv::Size(5, 5), cv::Size(-1, -1), criteria);
            for (auto &p : new_pts)// 将新提取的特征点添加到cur_pts中
            {
                cur_pts.push_back(p);
                ids.push_back(pt_id++);
                track_cnt.push_back(1);
            }
        }
    }


    UndistortedPoints();// 去畸变
    pre_img = cur_img;
    pre_pts = cur_pts;

//    for(auto &pts : cur_pts)
//    {
//        std::cout << pts << std::endl;
//    }

    img_id ++;
}

void PointFeature::MaskReject()// 将追踪成功的点按照成功次数由大到小排序
{
    mask = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(255));
    // prefer to keep features that are tracked for long time// 更倾向追踪更久的点
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void PointFeature::FundamentalReject()// 通过F矩阵去除outliers
{
    if (cur_pts.size() >= 8)
    {
        vector<cv::Point2f> un_pre_pts(pre_pts.size()), un_cur_pts(cur_pts.size());
//        for (unsigned int i = 0; i < pre_pts.size(); i++)// 把每个点反投影到归一化平面，去畸变再投影回来，得到去畸变的图像上的点
//        {
//            Eigen::Vector3d tmp_p;
//            m_camera->liftProjective(Eigen::Vector2d(pre_pts[i].x, pre_pts[i].y), tmp_p);
//            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + WIDTH / 2.0;
//            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + HEIGHT / 2.0;
//            un_pre_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

//            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
//            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + WIDTH / 2.0;
//            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + HEIGHT / 2.0;
//            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
//        }

        for (unsigned int i = 0; i < pre_pts.size(); i++)// 把每个点反投影到归一化平面，去畸变再投影回来，得到去畸变的图像上的点
        {
            Eigen::Vector2d tmp_p;
            m_camera->liftReprojective(Eigen::Vector2d(pre_pts[i].x, pre_pts[i].y), tmp_p);
            un_pre_pts[i] = cv::Point2f(tmp_p[0], tmp_p[1]);

            m_camera->liftReprojective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            un_cur_pts[i] = cv::Point2f(tmp_p[0], tmp_p[1]);
        }

        vector<uchar> status;// findFundamentalMat通过匹配好的特征点构建F矩阵，un_cur_pts, un_forw_pts都是匹配好一一对应的
        cv::findFundamentalMat(un_pre_pts, un_cur_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);

        reduceVector(cur_pts, status);
        reduceVector(pre_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
    }
}

void PointFeature::UndistortedPoints()
{
    cur_un_pts.clear();
    for (auto &cur_pt : cur_pts)
    {
        Vector2d a(cur_pt.x, cur_pt.y);
        Vector3d b;
        m_camera->liftProjective(a, b);// 相机模型去畸变
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
}

cv::Mat PointFeature::GetUndistortionImg()
{
    cv::Mat undistortedImg(HEIGHT + 600, WIDTH + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < WIDTH; i++)
        for (int j = 0; j < HEIGHT; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + WIDTH / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + HEIGHT / 2;
        pp.at<float>(2, 0) = 1.0;
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < HEIGHT + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < WIDTH + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
    }
    return undistortedImg;
}

cv::Mat PointFeature::GetPointsImg()
{
    cv::Mat PointsImg;
    cv::cvtColor(cur_img, PointsImg, cv::COLOR_GRAY2RGB);
    cv::Scalar Color;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        if(track_cnt[i] > 10)
            Color = cv::Scalar(0, 255, 0);// 绿
        else if(track_cnt[i] > 5)
            Color = cv::Scalar(255, 0, 0);// 蓝
        else
            Color = cv::Scalar(0, 0, 255);// 红
        cv::circle(PointsImg, cur_pts[i], 2, Color, 2);
        cv::putText(PointsImg, to_string(ids[i]), cur_pts[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
    }
    return PointsImg;
}

void PointFeature::GetTrackingPoints(Points &TrackingPoints)
{
    for (size_t i = 0; i < track_cnt.size(); i++)
    {
        TrackingPoints.emplace_back(ids[i], track_cnt[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    sort(TrackingPoints.begin(), TrackingPoints.end(), [](const point_t &a, const point_t &b)
         {
            return a.id < b.id;// 由小到大排列
         });
}

void PointFeature::GetOriginPoints(Points &OriginPoints)
{
    for (size_t i = 0; i < track_cnt.size(); i++)
    {
        OriginPoints.emplace_back(ids[i], track_cnt[i], cur_pts[i].x, cur_pts[i].y);
    }
    sort(OriginPoints.begin(), OriginPoints.end(), [](const point_t &a, const point_t &b)
         {
            return a.id < b.id;// 由小到大排列
         });
}

void PointFeature::ReadIntrinsicParameter(const string &calib_file)
{
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}
