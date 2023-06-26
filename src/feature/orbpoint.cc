
#include "feature/orbpoint.h"

int ORBFeature::pt_id = 0;//用来作为特征点id，每检测到一个新的特征点，就将++n_id作为该特征点
int ORBFeature::img_id = 0;//图像id

int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}



ORBFeature::ORBFeature(ORBextractor* ORBExt): ORBExt(ORBExt)
{
}

void ORBFeature::PrecessImage(const Mat &_img, bool CheckOrientation, bool CheckFundamental)
{
    cv::Mat img;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(_img, img);// 图像均衡化
    cur_img = img;
    cur_pts.clear();
    Mat des;
    (*ORBExt)(cur_img, cv::Mat(), cur_pts, des);
    cur_des = des;

    if(cur_pts.empty())
    {
        pre_pts.clear();
        pre_ids.clear();
        pre_track_cnt.clear();
        img_id ++;
        cout << "empty!!!!!!!!!!!!!" << endl;
        return;
    }

    if(img_id == 0 || pre_pts.empty())
    {
        cur_ids.clear();
        cur_track_cnt.clear();
        for(int i = 0, ptsize = cur_pts.size(); i < ptsize; i++)
        {
            cur_ids.push_back(pt_id++);
            cur_track_cnt.push_back(1);
        }
    }
    else
    {
        int windowsize = ORB_WINDOWSIZE;
        vector<int> Match12(pre_pts.size(), -1);
        vector<int> Match21(cur_pts.size(), -1);
        vector<int> MatchedDist(cur_pts.size(), INT_MAX);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        for(int i = 0, ptsize = pre_pts.size(); i < ptsize; i++)
        {
            vector<int> Indices;
            Indices = GetFeaturesInArea(pre_pts[i].pt, windowsize);
            if(Indices.empty())
                continue;

            cv::Mat d1 = pre_des.row(i);
            int bestDist = INT_MAX;
            int bestDist2 = INT_MAX;
            int bestIdx = -1;
            for(auto &j : Indices)
            {
                cv::Mat d2 = cur_des.row(j);
                int dist = DescriptorDistance(d1, d2);

                if(MatchedDist[j] < dist)
                    continue;

                if(dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestIdx = j;
                }
                else if(dist < bestDist2)
                {
                    bestDist2 = dist;
                }

            }

//            if(bestDist < ORB_TH_LOW && bestDist < (float)bestDist2*ORB_mfNNratio)
            if(bestDist < ORB_TH_LOW)
            {
                if(Match21[bestIdx] >= 0)// 先前已经有匹配
                {
                    Match12[Match21[bestIdx]] = -1;
                }
                Match12[i] = bestIdx;
                Match21[bestIdx] = i;
                MatchedDist[bestIdx] = bestDist;

                if(CheckOrientation)
                {
                    float rot = pre_pts[i].angle - cur_pts[bestIdx].angle;
                    if(rot < 0.0)
                        rot += 360.0f;
                    int bin = round(rot*factor);
                    if(bin == HISTO_LENGTH)
                        bin = 0;
                    rotHist[bin].push_back(i);
                }
            }
        }

        if(CheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;
            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
            for(int i=0; i < HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    int idx1 = rotHist[i][j];
                    if(Match12[idx1] >= 0)
                    {
                        Match21[Match12[idx1]] = -1;
                        Match12[idx1] = -1;
                    }
                }
            }
        }

        if(CheckFundamental)
        {
            vector<Point2f> pre_gpt, cur_gpt;// 纯的匹配上的点
            vector<int> PtIdx;// 记录纯的匹配上的点id
            for(int i = 0, ptsize = pre_pts.size(); i < ptsize; i++)
            {
                if(Match12[i] >= 0)
                {
                    pre_gpt.push_back(pre_pts[i].pt);
                    cur_gpt.push_back(cur_pts[Match12[i]].pt);
                    PtIdx.push_back(i);
                }
            }
            // RANSAC剔除误匹配点
            if (PtIdx.size() >= 8)
            {
                vector<cv::Point2f> un_pre_pts(PtIdx.size()), un_cur_pts(PtIdx.size());
                for (unsigned int i = 0; i < PtIdx.size(); i++)// 把每个点反投影到归一化平面，去畸变再投影回来，得到去畸变的图像上的点
                {
                    Eigen::Vector2d tmp_p;
                    m_camera->liftReprojective(Eigen::Vector2d(pre_gpt[i].x, pre_gpt[i].y), tmp_p);
                    un_pre_pts[i] = cv::Point2f(tmp_p[0], tmp_p[1]);

                    m_camera->liftReprojective(Eigen::Vector2d(cur_gpt[i].x, cur_gpt[i].y), tmp_p);
                    un_cur_pts[i] = cv::Point2f(tmp_p[0], tmp_p[1]);
                }

                vector<uchar> status;// findFundamentalMat通过匹配好的特征点构建F矩阵，un_cur_pts, un_forw_pts都是匹配好一一对应的
                cv::findFundamentalMat(un_pre_pts, un_cur_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
                for(unsigned int i = 0; i < PtIdx.size(); i++)// 剔除不满足RANSAC的点
                {
                    if(status[i] == 0)
                    {
                        Match21[Match12[PtIdx[i]]] = -1;
                        Match12[PtIdx[i]] = -1;
                    }
                }
            }
        }


        cur_ids.clear();
        cur_track_cnt.clear();
        for(int i = 0, ptsize = cur_pts.size(); i < ptsize; i++)
        {
            if(Match21[i] >= 0)// 如果已匹配
            {
                cur_ids.push_back(pre_ids[Match21[i]]);
                cur_track_cnt.push_back(pre_track_cnt[Match21[i]] + 1);
            }
            else
            {
                cur_ids.push_back(pt_id++);
                cur_track_cnt.push_back(1);
            }
        }
    }

    UndistortedPoints();// 去畸变
    pre_pts = cur_pts;
    pre_des = cur_des;
    pre_ids = cur_ids;
    pre_track_cnt = cur_track_cnt;
    img_id ++;
}


vector<int> ORBFeature::GetFeaturesInArea(Point2f pt, int windowsize)
{
    vector<int> Indices;
    for(int i = 0, ptsize = cur_pts.size(); i < ptsize; i++)
    {
        Point2f cur_pt = cur_pts[i].pt;
        int distance = ceil(fabs(pt.x - cur_pt.x) + fabs(pt.y - cur_pt.y));
        if(distance <= windowsize)
            Indices.push_back(i);
    }
    return Indices;
}

void ORBFeature::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


void ORBFeature::UndistortedPoints()
{
    cur_un_pts.clear();
    for (auto &cur_pt : cur_pts)
    {
        Vector2d a(cur_pt.pt.x, cur_pt.pt.y);
        Vector3d b;
        m_camera->liftProjective(a, b);// 相机模型去畸变
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
}


cv::Mat ORBFeature::GetPointsImg()
{
    cv::Mat PointsImg;
    cv::cvtColor(cur_img, PointsImg, cv::COLOR_GRAY2RGB);
    cv::Scalar Color;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        if(cur_track_cnt[i] > 10)
            Color = cv::Scalar(0, 255, 0);// 绿
        else if(cur_track_cnt[i] > 2)// 蓝
        {
//            Color = cv::Scalar(0, 255, 0);// 绿
            Color = cv::Scalar(255, 0, 0);
        }
        else
        {
            continue;
            Color = cv::Scalar(0, 0, 255);// 红
        }
        cv::circle(PointsImg, cur_pts[i].pt, 2, Color, 2);
//        cv::putText(PointsImg, to_string(cur_ids[i]), cur_pts[i].pt, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
    }
    return PointsImg;
}

void ORBFeature::GetTrackingPoints(Points &TrackingPoints)
{
    for (size_t i = 0; i < cur_track_cnt.size(); i++)
    {
        TrackingPoints.emplace_back(cur_ids[i], cur_track_cnt[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    sort(TrackingPoints.begin(), TrackingPoints.end(), [](const point_t &a, const point_t &b)
         {
            if(a.cnt == b.cnt)
                return a.id < b.id;
            return a.cnt > b.cnt;// 由小到大排列
         });
}

void ORBFeature::ReadIntrinsicParameter(const string &calib_file)
{
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}
