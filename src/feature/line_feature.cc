
#include "feature/line_feature.h"

int LineFeature::line_id = 0;
int LineFeature::img_id = 0;

void LineDetect(const Mat& _image, CV_OUT vector<line_descriptor::KeyLine>& keylines, const Mat& mask = Mat())
{
    if(mask.data != NULL && (mask.size() != _image.size() || mask.type() != CV_8UC1))
        CV_Error(Error::StsBadArg, "Mask error while detecting lines: please check its dimensions and that data type is CV_8UC1");
    else
    {
        Mat image;
        if(_image.channels() != 1)
            cvtColor(_image, image, COLOR_BGR2GRAY);
        else
            image = _image.clone();

        /*check whether image depth is different from 0 */
        if(image.depth() != 0)
            CV_Error( Error::BadDepth, "Error, depth image!= 0" );

        /* prepare a vector to host extracted segments */
        vector<Vec4f> lines_lsd;

        /* create an LSD extractor */
//        Ptr<LineSegmentDetector> lsd = createLineSegmentDetector();
//        lsd->detect(image, lines_lsd);

        /* create an FLD extractor */
        Ptr<ximgproc::FastLineDetector> fld = ximgproc::createFastLineDetector(
                    50,
                    1.414213562f,
                    50,
                    50);
        fld->detect(image, lines_lsd);

        /* create keylines */
        int class_counter = -1;
        vector<line_descriptor::KeyLine> prelines;
        for (int k = 0; k < (int)lines_lsd.size(); k++)
        {


            line_descriptor::KeyLine kl;
            Vec4f extremes = lines_lsd[k];

            /* fill KeyLine's fields */
            kl.startPointX = extremes[0];
            kl.startPointY = extremes[1];
            kl.endPointX = extremes[2];
            kl.endPointY = extremes[3];
            kl.sPointInOctaveX = extremes[0];
            kl.sPointInOctaveY = extremes[1];
            kl.ePointInOctaveX = extremes[2];
            kl.ePointInOctaveY = extremes[3];
            kl.lineLength = (float)sqrt(pow(extremes[0] - extremes[2], 2) + pow(extremes[1] - extremes[3], 2));

            /* compute number of pixels covered by line */
            LineIterator li(image, Point2f(extremes[0], extremes[1]), Point2f(extremes[2], extremes[3]));
            kl.numOfPixels = li.count;
            kl.angle = atan2((kl.endPointY - kl.startPointY), (kl.endPointX - kl.startPointX));
            kl.class_id = ++class_counter;
            kl.octave = 0;
            kl.size = (kl.endPointX - kl.startPointX) * (kl.endPointY - kl.startPointY);
            kl.response = kl.lineLength / max(image.cols, image.rows);
            kl.pt = Point2f((kl.endPointX + kl.startPointX)/2, (kl.endPointY + kl.startPointY)/2);

            prelines.push_back(kl);
        }

        keylines = prelines;
        // 按照lineLength从大到小排序
//        sort(prelines.begin(), prelines.end(), [](line_descriptor::KeyLine &a, line_descriptor::KeyLine &b)
//        {
//            return a.lineLength > b.lineLength;
//        });

//        Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1);
//        for(auto &l : prelines)
//        {
//            if((mask.at<uchar>(l.getStartPoint()) == 0)
//                    || (mask.at<uchar>(l.getEndPoint()) == 0)
//                    || (mask.at<uchar>(l.pt) == 0))
//            {
//                keylines.push_back(l);
//                line(mask, l.getStartPoint(), l.getEndPoint(), 1, 10);// 5为线宽
//            }
//        }




//        /* delete undesired KeyLines, according to input mask */
//        if(!mask.empty())
//        {
//            for (size_t keyCounter = 0; keyCounter < keylines.size(); keyCounter++)
//            {
//                line_descriptor::KeyLine kl = keylines[keyCounter];
//                if(mask.at<uchar>((int)kl.startPointY, (int)kl.startPointX) == 0 && mask.at<uchar>((int)kl.endPointY, (int)kl.endPointX) == 0)
//                {
//                    keylines.erase(keylines.begin() + keyCounter);
//                    keyCounter--;
//                }
//            }
//        }
    }
}

LineFeature::LineFeature()
{
}

void LineFeature::PrecessImage(const Mat &_img)
{
    Mat img;

    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    clahe->apply(_img, img);// 图像均衡化

//    Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

//    filter2D(_img, img, _img.depth(), kernel);
    cur_img = img;


    vector<line_descriptor::KeyLine> lines;
    LineDetect(cur_img, lines);

    Ptr<line_descriptor::BinaryDescriptor> bd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    Mat line_des;
    bd->compute(cur_img, lines, line_des);

    cur_lines = lines;
    cur_des = line_des;
    cur_ids.clear();
    cur_track_cnt.clear();
    cur_ids.resize(cur_lines.size(), - 1);
    cur_track_cnt.resize(cur_lines.size(), 1);

//    vector<int> overleap;
//    int count;
    if(pre_lines.size() > 0)
    {
        vector<DMatch> cur2pre;
        vector<DMatch> pre2cur;
        Ptr<line_descriptor::BinaryDescriptorMatcher> matcher;
        matcher = line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

        matcher->match(cur_des, pre_des, cur2pre);
        matcher->match(pre_des, cur_des, pre2cur);

//        for (int i = 0; i < (int)cur2pre.size(); i++)
        for (auto mt : cur2pre)
        {
            if(mt.distance < 40 && (pre2cur[mt.trainIdx].trainIdx == mt.queryIdx))// 匹配好的线段用两条线在图像间的距离做reject
            {
//                cout << mt.distance << ", " << pre2cur[mt.trainIdx].distance << "       ";
                line_descriptor::KeyLine line1 = cur_lines[mt.queryIdx];
                line_descriptor::KeyLine line2 = pre_lines[mt.trainIdx];
                Point2f ssrr = line1.getStartPoint() - line2.getStartPoint();
                Point2f esrr = line1.getEndPoint() - line2.getStartPoint();
                Point2f serr = line1.getStartPoint() - line2.getEndPoint();
                Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                if( ((ssrr.dot(ssrr) < 50 * 50) || (esrr.dot(esrr) < 50 * 50)) &&
                    ((serr.dot(serr) < 50 * 50) || (eerr.dot(eerr) < 50 * 50)) &&
                    abs(line1.angle-line2.angle) < 0.3)
//                if(abs(line1.angle-line2.angle) < 0.2)
                {
                    cur_ids[mt.queryIdx] = pre_ids[mt.trainIdx];
                    cur_track_cnt[mt.queryIdx] = pre_track_cnt[mt.trainIdx] + 1;

//                    if(find(overleap.begin(), overleap.end(), pre_ids[mt.trainIdx]) == overleap.end())
//                    overleap.emplace_back(pre_ids[mt.trainIdx]);
//                    else
//                        count++;
                }
            }
        }
    }
//    cout << count << ", ";

    for (auto &cur_id : cur_ids)
        if(cur_id == -1)
            cur_id = line_id++;

    UndistortedEndPoints();

    pre_lines = cur_lines;
    pre_ids = cur_ids;
    pre_track_cnt = cur_track_cnt;
    pre_des = cur_des;

    img_id ++;
}


void LineFeature::UndistortedEndPoints()
{
    cur_un_lines.clear();
    for (auto &cur_line : cur_lines)
    {
        Vector2d s(cur_line.startPointX, cur_line.startPointY);
        Vector2d e(cur_line.endPointX, cur_line.endPointY);
        Vector3d s_un;
        Vector3d e_un;
        m_camera->liftProjective(s, s_un);// 相机模型去畸变
        m_camera->liftProjective(e, e_un);// 相机模型去畸变
        Point2f a = Point2f(s_un.x() / s_un.z(), s_un.y() / s_un.z());
        Point2f b = Point2f(e_un.x() / e_un.z(), e_un.y() / e_un.z());
        cur_un_lines.push_back(pair<Point2f, Point2f>(a, b));
    }
}


Mat LineFeature::GetLinesImg()
{
    Mat PointsImg;
    cvtColor(cur_img, PointsImg, COLOR_GRAY2RGB);
    Scalar Color;

    for (unsigned int i = 0; i < cur_lines.size(); i++)
    {
        if(cur_track_cnt[i] > 3)
            Color = Scalar(0, 255, 0);// 绿
//        else if(cur_track_cnt[i] > 5)
//            continue;
//            Color = Scalar(255, 0, 0);// 蓝
        else if(cur_track_cnt[i] > 0)
//            continue;
            Color = Scalar(0, 0, 255);// 红
        line(PointsImg, cur_lines[i].getStartPoint(), cur_lines[i].getEndPoint(), Color, 2);
    }
    return PointsImg;
}


Mat LineFeature::GetLinesImg(Lines &TrackingLines)
{
    Mat PointsImg;
    cvtColor(cur_img, PointsImg, COLOR_GRAY2RGB);
    Scalar Color = Scalar(0, 255, 0);// 绿

    vector<pair<int, line_descriptor::KeyLine>> SortLines;
    for (size_t i = 0; i < cur_track_cnt.size(); i++)
    {
        SortLines.push_back(pair<int, line_descriptor::KeyLine>(cur_ids[i], cur_lines[i]));
    }
    sort(SortLines.begin(), SortLines.end(), [](const pair<int, line_descriptor::KeyLine> &a,
         const pair<int, line_descriptor::KeyLine> &b)
         {
            return a.first < b.first;// 由小到大排列
         });

    for (int i = 0; i < 10; i++)
    {
        line(PointsImg, SortLines[i].second.getStartPoint(), SortLines[i].second.getEndPoint(), Color, 2);
    }
    return PointsImg;
}

void LineFeature::GetTrackingLines(Lines &TrackingLines)
{
    for (size_t i = 0; i < cur_track_cnt.size(); i++)
    {
        TrackingLines.emplace_back(cur_ids[i], cur_track_cnt[i],
                                   cur_un_lines[i].first.x, cur_un_lines[i].first.y,
                                   cur_un_lines[i].second.x, cur_un_lines[i].second.y);
    }
    sort(TrackingLines.begin(), TrackingLines.end(), [](const line_t &a, const line_t &b)
         {
            return a.id < b.id;// 由小到大排列
         });
}


void LineFeature::ReadIntrinsicParameter(const string &calib_file)
{
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}


