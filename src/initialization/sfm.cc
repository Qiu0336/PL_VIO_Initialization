#include "sfm.h"

int cantconver = 0;

SFM::SFM(){}

void SFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool SFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}


bool SFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
                                vector<SFMFeature> &sfm_f, int &trinum)
{
    vector<cv::Point2f> pts_2_vector;
    vector<cv::Point3f> pts_3_vector;
    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state != true)
            continue;
        Vector2d point2d;
        for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
        {
            if (sfm_f[j].observation[k].first == i)
            {
                Vector2d img_pts = sfm_f[j].observation[k].second;
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
                cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
                pts_3_vector.push_back(pts_3);
                break;
            }
        }
    }
//    if (int(pts_2_vector.size()) < 15 || int(pts_2_vector.size()) < 0.5*trinum)
    if (int(pts_2_vector.size()) < 15)
    {
        printf("unstable features tracking, please slowly move you device!\n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
    if(!pnp_succ)
    {
        return false;
    }
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;

}


// 这里的pose是Tji，i是参考帧
int SFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
    int num = 0;
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
        if (has_0 && has_1)
        {
			Vector3d point_3d;
            // 三角化，求出来的点是在参考帧i下的3D坐标
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
            num ++;
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
    return num;
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool SFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
    feature_num = sfm_f.size();

	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
    T[frame_num - 1] = relative_T;

	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    // 获取Rji
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}
		// triangulate point based on the solve pnp result
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
        }
    }

	//full BA
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
                                    sfm_f[i].position);
		}

	}
	ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
        q[i] = q[i].inverse();
	}
	for (int i = 0; i < frame_num; i++)
	{
        T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;
}


bool SFM::ConstructInitial(int frame_num, Quaterniond* q, Vector3d* T, int l,
              const Matrix3d relative_R, const Vector3d relative_T,
              vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
    feature_num = sfm_f.size();

    q[0].w() = 1;
    q[0].x() = 0;
    q[0].y() = 0;
    q[0].z() = 0;
    T[0].setZero();
    q[l] = Quaterniond(relative_R);// R0i
    T[l] = relative_T;// T0i

    Matrix3d c_Rotation[frame_num];
    Vector3d c_Translation[frame_num];
    Quaterniond c_Quat[frame_num];
    double c_rotation[frame_num][4];
    double c_translation[frame_num][3];
    Eigen::Matrix<double, 3, 4> Pose[frame_num];


    c_Quat[0] = q[0].inverse();
    c_Rotation[0] = c_Quat[0].toRotationMatrix();
    c_Translation[0] = - c_Rotation[0]*T[0];
    Pose[0].block<3, 3>(0, 0) = c_Rotation[0];
    Pose[0].block<3, 1>(0, 3) = c_Translation[0];

    // 获取Ri0
    c_Quat[l] = q[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = - c_Rotation[l]*T[l];
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

//    triangulateTwoFrames(0, Pose[0], l, Pose[l], sfm_f);// 位姿都是Ri0，Rl0,

    for (int i = 0; i < l; i++)
    {
        // solve pnp
        if (i > 0)
        {
            Matrix3d R_initial = c_Rotation[i - 1];
            Vector3d P_initial = c_Translation[i - 1];
            if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))// 根据已有的3D点坐标求解位姿
                return false;
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }

        triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);// 位姿都是Ri0，Rl0,
        // 故三角化出来的点都是在第0帧下的点的坐标
    }

    for (int i = 1; i < l; i++)
        triangulateTwoFrames(0, Pose[0], i, Pose[i], sfm_f);


    for (int i = l + 1; i < frame_num; i++)
    {
        Matrix3d R_initial = c_Rotation[i - 1];
        Vector3d P_initial = c_Translation[i - 1];
        if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
            return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        //triangulate
        triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
    }
    //5: triangulate all other points
    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state == true)
            continue;
        if ((int)sfm_f[j].observation.size() >= 2)
        {
            Vector2d point0, point1;
            int frame_0 = sfm_f[j].observation[0].first;
            point0 = sfm_f[j].observation[0].second;
            int frame_1 = sfm_f[j].observation.back().first;
            point1 = sfm_f[j].observation.back().second;
            Vector3d point_3d;
            triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
            //cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
        }
    }

    //full BA
    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
    //cout << " begin full BA " << endl;
    for (int i = 0; i < frame_num; i++)
    {
        //double array for ceres
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();
        problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translation[i], 3);
        if (i == 0)
        {
            problem.SetParameterBlockConstant(c_rotation[i]);
        }
        if (i == 0 || i == l)
        {
            problem.SetParameterBlockConstant(c_translation[i]);
        }
    }

    for (int i = 0; i < feature_num; i++)
    {
        if (sfm_f[i].state != true)
            continue;
        for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
        {
            int l = sfm_f[i].observation[j].first;
            ceres::CostFunction* cost_function = ReprojectionError3D::Create(
                                                sfm_f[i].observation[j].second.x(),
                                                sfm_f[i].observation[j].second.y());

            problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l],
                                    sfm_f[i].position);
        }

    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
    {
        //cout << "vision only BA converge" << endl;
    }
    else
    {
        cout << "vision only BA can't converge !!! " << endl;
        return false;
    }
    for (int i = 0; i < frame_num; i++)
    {
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i] = q[i].inverse();
    }
    for (int i = 0; i < frame_num; i++)
    {
        T[i] = - (q[i]*Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    }
    for (int i = 0; i < (int)sfm_f.size(); i++)
    {
        if(sfm_f[i].state)
            sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
    }
    return true;

}





bool SFM::ConstructIncreasing(int frame_num, Quaterniond* q, Vector3d* T, int l,
              const Matrix3d relative_R, const Vector3d relative_T,
              vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points,
                         const io::AllPoints &AP)
{
    feature_num = sfm_f.size();

    q[0].w() = 1;
    q[0].x() = 0;
    q[0].y() = 0;
    q[0].z() = 0;
    T[0].setZero();
    q[l] = Quaterniond(relative_R);// R0i
    T[l] = relative_T;// T0i

    Matrix3d c_Rotation[frame_num];
    Vector3d c_Translation[frame_num];
    Quaterniond c_Quat[frame_num];
    double c_rotation[frame_num][4];
    double c_translation[frame_num][3];
    Eigen::Matrix<double, 3, 4> Pose[frame_num];


    c_Quat[0] = q[0].inverse();
    c_Rotation[0] = c_Quat[0].toRotationMatrix();
    c_Translation[0] = - c_Rotation[0]*T[0];
    Pose[0].block<3, 3>(0, 0) = c_Rotation[0];
    Pose[0].block<3, 1>(0, 3) = c_Translation[0];

    // 获取Ri0
    c_Quat[l] = q[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = - c_Rotation[l]*T[l];
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    int trinum = triangulateTwoFrames(0, Pose[0], l, Pose[l], sfm_f);// 位姿都是Ri0，Rl0,

    for(int i = 1; i < l; i++)
    {
        Matrix3d R_initial = c_Rotation[i - 1];
        Vector3d P_initial = c_Translation[i - 1];
        if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f, trinum))// 根据已有的3D点坐标求解位姿
            return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    }

    int lasti = 0;// 用来判断死循环
    for (int i = l + 1; i < frame_num; i++)
    {
        Matrix3d R_initial = c_Rotation[i - 1];
        Vector3d P_initial = c_Translation[i - 1];
        if(solveFrameByPnP(R_initial, P_initial, i, sfm_f, trinum))// 3D点数量还够
        {
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }
        else// 3D点数量不够了，该三角化了
        {
            if(i == lasti)// 陷入死循环
                return false;
            lasti = i;

            trinum = 0;
            for(int j = i - 2; j >= 0; j--)// 寻找与最新帧（i - 1）帧视差足够大的帧进行三角化
            {
                vector<pair<Vector3d, Vector3d>> corres;
                corres = GetMatchingPoints(AP[j].points, AP[i - 1].points);
                if(corres.size() < 20)
                    return false;
                double ave_parallax = ComputeParallax(corres);
                if(ave_parallax*460 > 30)
                {
                    trinum = corres.size();
                    triangulateTwoFrames(j, Pose[j], i - 1, Pose[i - 1], sfm_f);
                    break;
                }
            }
            if(trinum == 0)
                return false;
            i--;
        }
    }
    //5: 三角化剩余的点
    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state == true)
            continue;
        if ((int)sfm_f[j].observation.size() >= 2)
        {
            Vector2d point0, point1;
            int frame_0 = sfm_f[j].observation[0].first;
            point0 = sfm_f[j].observation[0].second;
            int frame_1 = sfm_f[j].observation.back().first;
            point1 = sfm_f[j].observation.back().second;
            Vector3d point_3d;
            triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
            //cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
        }
    }

    //full BA
    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
    //cout << " begin full BA " << endl;
    for (int i = 0; i < frame_num; i++)
    {
        //double array for ceres
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();
        problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translation[i], 3);
        if (i == 0)
        {
            problem.SetParameterBlockConstant(c_rotation[i]);
        }
        if (i == 0 || i == l)
        {
            problem.SetParameterBlockConstant(c_translation[i]);
        }
    }

    for (int i = 0; i < feature_num; i++)
    {
        if (sfm_f[i].state != true)
            continue;
        for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
        {
            int l = sfm_f[i].observation[j].first;
            ceres::CostFunction* cost_function = ReprojectionError3D::Create(
                                                sfm_f[i].observation[j].second.x(),
                                                sfm_f[i].observation[j].second.y());

            problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l],
                                    sfm_f[i].position);
        }

    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
//    options.max_solver_time_in_seconds = 2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
    {
        //cout << "vision only BA converge" << endl;
    }
    else
    {
        cout << "vision only BA can't converge !!! " << endl;
        cantconver++;
        return false;
    }
    for (int i = 0; i < frame_num; i++)
    {
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i] = q[i].inverse();
    }
    for (int i = 0; i < frame_num; i++)
    {
        T[i] = - (q[i]*Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    }
    for (int i = 0; i < (int)sfm_f.size(); i++)
    {
        if(sfm_f[i].state)
            sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
    }
    return true;

}



// 获取两帧之间的所有匹配点
vector<pair<Vector3d, Vector3d>> GetMatchingPoints(const io::Points &PtFrames1, const io::Points &PtFrames2)
{
    vector<pair<Vector3d, Vector3d>> Res;
    for(auto it1 = PtFrames1.begin(), it2 = PtFrames2.begin();
        it1 != PtFrames1.end() && it2 != PtFrames2.end();)
    {
        if(it1->id < it2->id)
            ++ it1;
        else if(it1->id > it2->id)
            ++ it2;
        else if(it1->id == it2->id)
        {
            pair<Vector3d, Vector3d> tmp;
            tmp.first = Vector3d(it1->x, it1->y, 1.0f);
            tmp.second = Vector3d(it2->x, it2->y, 1.0f);
            Res.push_back(tmp);
            ++ it1;
            ++ it2;
        }
    }
    return Res;
}

// 获取匹配点的视差
double ComputeParallax(vector<pair<Vector3d, Vector3d>>& corres)
{
    double sum_parallax = 0;
    double average_parallax;
    for(int j = 0; j < int(corres.size()); j++)
    {
      //第j个对应点在第i帧和最后一帧的(x,y)
        Vector2d pts_0(corres[j].first(0), corres[j].first(1));
        Vector2d pts_1(corres[j].second(0), corres[j].second(1));
        double parallax = (pts_0 - pts_1).norm();
        sum_parallax += parallax;
    }
    average_parallax = 1.0 * sum_parallax / int(corres.size());
    return average_parallax;
}


bool SolveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation)
{
    if(corres.size() <= 15)
        return false;

    vector<cv::Point2f> ll, rr;
    for(int i = 0; i < int(corres.size()); i++)
    {
        ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
        rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
    }
    cv::Mat mask;
    cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat rot, trans;
    int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);

    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    for(int i = 0; i < 3; i++)
    {
        T(i) = trans.at<double>(i, 0);
        for(int j = 0; j < 3; j++)
            R(i, j) = rot.at<double>(i, j);
    }
    Rotation = R.transpose();
    Translation = - R.transpose() * T;
    if(inlier_cnt > 12)
        return true;
    else
        return false;
}


//bool CheckParallel()
//{
//    s;
//}


bool StructureInit(const io::AllPoints &AP, vector<Eigen::Matrix3d> &r, vector<Eigen::Vector3d> &t)
{
    Timer timer;
    timer.Start();

    const int frames = AP.size();
    vector<SFMFeature> sfm_f;

    int last_id = -1;
    int frameid = 0;

    for(auto &fp : AP)
    {
        for(auto &pt : fp.points)
        {
            const int ptid = pt.id;
            if(ptid > last_id)// sfm_f还未添加该点，故添加
            {
                last_id = pt.id;
                SFMFeature tmp_feature;
                tmp_feature.state = false;
                tmp_feature.id = pt.id;
                tmp_feature.observation.push_back(make_pair(frameid, Eigen::Vector2d(pt.x, pt.y)));
                sfm_f.push_back(tmp_feature);
            }
            else// sfm_f已添加该点
            {
                for(auto &s : sfm_f)
                {
                    if(s.id == ptid)
                    {
                        s.observation.push_back(make_pair(frameid, Eigen::Vector2d(pt.x, pt.y)));
                        break;
                    }
                }
            }
        }
        frameid ++;
    }

    // 删除只有一个观测的点
    for(vector<SFMFeature>::iterator it = sfm_f.begin(); it != sfm_f.end();)
    {
        if(it->observation.size() == 1)
            it = sfm_f.erase(it);
        else
            it++;
    }

    Matrix3d relative_R;
    Vector3d relative_T;
    int ref = -1;// 参考帧索引

    timer.PrintSeconds();

    // find previous frame which contians enough correspondance and parallex with newest frame
    // 寻找参考帧，（与最后一帧需要有足够视差）
    for(int i = 0; i < frames - 1; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        //寻找第i帧到窗口最后一帧的对应特征点
        corres = GetMatchingPoints(AP[i].points, AP.back().points);
        if(corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for(int j = 0; j < int(corres.size()); j++)
            {
              //第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax += parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            //判断是否满足初始化条件：视差>30和内点数满足要求(大于12)
            //solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
            //同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的relative_R，relative_T
            if(average_parallax * 460 > 30 && SolveRelativeRT(corres, relative_R, relative_T))
            {
                ref = i;
                break;
            }
        }
    }

    if(ref == -1)// 未找到参考帧
    {
        cout << "Not enough features or parallax; Move device around" << endl;
        return false;
    }

    // 这里得到的Q T全部是以i为参考帧的Tij
    Quaterniond Q[frames];
    Vector3d T[frames];
    map<int, Vector3d> sfm_tracked_points;
    SFM sfm;
    // 这里传入的relative_R和relative_T是Rij,tij，其中i是参考帧，j是最后一帧
    if(!sfm.construct(frames, Q, T, ref,
            relative_R, relative_T,
            sfm_f, sfm_tracked_points))
    {
        cout << "global SFM failed!" << endl;
        return false;
    }

    Matrix3d Rot0i = Q[0].toRotationMatrix().transpose();
    Matrix3d Rot[frames];
    for(int i = 0; i < frames; i++)
    {
        Rot[i] = Rot0i*Q[i].toRotationMatrix();
    }
    Vector3d T0i = - Rot0i*T[0];
    for(int i = 0; i < frames; i++)
    {
        T[i] = Rot0i*T[i] + T0i;
    }

//    for(int i = 0; i < frames; i++)
//    {
//        cout << "frame " << i << " rot: " << endl << Rot[i] << endl;
//        cout << "trans: " << T[i].transpose() << endl << endl;
//    }

    for(int i = 0; i < frames; i++)
    {
        r.push_back(Rot[i]);
        t.push_back(T[i]);
    }

    return true;

}



bool StructureInitFromInitial(const io::AllPoints &AP, vector<Eigen::Matrix3d> &r, vector<Eigen::Vector3d> &t)
{
    Timer timer;
    timer.Start();

    const int frames = AP.size();
    vector<SFMFeature> sfm_f;

    int last_id = -1;
    int frameid = 0;

    for(auto &fp : AP)
    {
        for(auto &pt : fp.points)
        {
            const int ptid = pt.id;
            if(ptid > last_id)// sfm_f还未添加该点，故添加
            {
                last_id = pt.id;
                SFMFeature tmp_feature;
                tmp_feature.state = false;
                tmp_feature.id = pt.id;
                tmp_feature.observation.push_back(make_pair(frameid, Eigen::Vector2d(pt.x, pt.y)));
                sfm_f.push_back(tmp_feature);
            }
            else// sfm_f已添加该点
            {
                for(auto &s : sfm_f)
                {
                    if(s.id == ptid)
                    {
                        s.observation.push_back(make_pair(frameid, Eigen::Vector2d(pt.x, pt.y)));
                        break;
                    }
                }
            }
        }
        frameid ++;
    }

    // 删除只有一个观测的点
    for(vector<SFMFeature>::iterator it = sfm_f.begin(); it != sfm_f.end();)
    {
        if(it->observation.size() == 1)
            it = sfm_f.erase(it);
        else
            it++;
    }

    Matrix3d relative_R;
    Vector3d relative_T;
    int ref = -1;// 初始化参考帧索引

//    timer.PrintSeconds();

    // find previous frame which contians enough correspondance and parallex with newest frame
    // 寻找参考帧，（与第一帧需要有足够视差）
    for(int i = 1; i < frames; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        //寻找第i帧到窗口最后一帧的对应特征点
        corres = GetMatchingPoints(AP[0].points, AP[i].points);
        if(corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for(int j = 0; j < int(corres.size()); j++)
            {
              //第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax += parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            //判断是否满足初始化条件：视差>30和内点数满足要求(大于12)
            //solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
            //同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的relative_R，relative_T
            if(average_parallax * 460 > 30 && SolveRelativeRT(corres, relative_R, relative_T))
            {
                ref = i;
                break;
            }
        }
    }

    if(ref == -1)// 未找到参考帧
    {
        cout << "Can't initialize frames, Not enough features or parallax, Move device" << endl;
        return false;
    }

    cout << "ref frame: " << ref << endl;

    // 这里得到的Q T全部是以i为参考帧的Tij
    Quaterniond Q[frames];
    Vector3d T[frames];
    map<int, Vector3d> sfm_tracked_points;
    SFM sfm;
    // 这里传入的relative_R和relative_T是R0i,t0i，其中0是第一帧，i是初始化参考帧
    if(!sfm.ConstructInitial(frames, Q, T, ref,
            relative_R, relative_T,
            sfm_f, sfm_tracked_points))
    {
        cout << "global SFM failed!" << endl;
        return false;
    }

    Matrix3d Rot[frames];
    for(int i = 0; i < frames; i++)
    {
        Rot[i] = Q[i].toRotationMatrix();
    }

    for(int i = 0; i < frames; i++)
    {
        cout << "frame " << i << " rot: " << endl << Rot[i] << endl;
        cout << "trans: " << T[i].transpose() << endl << endl;
    }

    for(int i = 0; i < frames; i++)
    {
        r.push_back(Rot[i]);
        t.push_back(T[i]);
    }


    return true;

}



bool StructureInitIncreasing(const io::AllPoints &AP, vector<Eigen::Matrix3d> &r, vector<Eigen::Vector3d> &t)
{
    Timer timer;
    timer.Start();

    const int frames = AP.size();
    vector<SFMFeature> sfm_f;

    int last_id = -1;
    int frameid = 0;

    for(auto &fp : AP)
    {
        for(auto &pt : fp.points)
        {
            const int ptid = pt.id;
            if(ptid > last_id)// sfm_f还未添加该点，故添加
            {
                last_id = pt.id;
                SFMFeature tmp_feature;
                tmp_feature.state = false;
                tmp_feature.id = pt.id;
                tmp_feature.observation.push_back(make_pair(frameid, Eigen::Vector2d(pt.x, pt.y)));
                sfm_f.push_back(tmp_feature);
            }
            else// sfm_f已添加该点
            {
                for(auto &s : sfm_f)
                {
                    if(s.id == ptid)
                    {
                        s.observation.push_back(make_pair(frameid, Eigen::Vector2d(pt.x, pt.y)));
                        break;
                    }
                }
            }
        }
        frameid ++;
    }

    // 删除只有一个观测的点
    for(vector<SFMFeature>::iterator it = sfm_f.begin(); it != sfm_f.end();)
    {
        if(it->observation.size() == 1)
            it = sfm_f.erase(it);
        else
            it++;
    }

    Matrix3d relative_R;
    Vector3d relative_T;
    int ref = -1;// 初始化参考帧索引

    // 寻找参考帧，（与第一帧需要有足够视差）
    for(int i = 1; i < frames; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        //寻找第i帧到窗口最后一帧的对应特征点
        corres = GetMatchingPoints(AP[0].points, AP[i].points);
        if(corres.size() > 20)
        {
            double average_parallax = ComputeParallax(corres);
            //判断是否满足初始化条件：视差>30和内点数满足要求(大于12)
            //solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
            //同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的relative_R，relative_T
            if(average_parallax * 460 > 30 && SolveRelativeRT(corres, relative_R, relative_T))
            {
                ref = i;
                break;
            }
        }
    }

    if(ref == -1)// 未找到参考帧
    {
        cout << "Can't initialize frames, Not enough features or parallax, Move device" << endl;
        return false;
    }

    cout << "ref frame: " << ref << endl;

    // 这里得到的Q T全部是以i为参考帧的Tij
    Quaterniond Q[frames];
    Vector3d T[frames];
    map<int, Vector3d> sfm_tracked_points;
    SFM sfm;
    // 这里传入的relative_R和relative_T是R0i,t0i，其中0是第一帧，i是初始化参考帧
    if(!sfm.ConstructIncreasing(frames, Q, T, ref,
            relative_R, relative_T,
            sfm_f, sfm_tracked_points, AP))
    {
        cout << "global SFM failed!" << endl;
        return false;
    }

    Matrix3d Rot[frames];
    for(int i = 0; i < frames; i++)
    {
        Rot[i] = Q[i].toRotationMatrix();
    }

    for(int i = 0; i < frames; i++)
    {
        cout << "frame " << i << " rot: " << endl << Rot[i] << endl;
        cout << "trans: " << T[i].transpose() << endl << endl;
    }

    for(int i = 0; i < frames; i++)
    {
        r.push_back(Rot[i]);
        t.push_back(T[i]);
    }


    return true;

}



