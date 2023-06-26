// SO3 FUNCTIONS

#include "so3.h"

// 取w^
Eigen::Matrix3d Skew(const Eigen::Vector3d &w) {
  Eigen::Matrix3d W;
  W << 0.0, -w.z(), w.y(), w.z(), 0.0, -w.x(), -w.y(),  w.x(), 0.0;
  return W;
}

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z) {
  const double theta2 = x*x+y*y+z*z;
  const double theta  = std::sqrt(theta2);
  Eigen::Matrix3d W;
  W << 0.0, -z, y, z, 0.0, -x, -y,  x, 0.0;//(x,y,z)^
  // 轴角法表示的旋转转变为矩阵表示，使用罗德里格斯公式：（14讲P79页、P53页）
  if (theta < 1e-5)
      return Eigen::Matrix3d::Identity() + W;// + 0.5*W*W;
  else
      return Eigen::Matrix3d::Identity() + W*std::sin(theta)/theta + W*W*(1.0-std::cos(theta))/theta2;
}

// 旋转矩阵转李代数，（14讲P81页、P53页）
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R) {
  double costheta = 0.5*(R.trace()-1.0);
  if (costheta > +1.0) costheta = +1.0;
  if (costheta < -1.0) costheta = -1.0;
  const double theta = std::acos(costheta);// 先用迹的性质求转角theta，
  const Eigen::Vector3d w(R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1));// 先求转轴w，这里应该是2w，返回时*0.5
  if (theta < 1e-5)
      return 0.5*w;
  else
      return 0.5*theta*w/std::sin(theta);
}

Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z) {// 右扰动，BCH近似，公式见slam14讲82页
  const double theta2 = x*x+y*y+z*z;
  const double theta  = std::sqrt(theta2);

  Eigen::Matrix3d W;
  W << 0.0, -z, y, z, 0.0, -x, -y,  x, 0.0;
  if (theta < 1e-5)
      return Eigen::Matrix3d::Identity() - 0.5*W;// + 1.0/6.0*W*W;
  else
      return Eigen::Matrix3d::Identity() - W*(1.0-std::cos(theta))/theta2 + W*W*(theta-std::sin(theta))/(theta2*theta);
}

Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z) {
  const double theta2 = x*x+y*y+z*z;
  const double theta  = std::sqrt(theta2);

  Eigen::Matrix3d W;
  W << 0.0, -z, y, z, 0.0, -x, -y,  x, 0.0;
  if (theta < 1e-5)
      return Eigen::Matrix3d::Identity() + 0.5*W;// + 1.0/12.0*W*W;
  else
      return Eigen::Matrix3d::Identity() + 0.5*W + W*W*(1.0/theta2 - (1.0+std::cos(theta))/(2.0*theta*std::sin(theta)));
}

Eigen::Matrix3d LeftJacobianSO3(const double x, const double y, const double z) {// 右扰动，BCH近似，公式见slam14讲82页
  const double theta2 = x*x+y*y+z*z;
  const double theta  = std::sqrt(theta2);

  Eigen::Matrix3d W;
  W << 0.0, -z, y, z, 0.0, -x, -y,  x, 0.0;
  if (theta < 1e-5)
      return Eigen::Matrix3d::Identity() + 0.5*W;// + 1.0/6.0*W*W;
  else
      return Eigen::Matrix3d::Identity() + W*(1.0-std::cos(theta))/theta2 + W*W*(theta-std::sin(theta))/(theta2*theta);
}

Eigen::Matrix3d InverseLeftJacobianSO3(const double x, const double y, const double z) {
  const double theta2 = x*x+y*y+z*z;
  const double theta  = std::sqrt(theta2);

  Eigen::Matrix3d W;
  W << 0.0, -z, y, z, 0.0, -x, -y,  x, 0.0;
  if (theta < 1e-5)
      return Eigen::Matrix3d::Identity() - 0.5*W;
  else
      return Eigen::Matrix3d::Identity() - 0.5*W + W*W*(1.0/theta2 - (1.0+std::cos(theta))/(2.0*theta*std::sin(theta)));
}
