#pragma once

#include "iostream"
#include "vector"

#include "Eigen/Core"
#include "Eigen/Dense"

#include "msckf/datas.h"

using namespace std;

namespace MATH_UTILS {

template<typename T>
Eigen::Matrix<T, 3, 3> skewMatrix(const Eigen::Matrix<T, 3, 1>& vec) {
  Eigen::Matrix<T, 3, 3> skew;
  skew << T(0), -vec(2), vec(1), 
          vec(2), T(0), -vec(0),
          -vec(1), vec(0), T(0);

  return skew;
}

template<typename T>
Eigen::Matrix<T, 3, 1> quaternionToRotateVector(const Eigen::Quaternion<T>& q)
{
  Eigen::Quaternion<T> q_n = q;
  if (0.9998 > q.norm() || q.norm() > 1.0002) {
    q_n.normalize();
  }
  T theta = 2.0*acos(q_n.w());
  T sin_half_theta = std::sqrt(1.0 - q_n.w() * q_n.w());
  Eigen::Matrix<T, 3, 1> vec = q_n.vec() / (sin_half_theta + 1.0e-7);

  return vec * theta;
}

template<typename T> 
Eigen::Quaternion<T> rotateVecToQuaternion(const Eigen::Matrix<T, 3, 1>& rotate_vec) {
  const double theta = rotate_vec.norm();
  Eigen::Matrix<T, 3, 1> axis  = rotate_vec.normalized();
  Eigen::Matrix<T, 3, 1> sin_axis = sin(0.5*theta)*axis;
  return Eigen::Quaternion<T>(cos(0.5*theta), sin_axis(0), sin_axis(1), sin_axis(2));
}

// in ZYX order
template<typename T> 
Eigen::Quaternion<T> eulerToQuaternion(const Eigen::Matrix<T, 3, 1>& ypr) {
  return  Eigen::AngleAxis<T>(ypr(0), Eigen::Vector3d::UnitZ())
         *Eigen::AngleAxis<T>(ypr(1), Eigen::Vector3d::UnitY())
         *Eigen::AngleAxis<T>(ypr(2), Eigen::Vector3d::UnitX());
}

template<typename T> 
Eigen::Matrix<T, 3, 1> rotate2ypr(const Eigen::Matrix<T, 3, 3> &R) {
  Eigen::Vector3d n = R.col(0);
  Eigen::Vector3d o = R.col(1);
  Eigen::Vector3d a = R.col(2);

  Eigen::Vector3d ypr(3);
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr;
}

} // namespace MATH_UTILS

