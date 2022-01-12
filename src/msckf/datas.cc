#include "msckf/datas.h"
#include "msckf/config.h"
#include "msckf/math_utils.h"
#include "glog/logging.h"

namespace MSCKF {

std::ostream& operator<<(std::ostream& os, const ImuStatus& imu)
{
  os << "Imu Status:" << endl;
  os << "|- qwb:    " << imu.Rwb.coeffs().transpose() << endl;
  os << "|- vwb:    " << imu.vwb.transpose() << endl;
  os << "|- pwb:    " << imu.pwb.transpose() << endl;
  os << "|- bias_a: " << imu.ba.transpose() << endl;
  os << "|- bias_w: " << imu.bg.transpose() << endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const CameraStatus& cam)
{
  os << setprecision(15) 
     << cam.ts << ", " \
     << cam.pwc.x() << ", " << cam.pwc.y() << ", " << cam.pwc.z() << ", " \
     << cam.Rwc.w() << ", " << cam.Rwc.x() << ", " << cam.Rwc.y() << ", " << cam.Rwc.z();
  return os;
}

void ImuStatus::boxPlus(const Eigen::VectorXd& delta_imu)
{
  Rwb *= MATH_UTILS::rotateVecToQuaternion<double>(delta_imu.segment<3>(J_R)); Rwb.normalize();
  vwb += delta_imu.segment<3>(J_V);
  pwb += delta_imu.segment<3>(J_P);
  bg  += delta_imu.segment<3>(J_BG);
  ba  += delta_imu.segment<3>(J_BA);

  // gravity update
  Eigen::MatrixXd bu = S2Bx(); // 3 x 2
  Eigen::Vector3d vec = bu*delta_imu.segment<2>(J_G);
  Eigen::Quaterniond q = MATH_UTILS::rotateVecToQuaternion(vec);
  g = q*g;
}

Eigen::VectorXd ImuStatus::boxMinus(const ImuStatus& state) const
{
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(IMU_STATUS_DIM);
  dx.segment<3>(J_R) = MATH_UTILS::quaternionToRotateVector<double>(state.Rwb.inverse()*Rwb);
  dx.segment<3>(J_P) = pwb - state.pwb;
  dx.segment<3>(J_V) = vwb - state.vwb;
  dx.segment<3>(J_BA) = ba - state.ba;
  dx.segment<3>(J_BG) = bg - state.bg;

  // gravity minus
  const Eigen::Vector3d& vec = g;
  Eigen::Vector2d res;
  double v_sin = (MATH_UTILS::skewMatrix(vec)*state.g).norm();
  double v_cos = vec.transpose() * state.g;
  double theta = std::atan2(v_sin, v_cos);
  if(v_sin < eps) {
    if(std::fabs(theta) > eps) {
      res[0] = 3.1415926;
      res[1] = 0;
    } else {
      res[0] = 0;
      res[1] = 0;
    }
  } else {
    Eigen::MatrixXd Bx = state.S2Bx();
    res = theta/v_sin * Bx.transpose() * MATH_UTILS::skewMatrix(state.g)*vec;
  }
  dx.segment<2>(J_G) = res;
  return dx;
}

Eigen::MatrixXd ImuStatus::S2Bx() const
{
  const Eigen::Vector3d& vec = g;
  const double length = vec.norm();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(3, 2);
  if(vec[2] + length > 1.e-7) { 
    res << length - vec[0]*vec[0]/(length+vec[2]), -vec[0]*vec[1]/(length+vec[2]),
        -vec[0]*vec[1]/(length+vec[2]), length-vec[1]*vec[1]/(length+vec[2]),
        -vec[0], -vec[1];
    res /= length;
  } else {
    res(1, 1) = -1;
    res(2, 0) = 1;
  }
  return res;
}

void CameraStatus::boxPlus(const Eigen::VectorXd& delta_cam)
{
  Rwc *= MATH_UTILS::rotateVecToQuaternion<double>(delta_cam.segment<3>(C_R)); Rwc.normalize();
  pwc += delta_cam.segment<3>(C_P);
}

Eigen::VectorXd CameraStatus::boxMinus(const CameraStatus& state) const
{
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(6);
  dx.segment<3>(C_R) = MATH_UTILS::quaternionToRotateVector<double>(state.Rwc.inverse()*Rwc);
  dx.segment<3>(C_P) = pwc - state.pwc;
  return dx;
}

} // namespace MSCKF
