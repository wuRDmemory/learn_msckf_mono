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
}

Eigen::VectorXd ImuStatus::boxMinus(const ImuStatus& state) const
{
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(IMU_STATUS_DIM);
  dx.segment<3>(J_R) = MATH_UTILS::quaternionToRotateVector<double>(state.Rwb.inverse()*Rwb);
  dx.segment<3>(J_P) = pwb - state.pwb;
  dx.segment<3>(J_V) = vwb - state.vwb;
  dx.segment<3>(J_BA) = ba - state.ba;
  dx.segment<3>(J_BG) = bg - state.bg;
  return dx;
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
