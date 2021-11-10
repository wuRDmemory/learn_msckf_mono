#include "msckf/datas.h"
#include "msckf/config.h"
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

} // namespace MSCKF
