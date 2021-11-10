#include "camera_factory.h"
#include "glog/logging.h"

namespace CAMERA {

Camera *CameraFactory::createCamera(int width, int height, std::string type, 
         float fx, float fy, float cx, float cy, 
         float k1, float k2, float d1, float d2)
{
  if (type == "pinhole" || type == "PINHOLE") {
    return new PinHoleCamera(width, height, type, fx, fy, cx, cy, k1, k2, d1, d2);
  } else {
    LOG(ERROR) << "[CAMER] Do not support other camera.";
  }
}

} // namespace CAMERA
