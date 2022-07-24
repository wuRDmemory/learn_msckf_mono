#pragma once

#include "camera.h"
#include "pinhole_camera.h"

namespace CAMERA {

class CameraFactory {
public:
  static Camera *createCamera(int width, int height, std::string type, 
         float fx, float fy, float cx, float cy, 
         float k1, float k2, float d1, float d2);
};

} // namespace CAMERA