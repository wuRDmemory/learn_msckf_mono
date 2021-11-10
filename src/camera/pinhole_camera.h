#pragma once

#include "camera.h"

namespace CAMERA {

class PinHoleCamera : public Camera {
public:
  PinHoleCamera(int width, int height, string type, 
         float fx, float fy, float cx, float cy, 
         float k1, float k2, float d1, float d2);

  PinHoleCamera(const PinHoleCamera& camera) = delete;

  PinHoleCamera& operator=(const PinHoleCamera& camera) = delete;

  ~PinHoleCamera() override {}

  cv::Point3f imageToSpace(const cv::Point2f& pi) const override;

  cv::Point2f spaceToImage(const cv::Point3f& pc) const override;

  cv::Mat undistortImage(const cv::Mat& image, cv::Mat* new_camera_K) const override;

private:
  cv::Point2f distortion(const cv::Point2f& pu) const ;

private:
  bool no_undistort_ = true;
};

} // namespace CAMERA
