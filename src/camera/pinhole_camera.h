#pragma once

#include "camera.h"

namespace CAMERA {

class PinHoleCamera : public Camera {
public:
  PinHoleCamera(cv::FileNode& n);

  PinHoleCamera(string camera_config_path);

  PinHoleCamera(const PinHoleCamera& camera) = delete;

  PinHoleCamera& operator=(const PinHoleCamera& camera) = delete;

  ~PinHoleCamera() override {}

  cv::Point3f imageToSpace(const cv::Point2f& pi) const override;

  cv::Point2f spaceToImage(const cv::Point3f& pc) const override;

  cv::Mat undistortImage(const cv::Mat& image, cv::Mat* new_camera_K) const override;

private:
  cv::Point2f distortion(const cv::Point2f& pu) const ;

private:
  bool no_undistort_;
};

} // namespace CAMERA
