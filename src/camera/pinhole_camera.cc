#include "pinhole_camera.h"

namespace CAMERA {

PinHoleCamera::PinHoleCamera(cv::FileNode& n):
    Camera(n) {
  no_undistort_ = (k1_ == 0);
}

PinHoleCamera::PinHoleCamera(string camera_config_path):
    Camera(camera_config_path) {
  no_undistort_ = (k1_ == 0);
}

cv::Point3f PinHoleCamera::imageToSpace(const cv::Point2f& pi) const {
  cv::Point2f pc;
  pc.x = (pi.x - cx_)/fx_;
  pc.y = (pi.y - cy_)/fy_;

  if (!no_undistort_) {
    const int n = 8;
    cv::Point2f du = distortion(cv::Point2f(pc.x, pc.y));
    cv::Point2f pu = pc - du;

    for (int i = 1; i < n; ++i) {
      du = distortion(pu);
      pu = pc - du;
    }

    return cv::Point3f(pu.x, pu.y, 1.0);
  }
  
  return cv::Point3f(pc.x, pc.y, 1.0);
}

cv::Point2f PinHoleCamera::spaceToImage(const cv::Point3f& pc) const {
  cv::Point2f pc_dis(pc.x/pc.z, pc.y/pc.z);
  cv::Point2f pi;  

  if (!no_undistort_) {
    cv::Point2f du = distortion(cv::Point2f(pc.x, pc.y));
    pc_dis += du;
  }

  pi.x = pc_dis.x*fx_ + cx_;
  pi.y = pc_dis.y*fy_ + cy_;

  return pi;
}

cv::Point2f PinHoleCamera::distortion(const cv::Point2f& pu) const {
  const double xx_u = pu.x*pu.x;
  const double yy_u = pu.y*pu.y;
  const double xy_u = pu.x*pu.y;
  const double rho2_u = xx_u + yy_u;
  
  double rad_dist_u = k1_*rho2_u + k2_*rho2_u*rho2_u;
  return cv::Point2f(
        pu.x*rad_dist_u + 2.0*d1_*xy_u + d2_*(rho2_u + 2.0*xx_u),
        pu.y*rad_dist_u + 2.0*d2_*xy_u + d1_*(rho2_u + 2.0*yy_u));
}

cv::Mat PinHoleCamera::undistortImage(const cv::Mat& image, cv::Mat* new_camera_K) const {
  cv::Mat undistort_image;

  cv::Mat K = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
  cv::Mat D = (cv::Mat_<double>(5, 1) << k1_, k2_, d1_, d2_, 0);

  cv::Mat  m1, m2;
  cv::Mat  new_K = cv::getOptimalNewCameraMatrix(K, D, cv::Size(width_, height_), 1.0);
  cv::initUndistortRectifyMap(K, D, cv::Mat(), new_K, cv::Size(width_, height_), CV_16SC2, m1, m2);

  cv::remap(image, undistort_image, m1, m2, cv::INTER_LINEAR);

  if (new_camera_K) {
    *new_camera_K = new_K.clone();
  }

  return undistort_image;
}

} // namespace CAMERA
