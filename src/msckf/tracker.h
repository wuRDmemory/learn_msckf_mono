#pragma once

#include "common.h"
#include "config.h"
#include "tic_toc.h"

#include "msckf/utils.h"
#include "msckf/datas.h"
#include "camera/camera_factory.h"

using namespace std;

namespace MSCKF {

class ImageTracker {
private:
  static int ID;
  double  prev_pub_ts_ = -1;
  double  last_ts_ = -1;

  cv::Mat last_image_;
  cv::Mat prev_image_;
  cv::Point2f velocity_;

  std::vector<int>         points_id_;
  std::vector<int>         track_cnt_;
  std::vector<cv::Point2f> prev_points_;
  std::vector<cv::Point2f> last_points_;
  std::vector<cv::Point2f> prev_points_un_;
  std::vector<cv::Point2f> last_points_un_;
  std::vector<cv::Rect>    rois_;
  CAMERA::Camera *camera_ = NULL;

  const TrackParam &param_;

public:
  ImageTracker(TrackParam &param, CameraParam &camera_param);
  ~ImageTracker();
  bool feedImage(const double& time, const cv::Mat& curr_image);
  TrackResult fetchResult();

private:
  std::vector<uchar> checkOutOfBorder(const std::vector<cv::Point2f> &pts);
  std::vector<uchar> checkWithFundamental();
  cv::Mat setMask();
  cv::Point2f computeVelocity(bool show_match = false);

private:
  void testUndistortPoint();
  void testTrackConsistency();
};

} // namespace MSCKF

