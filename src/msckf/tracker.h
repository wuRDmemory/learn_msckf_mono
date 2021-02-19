#pragma once

#include "iostream"
#include "vector"
#include "algorithm"
#include "memory.h"

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"

#include "opencv2/opencv.hpp"

#include "msckf/utils.h"
#include "msckf/datas.h"
#include "camera/camera.h" 
#include "camera/pinhole_camera.h"

using namespace std;

namespace MSCKF {

class ImageTracker {
private:
  static int ID;
  bool    verbose_;
  double  prev_pub_ts_;
  double  last_ts_;

  cv::Mat last_image_;
  cv::Mat prev_image_;

  vector<int>         points_id_;
  vector<int>         track_cnt_;
  vector<cv::Point2f> prev_points_;
  vector<cv::Point2f> last_points_;
  vector<cv::Point2f> prev_points_un_;
  vector<cv::Point2f> last_points_un_;
  vector<cv::Rect>    rois_;
 
  cv::Point2f velocity_;
 
  unique_ptr<CAMERA::Camera> camera_;

public:
  ImageTracker(string camera_config_file, bool verbose = true);

  bool feedImage(const double& time, const cv::Mat& curr_image);

  TrackResult fetchResult();

private:

  vector<uchar> checkWithFundamental();

  cv::Mat setMask();

  cv::Point2f computeVelocity(bool show_match = false);

private:
  void testUndistortPoint();

  void testTrackConsistency();

};

} // namespace MSCKF

