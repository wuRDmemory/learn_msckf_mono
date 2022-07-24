#pragma once

#include "common.h"
#include "config.h"
#include "tic_toc.h"

#include "msckf/utils.h"
#include "msckf/datas.h"
#include "opencv2/line_descriptor.hpp"
#include "opencv2/features2d.hpp"
#include "camera/camera_factory.h"

using namespace std;

namespace MSCKF {

class ImageTracker {
private:
  static int ID;
  static int LID;
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

  // Line feature
  std::vector<int>    line_track_cnt_;
  std::vector<int>    line_feature_id_;
  std::vector<double> line_feature_ts_;
  cv::Ptr<cv::DescriptorMatcher> line_matcher_;
  std::vector<cv::line_descriptor::KeyLine> prev_line_features_;
  cv::Mat prev_line_features_desc_;

public:
  ImageTracker(TrackParam &param, CameraParam &camera_param);
  ~ImageTracker();
  bool feedImage(const double& time, const cv::Mat& curr_image);
  bool feedImageLine(const double& time, const cv::Mat& curr_image);
  TrackResult fetchResult();

private:
  std::vector<uchar> checkOutOfBorder(const std::vector<cv::Point2f> &pts);
  std::vector<uchar> checkWithFundamental();
  cv::Mat setMask();
  cv::Mat setLineMask();
  bool isIntersect(LINE_ENDS line, cv::Mat& image);
  cv::Point2f computeVelocity(bool show_match = false);
  int extractLineFeature(const cv::Mat& image, const cv::Mat& mask, std::vector<cv::line_descriptor::KeyLine>& lines, cv::Mat& descrips);
  int matchLineFeature(const cv::Mat& ref_descrip, const cv::Mat& dst_descrip, std::vector<cv::DMatch>& matches);
  int outlierCheck(const std::vector<LINE_ENDS>& ref_lines, std::vector<LINE_ENDS>& cur_lines);
  int fetchMatchLines(vector<cv::line_descriptor::KeyLine>& train_lines, vector<cv::line_descriptor::KeyLine>& query_lines, vector<cv::DMatch>& matches, 
      vector<LINE_ENDS>& ref_lines, vector<LINE_ENDS>& cur_lines);

private:
  void testUndistortPoint();
  void testTrackConsistency();
};

} // namespace MSCKF

