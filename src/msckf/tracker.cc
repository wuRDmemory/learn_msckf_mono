#include "tracker.h"

namespace MSCKF {

namespace {

template<typename T>
int reduceVector(vector<T>& data, const vector<uchar>& status) {
  int j = 0;
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i]) {
      data[j++] = data[i];
    }
  }
  data.resize(j);
  return j;
}

cv::Point2f projectToImage(const cv::Point3f& pc, const cv::Mat& K) {
  cv::Point2f pi;
  pi.x = K.at<double>(0, 0)*pc.x/pc.z + K.at<double>(0, 2);
  pi.y = K.at<double>(1, 1)*pc.y/pc.z + K.at<double>(1, 2);
  return pi;
}

} // namespace anonymous

int ImageTracker::ID = 0;
int ImageTracker::LID = 0;

ImageTracker::ImageTracker(TrackParam &param, CameraParam &camera_param) :
    param_(param)
{
  camera_ = CAMERA::CameraFactory::createCamera(
    camera_param.width, camera_param.height, camera_param.type,
    camera_param.fx, camera_param.fy, camera_param.cx, camera_param.cy,
    camera_param.k1, camera_param.k2, camera_param.d1, camera_param.d2
  );

  const int width  = camera_->width();
  const int height = camera_->height();
  const int width_step  = width/3;
  const int height_step = height/3;
  rois_.resize(9);

  rois_[0] = cv::Rect(0*width_step, 0*height_step,         width_step, height_step); 
  rois_[1] = cv::Rect(1*width_step, 0*height_step,         width_step, height_step); 
  rois_[2] = cv::Rect(2*width_step, 0*height_step, width-2*width_step, height_step);
  rois_[3] = cv::Rect(0*width_step, 1*height_step,         width_step, height_step); 
  rois_[4] = cv::Rect(1*width_step, 1*height_step,         width_step, height_step); 
  rois_[5] = cv::Rect(2*width_step, 1*height_step, width-2*width_step, height_step);
  rois_[6] = cv::Rect(0*width_step, 2*height_step,         width_step, height-2*height_step); 
  rois_[7] = cv::Rect(1*width_step, 2*height_step,         width_step, height-2*height_step); 
  rois_[8] = cv::Rect(2*width_step, 2*height_step, width-2*width_step, height-2*height_step);
  
  LOG(INFO) << "Image Tracker start!!!";
}

ImageTracker::~ImageTracker()
{
  if (camera_) {
    delete camera_;
  }
}

bool ImageTracker::feedImage(const double& time, const cv::Mat& curr_image)
{
  last_ts_ = time;
  
  char info[256];
  const double delta_time = time - prev_pub_ts_;
  const bool   publish    = (delta_time >= param_.track_frequency);

  // track point process
  vector<cv::Point2f> curr_points;
  if (!last_points_.empty()) {
    std::vector<uchar> status;
    std::vector<float> errors;
    cv::calcOpticalFlowPyrLK(last_image_, curr_image, last_points_, curr_points, status, errors);
    reduceVector<int>(points_id_, status);
    reduceVector<int>(track_cnt_, status);
    reduceVector<cv::Point2f>(curr_points,  status);
    reduceVector<cv::Point2f>(prev_points_, status);
    
    status = checkOutOfBorder(curr_points);
    reduceVector<int>(points_id_, status);
    reduceVector<int>(track_cnt_, status);
    reduceVector<cv::Point2f>(curr_points,  status);
    reduceVector<cv::Point2f>(prev_points_, status);
  }

  for (int& track_n : track_cnt_) {
    ++track_n;
  }

  last_points_ = std::move(curr_points);
  last_image_  = curr_image.clone();

  if (publish) {
    sprintf(info, "[TRACKER] Publish data in %lf", time);
    LOG_IF(INFO, param_.verbose) << info;

    prev_pub_ts_ = time;

    int old_point_cnt = last_points_.size();
    vector<uchar> status = checkWithFundamental();
    reduceVector<int>(points_id_, status);
    reduceVector<int>(track_cnt_, status);
    reduceVector<cv::Point2f>(prev_points_, status);
    reduceVector<cv::Point2f>(last_points_, status);
    reduceVector<cv::Point2f>(prev_points_un_, status);
    reduceVector<cv::Point2f>(last_points_un_, status);

    sprintf(info, "[TRACKER] Tracking %d -> %zu point.", old_point_cnt, last_points_.size());
    LOG_IF(INFO, param_.verbose) << info;

    velocity_ = computeVelocity(false);

    if (param_.max_cornor_num - (int)last_points_.size() >= 9) {
      COMMON::TicToc tick;
      cv::Mat mask = setMask();
      LOG_IF(INFO, param_.verbose) << "[TRACKER] Make mask takes " << tick.toc() << "s.";

      vector<cv::Point2f> add_points;
      const int add_points_cnt = param_.max_cornor_num - (int)last_points_.size();
      const int add_points_each_roi = floor(add_points_cnt/(int)rois_.size());
      for (int i = 0; i < rois_.size(); ++i) {
        int cnt = (i != (int)rois_.size()-1) ? add_points_each_roi : (add_points_cnt - int(rois_.size()-1)*add_points_each_roi);
        vector<cv::Point2f> add_points_vec_each_roi;
        cv::goodFeaturesToTrack(last_image_(rois_[i]), add_points_vec_each_roi, cnt, 0.01, param_.min_cornor_gap, mask(rois_[i]));

        cv::Point2f shift = rois_[i].tl();
        for (cv::Point2f& p : add_points_vec_each_roi) {
          add_points.push_back(cv::Point2f(shift.x + p.x, shift.y + p.y));
        }
      }

      for (size_t i = 0; i < add_points.size(); ++i) {
        points_id_.push_back(ID++);
        track_cnt_.push_back(1);
        last_points_.push_back(add_points[i]);

        cv::Point3f pc = camera_->imageToSpace(add_points[i]);
        last_points_un_.push_back(cv::Point2f(pc.x/pc.z, pc.y/pc.z));
      }

      sprintf(info, "[TRACKER] Add %d new point.", (int)add_points.size());
      LOG_IF(INFO, param_.verbose) << info;
    }

    prev_points_    = last_points_;
    prev_points_un_ = last_points_un_;
    prev_image_     = last_image_.clone();

    // testUndistortPoint();
    // testTrackConsistency();
  }

  // ----------- line feature part ---------------------
#if 0
  if (publish) {
    // 1. extract line features
    vector<cv::line_descriptor::KeyLine> lines;
    cv::Mat desc;
    extractLineFeature(curr_image, cv::Mat(), lines, desc);

    // 2. match
    vector<cv::DMatch> match_pairs;
    matchLineFeature(prev_line_features_desc_, desc, match_pairs);

    // 3. RANSAC to refuse outliers
    vector<LINE_ENDS> ref_lines;
    vector<LINE_ENDS> cur_lines;
    fetchMatchLines(prev_line_features_, lines, match_pairs, ref_lines, cur_lines);
    outlierCheck(ref_lines, cur_lines);

    // 4. add new line to prev line feature

    // 5. 
    if (param_.max_line_num - static_cast<int>(prev_line_features_.size()) > 5) {
      // 1. make line feature mask
      cv::Mat line_mask = setLineMask();

      // 2. extract line features
      vector<cv::line_descriptor::KeyLine> lines;
      cv::Mat desc;
      extractLineFeature(curr_image, line_mask, lines, desc);

      // 3. sort by line's length, and change to LINE_ENDS
      sort(lines.begin(), lines.end(), [](const auto& line1, const auto& line2) {
        return line1.lineLength > line2.lineLength;
      });
      
      
    }

    // TODO: 4. match line features

    // TODO: 5. add new line features

    // TODO: 6. check track loss
  }
  #endif

  return publish;
}

// int ImageTracker::outlierCheck(const std::vector<LINE_ENDS>& ref_lines, std::vector<LINE_ENDS>& cur_lines)
// {
//   do {

//   } while ();
// }

int ImageTracker::fetchMatchLines(
      vector<cv::line_descriptor::KeyLine>& train_lines, vector<cv::line_descriptor::KeyLine>& query_lines, vector<cv::DMatch>& matches, 
      vector<LINE_ENDS>& ref_lines, vector<LINE_ENDS>& cur_lines)
{
  ref_lines.clear();
  cur_lines.clear();
  ref_lines.reserve(matches.size());
  cur_lines.reserve(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    cv::DMatch dmatch = matches[i];
    cv::line_descriptor::KeyLine train_line = train_lines[dmatch.trainIdx];
    cv::line_descriptor::KeyLine query_line = query_lines[dmatch.queryIdx];
    LINE_ENDS line1 = std::make_pair(
        Eigen::Vector3d{train_line.startPointX, train_line.startPointY, 1.}, 
        Eigen::Vector3d{train_line.endPointX,   train_line.endPointY,   1.});
    
    LINE_ENDS line2 = std::make_pair(
        Eigen::Vector3d{query_line.startPointX, query_line.startPointY, 1.}, 
        Eigen::Vector3d{query_line.endPointX,   query_line.endPointY,   1.});
    
    ref_lines.emplace_back(line1);
    cur_lines.emplace_back(line2);
  }

  return static_cast<int>(matches.size());
}

int ImageTracker::extractLineFeature(const cv::Mat& image, const cv::Mat& mask, std::vector<cv::line_descriptor::KeyLine>& lines, cv::Mat& descrips)
{
  cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
  cv::Ptr<cv::line_descriptor::LSDDetector>      lsd = cv::line_descriptor::LSDDetector::createLSDDetector();

  lsd->detect(image, lines, 1.2, 1, mask);
  std::vector<uchar> status(lines.size(), 1);
  for (size_t i = 0; i < lines.size(); ++i) {
    const cv::line_descriptor::KeyLine& line = lines[i];
    cv::Point2f start_pt = line.getStartPoint();
    cv::Point2f end_pt = line.getEndPoint();
    if (cv::norm(end_pt - start_pt) < 5) {
      status[i] = 0;
    }
  }

  reduceVector<cv::line_descriptor::KeyLine>(lines, status);
  lbd->compute(image, lines, descrips);

  return static_cast<int>(lines.size());
}

int ImageTracker::matchLineFeature(const cv::Mat& ref_descrip, const cv::Mat& dst_descrip, std::vector<cv::DMatch>& matches)
{
  std::vector<std::vector<cv::DMatch>> lmatches;
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  line_matcher_->knnMatch(ref_descrip, dst_descrip, lmatches, 2);

  for(size_t i = 0; i < lmatches.size(); ++i) {
    const cv::DMatch& bestMatch   = lmatches[i][0];
    const cv::DMatch& betterMatch = lmatches[i][1];
    float distanceRatio = bestMatch.distance / betterMatch.distance;
    if (distanceRatio < param_.pass_thresh_) {
      matches.push_back(bestMatch);
    }
  }

  return static_cast<int>(matches.size());
}

// cv::Mat ImageTracker::setLineMask()
// {
//   cv::Mat mask(camera_->height(), camera_->width(), CV_8UC1, 255);

//   struct TempData {
//     int id, heat;
//     double ts;
//     LINE_ENDS ends;
//     cv::Mat desc;
//   };

//   vector<TempData> cache(prev_line_features_.size());
//   for (size_t i = 0; i < last_points_.size(); ++i) {
//     cache[i] = TempData{line_feature_id_[i], line_track_cnt_[i], line_feature_ts_[i],
//         prev_line_features_[i], prev_line_features_desc_.row(i).clone()};
//   }

//   sort(cache.begin(), cache.end(), [](const TempData& a, const TempData& b) {
//     return a.heat > b.heat;
//   });

//   line_track_cnt_.clear();
//   line_feature_ts_.clear();
//   line_feature_id_.clear();
//   prev_line_features_.clear();
//   line_matcher_->clear();
//   // prev_line_features_desc_ = cv::Mat();

//   vector<uchar> status(cache.size(), 1);
//   for (size_t i = 0; i < cache.size(); ++i) {
//     const TempData& data = cache[i];
//     cv::Point2f p1(data.ends.first(0), data.ends.first(1));
//     cv::Point2f p2(data.ends.second(0), data.ends.second(1));
//     if (mask.at<uchar>(p1) == 255 || mask.at<uchar>(p2) == 255) {
//       cv::line(mask, p1, p2, 0, param_.min_line_cornor_gap);
//       line_feature_ts_.emplace_back(data.ts);
//       line_feature_id_.emplace_back(data.id);
//       line_track_cnt_.emplace_back(data.heat);
//       prev_line_features_.emplace_back(data.ends);
//       line_matcher_->add(data.desc);
//     }
//   }

//   return mask;
// }

TrackResult ImageTracker::fetchResult()
{
  return TrackResult{
    last_ts_,
    points_id_, 
    last_points_,
    last_points_un_,
    velocity_
  };
}

vector<uchar> ImageTracker::checkWithFundamental()
{
  CHECK_EQ(last_points_.size(), prev_points_.size());
  
  last_points_un_.resize(last_points_.size());
  prev_points_un_.resize(prev_points_.size());
  
  for (size_t i = 0; i < last_points_un_.size(); ++i) {
    cv::Point3f pc;
    pc = camera_->imageToSpace(last_points_[i]);
    last_points_un_[i] = cv::Point2f(pc.x/pc.z, pc.y/pc.z);

    pc = camera_->imageToSpace(prev_points_[i]);
    prev_points_un_[i] =  cv::Point2f(pc.x/pc.z, pc.y/pc.z);
  }
  
  if (last_points_.size() > 8) {
    std::vector<uchar> status;
    cv::findFundamentalMat(prev_points_un_, last_points_un_, cv::FM_RANSAC, param_.fm_threshold/camera_->fx(), 0.99, status);
    return status;
  }
  else {
    return std::vector<uchar>(last_points_.size(), 1);
  }
}

std::vector<uchar> ImageTracker::checkOutOfBorder(const std::vector<cv::Point2f> &pts)
{
  std::vector<uchar> status(pts.size(), 1);
  const int width  = camera_->width();
  const int height = camera_->height();
  for (size_t i = 0; i < pts.size(); ++i) {
    const cv::Point2f &pt = pts[i];
    if (0 <= pt.x && pt.x < width && 0 <= pt.y && pt.y < height) {
      continue;
    }
    status[i] = 0;
  }
  return status;
}

cv::Mat ImageTracker::setMask()
{
  cv::Mat mask(camera_->height(), camera_->width(), CV_8UC1, 255);

  struct TempData {
    int id, heat;
    cv::Point2f point;
    cv::Point2f point_un;
  };

  vector<TempData> cache(last_points_.size());
  for (size_t i = 0; i < last_points_.size(); ++i) {
    cache[i] = TempData{points_id_[i], track_cnt_[i], last_points_[i], last_points_un_[i]};
  }

  sort(cache.begin(), cache.end(), [](const TempData& a, const TempData& b) {
    return a.heat > b.heat;
  });

  points_id_.clear();
  track_cnt_.clear();
  last_points_.clear();
  last_points_un_.clear();

  vector<uchar> status(cache.size(), 1);
  for (size_t i = 0; i < cache.size(); ++i) {
    const TempData& data = cache[i];
    if (mask.at<uchar>(data.point) == 255) {
      points_id_.push_back(data.id);
      track_cnt_.push_back(data.heat);
      last_points_.push_back(data.point);
      last_points_un_.push_back(data.point_un);
      cv::circle(mask, data.point, param_.min_cornor_gap, 0, -1);
    }
  }

  return mask;
}

cv::Point2f ImageTracker::computeVelocity(bool show_match)
{
  CHECK_EQ(last_points_.size(),    prev_points_.size());
  CHECK_EQ(last_points_un_.size(), prev_points_un_.size());

  if (show_match && !prev_image_.empty()) {
    cv::Mat merge;
    cv::hconcat(prev_image_, last_image_, merge);
    cv::cvtColor(merge, merge, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < last_points_.size(); ++i) {
      const cv::Point2f prev_point = prev_points_[i];
      const cv::Point2f last_point = last_points_[i] + cv::Point2f(camera_->width(), 0);

      cv::Scalar color;
      cv::theRNG().fill(color, CV_8U, 0, 255);

      cv::circle(merge, prev_point, 2, color, 1);
      cv::circle(merge, last_point, 2, color, 1);
      cv::line(merge, prev_point, last_point, color, 1);
    }

    cv::imshow("[show track]", merge);
    cv::waitKey();
  }

  cv::Point3f move(0, 0, 1.e-4f);
  for (size_t i = 0; i < last_points_un_.size(); ++i) {
    cv::Point2f delta_in_mm = last_points_un_[i] - prev_points_un_[i];
    move += cv::Point3f(delta_in_mm.x, delta_in_mm.y, 1);
  }

  return cv::Point2f(move.x/move.z, move.y/move.z);
}

void ImageTracker::testUndistortPoint()
{
  // test un point is or not correct.(pass)
  cv::Mat new_K;
  cv::Mat undistort_image = camera_->undistortImage(last_image_, &new_K);
  
  cv::Mat show;
  cv::hconcat(last_image_, undistort_image, show);
  cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);

  cv::Point2f stride(camera_->width(), 0);
  for (size_t i = 0; i < last_points_.size(); ++i) {

    cv::Point3f pc(last_points_un_[i].x, last_points_un_[i].y, 1.0);
    cv::Point2f pi = projectToImage(pc, new_K) + stride;

    cv::Scalar color;
    cv::theRNG().fill(color, CV_8U, 0, 255);
    cv::circle(show, last_points_[i], 2, color, 1);
    cv::circle(show, pi,              2, color, 1);
    cv::line(show, last_points_[i], pi, color, 1);
  }

  cv::imshow("[test undistort]", show);
  cv::waitKey();
}

void ImageTracker::testTrackConsistency()
{
  static cv::Mat record_image;

  CHECK_EQ(points_id_.size(), track_cnt_.size());
  CHECK_EQ(points_id_.size(), last_points_.size());

  const double font_size = 0.7;
  if (record_image.empty()) {
    cv::Mat show = last_image_.clone();
    cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);
    
    for (size_t i = 0; i < points_id_.size(); ++i) {
      cv::Scalar color;
      cv::theRNG().fill(color, CV_8U, 0, 255);

      string text = "("+to_string(points_id_[i])+", "+to_string(track_cnt_[i])+")"; 

      cv::circle(show, last_points_[i], 2, color, -1);
      cv::putText(show, text, last_points_[i], cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, color);
    }

    record_image = show.clone();
  } 
  else {
    cv::Mat color_last_image;
    cv::cvtColor(last_image_, color_last_image, cv::COLOR_GRAY2BGR);
    
    for (size_t i = 0; i < points_id_.size(); ++i) {
      cv::Scalar color;
      cv::theRNG().fill(color, CV_8U, 0, 255);

      string text = "("+to_string(points_id_[i])+", "+to_string(track_cnt_[i])+")"; 

      cv::circle(color_last_image, last_points_[i], 2, color, -1);
      cv::putText(color_last_image, text, last_points_[i], cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, color);
    }

    // TODO: show the consistency.
    cv::Mat show;
    cv::hconcat(record_image, color_last_image, show);

    cv::imshow("[consist]", show);
    cv::waitKey();

    record_image = color_last_image.clone();
  }  
}

} // namespace MSCKF
