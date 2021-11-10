#pragma once

#include "iostream"
#include "string"
#include "memory"
#include "opencv2/opencv.hpp"

using namespace std;

struct CameraParam {
  std::string type;
  int width, height;
  float fx, fy, cx, cy;
  float k1, k2, d1, d2;
};

struct InitParam {
  bool verbose;
  int imu_cnt;
  double imu_accl_cov;
  double imu_gyro_cov;
};

struct MsckfParam {
  bool verbose;
  // filter param
  double noise_accl;
  double noise_gyro;
  double noise_accl_bias;
  double noise_gyro_bias;
  double noise_observation;

  // Sliding window param
  int sliding_window_lens;
  double angle_threshold;
  double distance_threshold;
  double track_rate_threshold;
};

struct SFMParam {
  bool verbose;
  int max_iter_cnt;
  int max_try_cnt;
  double converge_threshold;
  double min_disparity_angle;
  double min_disparity_distance;
};

struct TrackParam {
  bool verbose;
  int max_cornor_num;
  int min_cornor_gap;
  double fm_threshold;
  double track_frequency;
};

class Config{
public:
  static void readConfig(std::string file_path);

private:
  Config() = default;

public:
  static cv::Mat Tbc;
  static CameraParam camera_param;
  static InitParam   init_param;
  static MsckfParam  msckf_param;
  static SFMParam    sfm_param;
  static TrackParam  track_param; 
};
