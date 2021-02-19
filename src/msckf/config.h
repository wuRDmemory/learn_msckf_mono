#pragma once

#include "iostream"
#include "string"
#include "memory"

#include "opencv2/opencv.hpp"

using namespace std;

class Config{
private:
  static Config* config_ptr_;

private:
  Config(string config_file_path);

public:
  struct FeatureConfig {
    int max_iter_cnt;
    int max_try_cnt;
    double converge_threshold;
    double min_disparity_angle;
    double min_disparity_distance;
  };

public:
  //@ camera
  static float fx, fy, cx, cy;
  static float d0, d1, d2, d3;
  static int   width, height;

  //@ tracking
  static bool  track_verbose;
  static int   max_cornor_num, min_cornor_gap;
  static float fm_threshold;
  static float track_frequency;

  //@ inertial
  static int   inertial_init_cnt;
  static float inertial_init_accl_cov;
  static float inertial_init_gyro_cov;

  //@ noise
  static float noise_accl, noise_accl_bias;
  static float noise_gyro, noise_gyro_bias;
  static float noise_observation;

  //@ sliding window
  static int   sliding_window_lens;
  static float distance_threshold;
  static float angle_threshold;

  //@ extrinsic matrix
  static cv::Mat Rbc;
  static cv::Mat tbc;

  //@ feature configuration
  static FeatureConfig feature_config;

public:
  static Config* getInstance(const char* file_path = NULL);
};
