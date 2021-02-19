#include "msckf/config.h"
#include "glog/logging.h"

Config* Config::config_ptr_ = NULL;

int Config::width  = 0;
int Config::height = 0;

float Config::fx = 0;
float Config::fy = 0;
float Config::cx = 0;
float Config::cy = 0;

float Config::d0 = 0;
float Config::d1 = 0;
float Config::d2 = 0;
float Config::d3 = 0;

bool Config::track_verbose = true;
int Config::min_cornor_gap = 0;
int Config::max_cornor_num = 0;
float Config::fm_threshold = 0;
float Config::track_frequency = 0;

int   Config::inertial_init_cnt = 0;
float Config::inertial_init_accl_cov = 0;
float Config::inertial_init_gyro_cov = 0;

float Config::noise_accl = 0;
float Config::noise_gyro = 0;
float Config::noise_accl_bias = 0;
float Config::noise_gyro_bias = 0;
float Config::noise_observation = 0;

int   Config::sliding_window_lens = 0;
float Config::distance_threshold  = 0;
float Config::angle_threshold     = 0;

cv::Mat Config::Rbc;
cv::Mat Config::tbc;

Config::FeatureConfig Config::feature_config = {0, 0, 0.};

Config::Config(string config_file_path) {
  cv::FileStorage fs(config_file_path, cv::FileStorage::READ);

  assert(fs.isOpened());

  { //* read camera paramters
    cv::FileNode n = fs["camera"];
    width  = (int)n["width"];
    height = (int)n["height"];

    fx = (float)n["fx"];
    fy = (float)n["fy"];
    cx = (float)n["cx"];
    cy = (float)n["cy"];
    
    d0 = (float)n["d0"];
    d1 = (float)n["d1"];
    d2 = (float)n["d2"];
    d3 = (float)n["d3"];
  }

  //@ track parameters
  track_verbose   = ((int)fs["track_verbose"] != 0);
  fm_threshold    = (float)fs["fm_threshold"];
  track_frequency = 1.0f/(float)fs["track_frequency"];
  max_cornor_num  = (float)fs["max_cornor_num"];
  min_cornor_gap  = (float)fs["min_cornor_gap"];

  //@ inertial parameters
  inertial_init_cnt = (int)fs["inertial_init_cnt"];
  inertial_init_accl_cov = (float)fs["inertial_init_accl_cov"];
  inertial_init_gyro_cov = (float)fs["inertial_init_gyro_cov"];

  noise_accl = (float)fs["noise_accl"];
  noise_gyro = (float)fs["noise_gyro"];
  noise_accl_bias = (float)fs["noise_accl_bias"];
  noise_gyro_bias = (float)fs["noise_gyro_bias"];
  noise_observation = (float)fs["noise_observation"];

  sliding_window_lens = (int)fs["sliding_window_lens"];
  distance_threshold  = (float)fs["distance_threshold"];
  angle_threshold     = (float)fs["angle_threshold"];
 
  { //@ extrinsic parameters
    cv::FileNode n = fs["Tbc"];
    n["Rbc"] >> Rbc;
    n["tbc"] >> tbc;

    LOG(INFO) << "Rbc: " << endl;
    LOG(INFO) << Rbc << endl;

    LOG(INFO) << "tbc: " << endl;
    LOG(INFO) << tbc.t() << endl;
  }

  { //@ feature configuration
    cv::FileNode n = fs["feature"];
    feature_config.max_iter_cnt = (int)n["max_iter_cnt"];
    feature_config.max_try_cnt  = (int)n["max_try_cnt"];
    feature_config.converge_threshold = (double)n["converge_threshold"];
    feature_config.min_disparity_angle = cos((double)n["min_disparity_angle"]*M_PI/180.);
    feature_config.min_disparity_distance = (double)n["min_disparity_distance"];
  }
}

Config* Config::getInstance(const char* file_path) {
  if (config_ptr_ == nullptr) {
    assert(file_path != NULL); //! here must not be null.
    config_ptr_ = new Config(file_path);
  }

  return config_ptr_;
}
