#include "msckf/config.h"
#include "glog/logging.h"

CameraParam Config::camera_param = {
  .type = "None",
  .width = 640,
  .height = 480,
  .fx = 0,
  .fy = 0,
  .cx = 0,
  .cy = 0,
  .k1 = 0,
  .k2 = 0,
  .d1 = 0,
  .d2 = 0,
  .scaler = 1
};

InitParam Config::init_param = {
  .verbose = false,
  .imu_cnt = 200,
  .imu_dtime = 2,
  .imu_accl_cov = 0.5,
  .imu_gyro_cov = 0.1,
};

MsckfParam Config::msckf_param = {
  .verbose = false,
  .noise_accl = 0.1,
  .noise_gyro = 0.1,
  .noise_accl_bias = 0.1,
  .noise_gyro_bias = 0.1,
  .noise_observation = 0.1,
  .large_dp = 1.0,
  .large_dv = 0.5,
  .ikf_iters = 10,
  .sliding_window_lens = 10,
  .angle_threshold = 15, // in degree
  .distance_threshold = 0.5, // in meter
  .track_rate_threshold = 0.5,
};

SFMParam Config::sfm_param = {
  .verbose = false,
  .finetune = true,
  .max_iter_cnt = 10,
  .max_try_cnt = 10,
  .max_cond_number = 20000,
  .min_dist = 0,
  .max_dist = 50,
  .converge_threshold = 0.1,
  .min_disparity_angle = 0.1,
  .min_disparity_distance = 0.1
};

TrackParam Config::track_param = {
  .verbose = false,
  .max_cornor_num = 200,
  .min_cornor_gap = 30,
  .fm_threshold = 1.0,
  .track_frequency = 10,
};

cv::Mat Config::Tbc;

void Config::readConfig(string config_file_path)
{
  cv::FileStorage fs(config_file_path, cv::FileStorage::READ);
  assert(fs.isOpened());

  { //* read camera paramters
    cv::FileNode n = fs["camera"];
    n["type"] >> camera_param.type;
    camera_param.scaler = (int)n["scaler"];
    camera_param.width  = (int)n["width"] / camera_param.scaler;
    camera_param.height = (int)n["height"] / camera_param.scaler;

    camera_param.fx = (float)n["fx"] / camera_param.scaler;
    camera_param.fy = (float)n["fy"] / camera_param.scaler;
    camera_param.cx = (float)n["cx"] / camera_param.scaler;
    camera_param.cy = (float)n["cy"] / camera_param.scaler;
    
    camera_param.k1 = (float)n["k1"];
    camera_param.k2 = (float)n["k2"];
    camera_param.d1 = (float)n["d1"];
    camera_param.d2 = (float)n["d2"];

    LOG(INFO) << "[CONFIG] Camera type   : \t" << camera_param.type;
    LOG(INFO) << "[CONFIG] Camera width  : \t" << camera_param.width;
    LOG(INFO) << "[CONFIG] Camera height : \t" << camera_param.height;

    LOG(INFO) << std::fixed << "[CONFIG] Camera fx : \t" << camera_param.fx;
    LOG(INFO) << std::fixed << "[CONFIG] Camera fy : \t" << camera_param.fy;
    LOG(INFO) << std::fixed << "[CONFIG] Camera cx : \t" << camera_param.cx;
    LOG(INFO) << std::fixed << "[CONFIG] Camera cy : \t" << camera_param.cy;

    LOG(INFO) << std::fixed << "[CONFIG] Camera k1 : \t" << camera_param.k1;
    LOG(INFO) << std::fixed << "[CONFIG] Camera k2 : \t" << camera_param.k2;
    LOG(INFO) << std::fixed << "[CONFIG] Camera d1 : \t" << camera_param.d1;
    LOG(INFO) << std::fixed << "[CONFIG] Camera d2 : \t" << camera_param.d2;
  }

  { //* read Tbc
    if (!fs["Tbc"].empty()) {
      // LOGI("[PARAM] We get Tbc!!!");
      fs["Tbc"] >> Tbc;
    } else if (!fs["Tcb"].empty()) {
      // LOGI("[PARAM] We get Tcb!!!");
      fs["Tcb"] >> Tbc;

      cv::Mat R, t;
      Tbc(cv::Range(0, 3), cv::Range(0, 3)).copyTo(R);
      Tbc.col(3).rowRange(0, 3).copyTo(t);
      Tbc(cv::Range(0, 3), cv::Range(0, 3)) = R.t();
      Tbc.col(3).rowRange(0, 3) = -R.t() * t;
    } else {
      assert(false);
    }
  }

  { //* read track parameter
    cv::FileNode n = fs["track"];
    track_param.verbose = (int)n["verbose"] != 0;
    track_param.fm_threshold    = (double)n["fm_threshold"];
    track_param.track_frequency = 1.0/(double)n["track_frequency"];
    track_param.max_cornor_num  = (double)n["max_cornor_num"];
    track_param.min_cornor_gap  = (double)n["min_cornor_gap"];

    LOG(INFO) << std::fixed << "[CONFIG] Tracker fm_threshold : \t" << track_param.fm_threshold;
    LOG(INFO) << std::fixed << "[CONFIG] Tracker frequency    : \t" << track_param.track_frequency;
    LOG(INFO) << std::fixed << "[CONFIG] Tracker max_cornor_num : \t" << track_param.max_cornor_num;
    LOG(INFO) << std::fixed << "[CONFIG] Tracker min_cornor_gap : \t" << track_param.min_cornor_gap;
  }

  { //* read init parameter
    cv::FileNode n = fs["initial"];
    init_param.verbose = (int)n["verbose"] != 0;
    init_param.imu_cnt = (int)n["imu_cnt"];
    init_param.imu_dtime = (double)n["imu_dtime"];
    init_param.imu_accl_cov = (double)n["imu_accl_cov"];
    init_param.imu_gyro_cov = (double)n["imu_gyro_cov"];

    LOG(INFO) << std::fixed << "[CONFIG] Initial imu count : \t" << init_param.imu_cnt;
    LOG(INFO) << std::fixed << "[CONFIG] Initial imu dtime : \t" << init_param.imu_dtime;
    LOG(INFO) << std::fixed << "[CONFIG] Initial imu accl cov : \t" << init_param.imu_accl_cov;
    LOG(INFO) << std::fixed << "[CONFIG] Initial imu gyro cov : \t" << init_param.imu_gyro_cov;
  }

  { //* read msck paramter
    cv::FileNode n = fs["msckf"];
    msckf_param.verbose = (int)n["verbose"] != 0;
    msckf_param.noise_accl = (double)n["noise_accl"];
    msckf_param.noise_gyro = (double)n["noise_gyro"];
    msckf_param.noise_accl_bias = (double)n["noise_accl_bias"];
    msckf_param.noise_gyro_bias = (double)n["noise_gyro_bias"];
    msckf_param.noise_observation = (double)n["noise_observation"];

    msckf_param.large_dp = (double)n["large_dp"];
    msckf_param.large_dv = (double)n["large_dv"];

    msckf_param.ikf_iters = (int)n["ikf_iters"];
    msckf_param.sliding_window_lens = (int)n["sliding_window_lens"];
    msckf_param.distance_threshold  = (double)n["distance_threshold"];
    msckf_param.angle_threshold     = (double)n["angle_threshold"]*M_PI/180.0;
    msckf_param.track_rate_threshold = (double)n["track_rate_threshold"];

    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf imu noise accl : \t" << msckf_param.noise_accl; 
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf imu noise gyro : \t" << msckf_param.noise_gyro;
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf imu noise accl bias : \t" << msckf_param.noise_accl_bias; 
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf imu noise gyro bias : \t" << msckf_param.noise_gyro_bias;
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf observation noise   : \t" << msckf_param.noise_observation;
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf large dp   : \t" << msckf_param.large_dp;
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf large dv   : \t" << msckf_param.large_dv;
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf ikf iterates        : \t" << msckf_param.ikf_iters;
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf sliding window lens : \t" << msckf_param.sliding_window_lens;
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf distance threshold  : \t" << msckf_param.distance_threshold;
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf angular  threshold  : \t" << msckf_param.angle_threshold*180.0/M_PI;
    LOG(INFO) << std::fixed << std::setprecision(6) << "[CONFIG] Msckf track rate threshold  : \t" << msckf_param.track_rate_threshold;
  }
 
  { //* read SFM feature configuration
    cv::FileNode n = fs["sfm"];
    sfm_param.verbose = (int)n["verbose"] == 1;
    sfm_param.finetune = (int)n["finetune"] == 1;
    sfm_param.max_iter_cnt = (int)n["max_iter_cnt"];
    sfm_param.max_try_cnt  = (int)n["max_try_cnt"];
    sfm_param.max_cond_number = (int)n["max_cond_number"];
    sfm_param.max_dist = (double)n["max_dist"];
    sfm_param.min_dist = (double)n["min_dist"];
    sfm_param.converge_threshold = (double)n["converge_threshold"];
    sfm_param.min_disparity_angle = cos((double)n["min_disparity_angle"]*M_PI/180.0);
    sfm_param.min_disparity_distance = (double)n["min_disparity_distance"];

    LOG(INFO) << "[CONFIG] SFM verbose : \t" << sfm_param.verbose; 
    LOG(INFO) << "[CONFIG] SFM funetune : \t" << sfm_param.finetune; 
    LOG(INFO) << std::fixed << "[CONFIG] SFM max iter count : \t" << sfm_param.max_iter_cnt; 
    LOG(INFO) << std::fixed << "[CONFIG] SFM max try  count : \t" << sfm_param.max_try_cnt;
    LOG(INFO) << std::fixed << "[CONFIG] SFM max cond number : \t" << sfm_param.max_cond_number;
    LOG(INFO) << std::fixed << "[CONFIG] SFM max distance : \t" << sfm_param.max_dist;
    LOG(INFO) << std::fixed << "[CONFIG] SFM min distance : \t" << sfm_param.min_dist;
    LOG(INFO) << std::fixed << "[CONFIG] SFM converge threshold  : \t" << sfm_param.converge_threshold;
    LOG(INFO) << std::fixed << "[CONFIG] SFM min disparity angle : \t" << acos(sfm_param.min_disparity_angle)*180/M_PI;
    LOG(INFO) << std::fixed << "[CONFIG] SFM min disparity distance : \t" << sfm_param.min_disparity_distance;
  }
}

