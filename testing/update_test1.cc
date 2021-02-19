
#include "iostream"
#include "thread"
#include "opencv2/opencv.hpp"

#include "camera/camera.h"
#include "camera/pinhole_camera.h"
#include "msckf/config.h"
#include "msckf/tracker.h"
#include "msckf/msckf.h"
#include "msckf/datas.h"
#include "node/time_convert.h"

#include "helpers/directory_helper.h"

using namespace std;

using ImageInformation    = TEST::DirectoryHelper::ImageInformation;
using InertialInformation = TEST::DirectoryHelper::InertialInformation;

const cv::String keys =
    "{help h usage ?  || todo help              }"
    "{@config_path    || path to config path    }";

int main(int argc, char** argv) {

  cv::CommandLineParser parser(argc, argv, keys);

  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  if (!parser.has("@config_path")) {
    parser.printMessage();
    return 0;
  }

  string config_path = parser.get<string>(0);
  cout << "Config  path: \t" << config_path << endl;
  
  // config
  Config::getInstance(config_path.c_str());

  // msckf
  MSCKF::Msckf msckf(config_path);

  // real landmark's position in cam0.
  const int    ldmk_id   = 0;
  const double norm_ldmk = 2.0;
  const double step      = 0.2;
  const double gyro_noise = 0.0017;
  const double accl_noise = 0.02;
  const Eigen::Vector3d ldmk(norm_ldmk, 0, 0);
  const Eigen::Vector3d true_bg(-0.0025, 0.021, 0.0768);
  const Eigen::Vector3d true_ba(-0.0299, 0.125, 0.0571);
  
  // camera's observations.
  Eigen::Vector3d cam0_ob(cv::theRNG().gaussian(1)/Config::fx, cv::theRNG().gaussian(1)/Config::fx, 1.0);
  Eigen::Vector3d cam1_ob(cv::theRNG().gaussian(1)/Config::fx, cv::theRNG().gaussian(1)/Config::fx, 1.0);
  Eigen::Vector3d cam2_ob(cv::theRNG().gaussian(1)/Config::fx, cv::theRNG().gaussian(1)/Config::fx, 1.0);
  Eigen::Vector3d cam3_ob(cv::theRNG().gaussian(1)/Config::fx, cv::theRNG().gaussian(1)/Config::fx, 1.0);

  // camera's angle. in z axis
  const double angle_cam0 = 0;
  const double angle_cam1 = M_PI_2 - atan2(norm_ldmk, 1*step) + cv::theRNG().gaussian(0.1/57.3);
  const double angle_cam2 = M_PI_2 - atan2(norm_ldmk, 2*step) + cv::theRNG().gaussian(0.1/57.3);
  const double angle_cam3 = M_PI_2 - atan2(norm_ldmk, 3*step) + cv::theRNG().gaussian(0.1/57.3);
  const double angle_cam4 = M_PI_2 - atan2(norm_ldmk, 4*step) + cv::theRNG().gaussian(0.1/57.3);

  // fake initialize 
  {
    msckf.is_initial_ = true;
    MSCKF::ImuStatus& imu_status = msckf.data_.imu_status;
    imu_status.pwb.setZero();
    imu_status.Rwb.setIdentity();
    imu_status.vwb = Eigen::Vector3d(cv::theRNG().gaussian(0.01), -step/0.1, cv::theRNG().gaussian(0.01));
    imu_status.ba.setZero();
    imu_status.bg.setZero();
  }

  const vector<double>          cam_angles  = {angle_cam0, angle_cam1, angle_cam2, angle_cam3, angle_cam4};
  const vector<Eigen::Vector3d> cam_observe = {cam0_ob, cam1_ob, cam2_ob, cam3_ob};

  double ts = 0.0;
  double ts_step = 0.01; // sample in 100hz
  
  
  for (int i = 0; i < cam_angles.size(); ++i) {
    
    if (i != 0) {
      const double delta_angle = cam_angles[i] - cam_angles[i-1];
      const double mean_gyro   = delta_angle / (10*ts_step);
      for (int j = 0; j < 10; ++j) {
        Eigen::Vector3d gyro(cv::theRNG().gaussian(gyro_noise), cv::theRNG().gaussian(gyro_noise), mean_gyro + cv::theRNG().gaussian(gyro_noise));
        gyro += true_bg;

        Eigen::Vector3d accl(cv::theRNG().gaussian(accl_noise), cv::theRNG().gaussian(accl_noise), cv::theRNG().gaussian(accl_noise));
        accl += true_ba;

        msckf.feedImuData(ts, accl.cast<float>(), gyro.cast<float>());
        ts += ts_step;
      }
    }

    MSCKF::TrackResult track_result;
    track_result.ts = ts;
    if (i != cam_angles.size()-1) {
      track_result.point_id = {ldmk_id};
      track_result.point_f  = {cv::Point2f(cam_observe[i](0), cam_observe[i](1))};
      track_result.point_uv = {cv::Point2f(Config::width/2, Config::height/2)};
    }
    else {
      for (auto& id_data : msckf.data_.cameras_) {
        cout << id_data.first << ", " << id_data.second << endl;
      }

      cout << msckf.data_.imu_status << endl;

      track_result.point_id = {};
      track_result.point_f  = {};
      track_result.point_uv = {};
    }

    msckf.feedTrackResult(track_result);
  }  

  cout << "------------------------------" << endl;
  for (auto& id_data : msckf.data_.cameras_) {
    cout << id_data.first << ", " << id_data.second << endl;
  }

  cout << msckf.data_.imu_status << endl;

  return 1;
}
