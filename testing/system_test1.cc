#include "iostream"
#include "thread"
#include "opencv2/opencv.hpp"

#include "camera/camera.h"
#include "camera/pinhole_camera.h"
#include "msckf/config.h"
#include "msckf/tracker.h"
#include "msckf/msckf.h"
#include "node/time_convert.h"

#include "helpers/directory_helper.h"

using namespace std;

using ImageInformation    = TEST::DirectoryHelper::ImageInformation;
using InertialInformation = TEST::DirectoryHelper::InertialInformation;

const cv::String keys =
    "{help h usage ?  || todo help              }"
    "{@config_path    || path to config path    }"
    "{@dataset_name   || dataset name           }"
    "{@dataset_path   || path to dataset        }";

int main(int argc, char** argv) {

  cv::CommandLineParser parser(argc, argv, keys);

  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  if (   !parser.has("@config_path") 
      || !parser.has("@dataset_path")
      || !parser.has("@dataset_name")) {
    parser.printMessage();
    return 0;
  }

  string config_path   = parser.get<string>(0);
  string dataset_name  = parser.get<string>(1);
  string dataset_path  = parser.get<string>(2);
  cout << "Config  path: \t" << config_path  << endl;
  cout << "Dataset name: \t" << dataset_name << endl;
  cout << "Dataset path: \t" << dataset_path << endl;

  // config
  Config::getInstance(config_path.c_str());

  // scan directory.
  TEST::DirectoryHelper dir_walker;
  TEST::DirectoryHelper::DirectoryInformation 
  dir_info = dir_walker.process(dataset_name, dataset_path);

  // system
  MSCKF::Msckf msckf(config_path);

  int imu_idx   = 0;
  int image_idx = 0;
  long long stop_timestamp = 1403637140638319104;

  map<int, MSCKF::CameraStatus> all_node;
  while (imu_idx < dir_info.inertial_info.size()) {
    double image_ts = dir_info.images_info[image_idx].timestamps;
    while (dir_info.inertial_info[imu_idx].timestamps < image_ts) {
      const auto& imu_data = dir_info.inertial_info[imu_idx];
      Eigen::Vector3f accl(imu_data.ax, imu_data.ay, imu_data.az);
      Eigen::Vector3f gyro(imu_data.gx, imu_data.gy, imu_data.gz);
      msckf.feedImuData(imu_data.timestamps, accl, gyro);
      ++imu_idx;
    }

    cv::Mat image = cv::imread(dir_info.images_info[image_idx].image_path, 0);
    msckf.feedImage(dir_info.images_info[image_idx].timestamps, image);
    ++image_idx;

    if (dir_info.images_info[image_idx].timestamps > stop_timestamp*1.e-9) {
      LOG(INFO) << "attention";
    }
  }

  return 1;
}
