#include "iostream"
#include "thread"
#include "opencv2/opencv.hpp"

#include "camera/camera.h"
#include "camera/pinhole_camera.h"
#include "msckf/config.h"
#include "msckf/tracker.h"
#include "msckf/datas.h"
#include "msckf/msckf.h"
#include "msckf/math_utils.h"

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
  cout << "Config  path: \t" << config_path << endl;
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

  int imu_idx = 0;
  int imu_cnt = Config::inertial_init_cnt + 10;
  while (imu_idx < imu_cnt) {
    const auto& imu_data = dir_info.inertial_info[imu_idx];
    Eigen::Vector3f accl(imu_data.ax, imu_data.ay, imu_data.az);
    Eigen::Vector3f gyro(imu_data.gx, imu_data.gy, imu_data.gz);
    msckf.feedImuData(imu_data.timestamps, accl, gyro);
    ++imu_idx;
  }

  if (msckf.initialization()) {
    LOG(INFO) << "Initial success!!!";
  }
  else {
    LOG(INFO) << "Initial failed!!!";
    return 0;
  }

  MSCKF::ImuStatus old_imu_status = msckf.data_.imu_status;

  LOG(INFO) << "Before change";
  imu_cnt = 500;
  msckf.data_.Phi_test.setIdentity();
  for (int i = 0; i < 20; i++) {
    const auto& imu_data = dir_info.inertial_info[imu_cnt+i];
    Eigen::Vector3f accl(imu_data.ax, imu_data.ay, imu_data.az);
    Eigen::Vector3f gyro(imu_data.gx, imu_data.gy, imu_data.gz);
    msckf.predictImuStatus(accl.cast<double>(), gyro.cast<double>(), old_imu_status.ts + i*0.005);
  }

  LOG(INFO) << msckf.data_.imu_status;
  LOG(INFO) << "";

  LOG(INFO) << "After change";

  LOG(INFO) << "Use Phi to correction.";
  Eigen::Vector3d new_ba(0.001, -0.0045, 0.003);
  Eigen::Vector3d new_bg(0.002, 0.005, -0.001);

  Eigen::MatrixXd Phi    = msckf.data_.Phi_test;
  Eigen::Matrix3d J_R_Bg = Phi.block<3, 3>(J_R, J_BG);
  Eigen::Matrix3d J_R_Ba = Phi.block<3, 3>(J_R, J_BA);
  
  Eigen::Matrix3d J_V_Bg = Phi.block<3, 3>(J_V, J_BG);
  Eigen::Matrix3d J_V_Ba = Phi.block<3, 3>(J_V, J_BA);

  Eigen::Matrix3d J_P_Bg = Phi.block<3, 3>(J_P, J_BG);
  Eigen::Matrix3d J_P_Ba = Phi.block<3, 3>(J_P, J_BA);

  Eigen::Vector3d delta_ba = new_ba - old_imu_status.ba;
  Eigen::Vector3d delta_bg = new_bg - old_imu_status.bg;

  msckf.data_.imu_status.Rwb *= MATH_UTILS::rotateVecToQuaternion<double>(J_R_Bg*delta_bg + J_R_Ba*delta_ba);
  msckf.data_.imu_status.vwb += J_V_Bg*delta_bg + J_V_Ba*delta_ba;
  msckf.data_.imu_status.pwb += J_P_Bg*delta_bg + J_P_Ba*delta_ba;
  msckf.data_.imu_status.ba  += delta_ba;
  msckf.data_.imu_status.bg  += delta_bg;
  msckf.data_.imu_status.Rwb.normalize();
  
  LOG(INFO) << msckf.data_.imu_status;
  LOG(INFO) << "";

  LOG(INFO) << "Re-integration.";
  msckf.data_.imu_status = old_imu_status;
  msckf.data_.imu_status.bg = new_bg;
  msckf.data_.imu_status.ba = new_ba;

  msckf.data_.Phi_test.setIdentity();
  imu_cnt = 500;
  for (int i = 0; i < 20; i++) {
    const auto& imu_data = dir_info.inertial_info[imu_cnt+i];
    Eigen::Vector3f accl(imu_data.ax, imu_data.ay, imu_data.az);
    Eigen::Vector3f gyro(imu_data.gx, imu_data.gy, imu_data.gz);
    msckf.predictImuStatus(accl.cast<double>(), gyro.cast<double>(), old_imu_status.ts + i*0.005);
  }

  LOG(INFO) << msckf.data_.imu_status;

  return 1;
}
