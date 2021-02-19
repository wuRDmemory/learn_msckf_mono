
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

  string config_path   = parser.get<string>(0);
  cout << "Config  path: \t" << config_path << endl;
  
  // config
  Config::getInstance(config_path.c_str());

  // real landmark's position in cam0.
  Eigen::Vector3d ldmk(4, 0, 0);
  const double norm_ldmk = ldmk.norm();

  // camera's observations.
  Eigen::Vector3d cam0_ob(cv::theRNG().gaussian(2)/Config::fx, cv::theRNG().gaussian(2)/Config::fx, 1.0);
  Eigen::Vector3d cam1_ob(cv::theRNG().gaussian(2)/Config::fx, cv::theRNG().gaussian(2)/Config::fx, 1.0);
  Eigen::Vector3d cam2_ob(cv::theRNG().gaussian(2)/Config::fx, cv::theRNG().gaussian(2)/Config::fx, 1.0);
  Eigen::Vector3d cam3_ob(cv::theRNG().gaussian(2)/Config::fx, cv::theRNG().gaussian(2)/Config::fx, 1.0);

  double step = 1.0;
  Eigen::Vector3d cam0_pos(0*step                              , 0, 0);
  Eigen::Vector3d cam1_pos(1*step + cv::theRNG().gaussian(0.01), 0, 0);
  Eigen::Vector3d cam2_pos(2*step + cv::theRNG().gaussian(0.01), 0, 0);
  Eigen::Vector3d cam3_pos(3*step + cv::theRNG().gaussian(0.01), 0, 0);

  Eigen::Quaterniond cam0_qwc = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond cam1_qwc(Eigen::AngleAxisd(-(M_PI/2 - atan2(norm_ldmk, 1*step)), Eigen::Vector3d::UnitY()));
  Eigen::Quaterniond cam2_qwc(Eigen::AngleAxisd(-(M_PI/2 - atan2(norm_ldmk, 2*step)), Eigen::Vector3d::UnitY()));
  Eigen::Quaterniond cam3_qwc(Eigen::AngleAxisd(-(M_PI/2 - atan2(norm_ldmk, 3*step)), Eigen::Vector3d::UnitY()));

  MSCKF::CameraWindow cam_window;
  cam_window.insert(make_pair(0, MSCKF::CameraStatus{0, cam0_qwc, cam0_pos, cam0_qwc, cam0_pos}));
  cam_window.insert(make_pair(1, MSCKF::CameraStatus{1, cam1_qwc, cam1_pos, cam1_qwc, cam1_pos}));
  cam_window.insert(make_pair(2, MSCKF::CameraStatus{2, cam2_qwc, cam2_pos, cam2_qwc, cam2_pos}));
  cam_window.insert(make_pair(3, MSCKF::CameraStatus{3, cam3_qwc, cam3_pos, cam3_qwc, cam3_pos}));

  MSCKF::FeatureManager feature_manager;
  feature_manager.insert(make_pair(0, MSCKF::Feature{}));

  MSCKF::Feature& feature = feature_manager.at(0);
  feature.observes.insert(make_pair(0, cam0_ob));
  feature.observes.insert(make_pair(1, cam1_ob));
  feature.observes.insert(make_pair(2, cam2_ob));
  feature.observes.insert(make_pair(3, cam3_ob));

  MSCKF::checkMotion(feature, cam_window);
  MSCKF::initialFeature(feature, cam_window);

  return 1;
}
