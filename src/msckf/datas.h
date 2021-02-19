#pragma once

#include "istream"
#include "vector"
#include "deque"
#include "set"
#include "map"

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"

#include "glog/logging.h"

#include "absl/synchronization/mutex.h"
#include "absl/memory/memory.h"

#include "opencv2/opencv.hpp"
#include "Eigen/Core"
#include "Eigen/Dense"

using namespace std;

#define IMU_STATUS_NUM 5
#define IMU_NOISE_NUM  4

#define IMU_STATUS_DIM 3*IMU_STATUS_NUM
#define IMU_NOISE_DIM  3*IMU_NOISE_NUM

#define J_R  0
#define J_BG 3
#define J_V  6
#define J_BA 9
#define J_P  12

#define G_Ng   0
#define G_Nbg  3
#define G_Na   6
#define G_Nba  9

#define C_R 0
#define C_P 3

using IMU_Matrix   = Eigen::Matrix<double, IMU_STATUS_DIM, IMU_STATUS_DIM>;
using Drive_Matrix = Eigen::Matrix<double, IMU_STATUS_DIM, IMU_NOISE_DIM>;
using Noise_Matrix = Eigen::Matrix<double, IMU_NOISE_DIM,  IMU_NOISE_DIM>;

namespace MSCKF {

enum FeatureStatus{
  NotInit = 1,
  Inited  = 2,
  WaitForDelete = 3,
  Deleted = 4,
};

struct ImuData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double          ts;
  Eigen::Vector3f accl;
  Eigen::Vector3f gyro;
};

struct CameraData {
  double  ts;
  cv::Mat image;
};

struct TrackResult {
  double              ts;
  vector<int>         point_id;
  vector<cv::Point2f> point_uv;
  vector<cv::Point2f> point_f;
};

struct ImuStatus {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int    id;
  double ts;

  Eigen::Quaterniond Rwb;
  Eigen::Vector3d    bg;
  Eigen::Vector3d    vwb;
  Eigen::Vector3d    ba;
  Eigen::Vector3d    pwb;

  Eigen::Quaterniond Rbc;
  Eigen::Vector3d    pbc;

  Eigen::Quaterniond Rwb_nullspace;
  Eigen::Vector3d    vwb_nullspace;
  Eigen::Vector3d    pwb_nullspace;

  set<int>           observes;

  friend std::ostream& operator<<(std::ostream& os, const ImuStatus& imu);
};

struct CameraStatus {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double ts;

  Eigen::Quaterniond Rwc;
  Eigen::Vector3d    pwc;

  Eigen::Quaterniond Rwc_nullspace;
  Eigen::Vector3d    pwc_nullspace;

  friend std::ostream& operator<<(std::ostream& os, const CameraStatus& cam);
};

struct Feature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  map<int, Eigen::Vector3d> observes;
  Eigen::Vector3d point_3d;
  FeatureStatus status;
};

using CameraWindow   = map<int, CameraStatus>;
using FeatureManager = map<int, Feature>;

struct Data {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuStatus      imu_status;
  CameraWindow   cameras_;
  FeatureManager features_;

  Eigen::Vector3d gravity;

  Eigen::MatrixXd P_dx;
  Noise_Matrix    Q_imu;
  IMU_Matrix      Phi_test; // this only for test.
};

bool initialFeature(Feature& ftr, CameraWindow& cams);

bool checkMotion(Feature& ftr, CameraWindow& cams);

Eigen::Vector2d 
evaluate(const Eigen::Vector3d& Pj_w, const CameraStatus& T_ci, const Eigen::Vector3d& obs_ci, Eigen::Matrix<double, 2, 3>* jacobian);

} // namespace MSCKF

