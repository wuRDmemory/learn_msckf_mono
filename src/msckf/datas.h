#pragma once

#include "common.h"
#include "glog/logging.h"

#include "absl/synchronization/mutex.h"
#include "absl/memory/memory.h"

using namespace std;

namespace MSCKF {

using LINE_ENDS    = std::pair<Eigen::Vector3d, Eigen::Vector3d>;
using IMU_Matrix   = Eigen::Matrix<double, IMU_STATUS_DIM, IMU_STATUS_DIM>;
using Drive_Matrix = Eigen::Matrix<double, IMU_STATUS_DIM, IMU_NOISE_DIM>;
using Noise_Matrix = Eigen::Matrix<double, IMU_NOISE_DIM,  IMU_NOISE_DIM>;

enum FeatureStatus
{
  NotInit = 1,
  Inited  = 2,
  WaitForDelete = 3,
  Deleted = 4,
};

struct UpdateFeatureIds
{
  std::set<int> lose_ftr_id;
  std::set<int> prune_ftr_id;
  std::vector<int> remove_cam_id;
};

struct ImuData
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double          ts;
  Eigen::Vector3f accl;
  Eigen::Vector3f gyro;
};

struct CameraData
{
  double  ts;
  cv::Mat image;
};

struct TrackResult
{
  double              ts;
  vector<int>         point_id;
  vector<cv::Point2f> point_uv;
  vector<cv::Point2f> point_f;

  // line part
  vector<int>       line_id;
  vector<LINE_ENDS> line_ends;
  vector<bool>      tracking;
};

struct ImuStatus
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int    id;
  double ts;

  Eigen::Quaterniond Rwb;
  Eigen::Vector3d    bg;
  Eigen::Vector3d    vwb;
  Eigen::Vector3d    ba;
  Eigen::Vector3d    pwb;
  Eigen::Vector3d    g;

  Eigen::Quaterniond Rbc;
  Eigen::Vector3d    pbc;

  Eigen::Quaterniond Rwb_nullspace;
  Eigen::Vector3d    vwb_nullspace;
  Eigen::Vector3d    pwb_nullspace;

  std::set<int> observes;

  void boxPlus(const Eigen::VectorXd& delta_x);
  Eigen::VectorXd boxMinus(const ImuStatus& state) const;
  Eigen::MatrixXd S2Bx() const;

  friend std::ostream& operator<<(std::ostream& os, const ImuStatus& imu);
};

struct CameraStatus
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double ts;

  Eigen::Quaterniond Rwc;
  Eigen::Vector3d    pwc;

  Eigen::Quaterniond Rwc_nullspace;
  Eigen::Vector3d    pwc_nullspace;

  void boxPlus(const Eigen::VectorXd& delta_x);
  Eigen::VectorXd boxMinus(const CameraStatus& state) const;

  friend std::ostream& operator<<(std::ostream& os, const CameraStatus& cam);
};

struct Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  map<int, Eigen::Vector3d> observes;
  Eigen::Vector3d point_3d;
  Eigen::Vector3d point_3d_init;
  FeatureStatus status;
};

// line part
struct LineFeature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  map<int, LINE_ENDS> observes;
  Eigen::Vector3d direct_vec;
  Eigen::Vector3d normal_vec;
  FeatureStatus status;
};

using CameraWindow   = map<int, CameraStatus>;
using FeatureManager = map<int, Feature>;
using LineFeatureManager = map<int, LineFeature>;

struct Data
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuStatus      imu_status;
  CameraWindow   cameras_;
  FeatureManager features_;

  // Eigen::Vector3d gravity;

  Eigen::MatrixXd P_dx;
  Noise_Matrix    Q_imu;
  IMU_Matrix      Phi_test; // this only for test.
};

} // namespace MSCKF

