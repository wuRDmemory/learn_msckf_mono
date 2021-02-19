#pragma once

#include "iostream"
#include "vector"
#include "queue"
#include "deque"
#include "algorithm"
#include "mutex"
#include "thread"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/QR"
#include "Eigen/SparseCore"
#include "Eigen/SPQRSupport"
#include "opencv2/opencv.hpp"
#include "opencv/cxeigen.hpp"
#include "absl/synchronization/mutex.h"
#include "boost/math/distributions/chi_squared.hpp"

#include "msckf/datas.h"
#include "msckf/tracker.h"
#include "msckf/math_utils.h"

using namespace std;

namespace MSCKF {

class Msckf {
// private:
public:
  mutable mutex imu_mutex_;
  mutable mutex cam_mutex_;
  mutable mutex mutex_;

  bool is_stop_;
  bool is_initial_;
  bool is_first_image_;

  string config_path_;
  double track_rate_;
  map<int, double> chi_square_distribution_;

  ImuData     last_imu_;
  TrackResult last_track_;

  deque<ImuData>     imu_buffer_   GUARDED_BY(imu_mutex_);
  deque<CameraData>  cam_buffer_   GUARDED_BY(cam_mutex_);
  deque<TrackResult> track_buffer_ GUARDED_BY(mutex_);

  unique_ptr<ImageTracker> image_tracker_;

  thread image_process_thread_;
  thread main_loop_thread_;

public:
  Data data_ GUARDED_BY(mutex_);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  Msckf(string config_file);
  ~Msckf();

  Msckf(const Msckf&) = delete;
  Msckf& operator=(const Msckf&) = delete;

  int feedImuData(double time, const Eigen::Vector3f& accl, const Eigen::Vector3f& gyro)
      LOCKS_EXCLUDED(imu_mutex_);

  int feedImage(double time, const cv::Mat& image)
      LOCKS_EXCLUDED(cam_mutex_);

  FeatureManager clearLoseFeature()
      LOCKS_EXCLUDED(mutex_);

// private:
  void imageProcess() 
      LOCKS_EXCLUDED(imu_mutex_) LOCKS_EXCLUDED(cam_mutex_) LOCKS_EXCLUDED(mutex_);

//   void mainLoop()
//       LOCKS_EXCLUDED(imu_mutex_) LOCKS_EXCLUDED(cam_mutex_) LOCKS_EXCLUDED(mutex_);

  bool initialization()
      LOCKS_EXCLUDED(imu_mutex_) LOCKS_EXCLUDED(cam_mutex_) LOCKS_EXCLUDED(mutex_);

  bool setup();

	bool feedTrackResult(TrackResult& track_result) 
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool predictImuStatus(const Eigen::Vector3d& accl, const Eigen::Vector3d& gyro, double dt)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool predictCamStatus(const TrackResult& track_result) 
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool featureUpdateStatus() 
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

	bool pruneCameraStatus()
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

	bool featureJacobian(const Feature& ftr, const CameraWindow& cam_window, const set<int>& constraint_cam_id, Eigen::MatrixXd& H_fj, Eigen::VectorXd& e_fj) 
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

	bool measurementUpdateStatus(const Eigen::MatrixXd& H, const Eigen::VectorXd& e) 
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

	bool findRedundanceCam(CameraWindow& cameras, vector<int>& remove_cam_id) 
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

	bool gatingTest(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, const int v)
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};

} // namespace MSCKF
