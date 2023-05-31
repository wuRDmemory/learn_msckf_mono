#pragma once

#include "common.h"
#include "config.h"
#include "datas.h"
#include "tracker.h"
#include "sfm.h"
#include "math_utils.h"
#include "absl/synchronization/mutex.h"

using namespace std;

namespace MSCKF {

using PoseCallBack = std::function<void(const MSCKF::ImuStatus&, const MSCKF::CameraWindow&, const MSCKF::FeatureManager&)>;

class Msckf {
public:

  mutable mutex imu_mutex_;
  mutable mutex cam_mutex_;
  mutable mutex mutex_;

  bool is_stop_ = true;
  bool is_initial_ = false;
  bool is_first_image_ = false;
  double track_rate_ = 0;

  std::string config_path_;
  std::map<int, double> chi_square_distribution_;

  std::deque<ImuData>     imu_buffer_   GUARDED_BY(imu_mutex_);
  std::deque<CameraData>  cam_buffer_   GUARDED_BY(cam_mutex_);
  std::deque<TrackResult> track_buffer_ GUARDED_BY(mutex_);

  std::unique_ptr<ImageTracker> image_tracker_;
  std::unique_ptr<SFM> sfm_ptr_;
  std::thread image_process_thread_;
  std::thread main_loop_thread_;

  ImuData     last_imu_;
  TrackResult last_track_;
	PoseCallBack callback_;
  Data data_ GUARDED_BY(mutex_);
	FeatureManager mature_features_ GUARDED_BY(mutex_);

private:

  MsckfParam param_;

public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  Msckf(string config_file);
  ~Msckf();

  Msckf(const Msckf&) = delete;
  Msckf& operator=(const Msckf&) = delete;

	inline void setCallBack(PoseCallBack callback) {
		callback_ = callback;
	}

  int feedImuData(double time, const Eigen::Vector3f& accl, const Eigen::Vector3f& gyro)
      LOCKS_EXCLUDED(imu_mutex_);

  int feedImage(double time, const cv::Mat& image)
      LOCKS_EXCLUDED(cam_mutex_);

  FeatureManager clearLoseFeature()
      LOCKS_EXCLUDED(mutex_);

private:

  bool setup();

  void imageProcess() 
      LOCKS_EXCLUDED(imu_mutex_) LOCKS_EXCLUDED(cam_mutex_) LOCKS_EXCLUDED(mutex_);

  bool initialization()
      LOCKS_EXCLUDED(imu_mutex_) LOCKS_EXCLUDED(cam_mutex_) LOCKS_EXCLUDED(mutex_);

	bool feedTrackResult(const TrackResult& track_result) 
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool predictImuStatus(const Eigen::Vector3d& accl, const Eigen::Vector3d& gyro, double dt)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool predictCamStatus(const TrackResult& track_result) 
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool getUpdateFeatures(UpdateFeatureIds &update_ids);

  bool featureUpdate(const UpdateFeatureIds &update_ids);

  bool featureUpdateStatus() 
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

	bool pruneCameraStatus()
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

	bool featureJacobian(const Feature& ftr, const CameraWindow& cam_window, const set<int>& constraint_cam_id, Eigen::MatrixXd& H_fj, Eigen::VectorXd& e_fj) 
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

	bool measurementUpdateStatus(Eigen::MatrixXd& H, Eigen::VectorXd& e) 
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool measurementUpdateIKF(const std::vector<int>& ftr_id, int measure_dim, std::string stage) 
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

	bool findRedundanceCam(CameraWindow& cameras, vector<int>& remove_cam_id) 
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

	bool gatingTest(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, const int v)
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool buildProblemWithFeature(const std::vector<int>& ftr_id, Eigen::MatrixXd& H, Eigen::VectorXd &e)
			EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  Eigen::MatrixXd kalmanGain(const Eigen::MatrixXd& P, Eigen::MatrixXd& H, Eigen::VectorXd& e)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool projectHpNullSpace(Eigen::MatrixXd& H_x, Eigen::MatrixXd& H_p, Eigen::VectorXd& e)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool updateStates(const Eigen::VectorXd& delta_x) 
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};

} // namespace MSCKF
