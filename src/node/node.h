#pragma once

#include "iostream"
#include "unordered_map"
#include "memory"
#include "functional"
#include "mutex"

#include "boost/functional.hpp"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "absl/synchronization/mutex.h"
#include "absl/memory/memory.h"

#include "ros/ros.h"
#include "tf2_ros/transform_broadcaster.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Point32.h"
#include "geometry_msgs/Point.h"
#include "sensor_msgs/PointCloud.h"
#include "nav_msgs/Path.h"
#include "nav_msgs/Odometry.h"
#include "visualization_msgs/MarkerArray.h"
#include "tf2_ros/transform_broadcaster.h"
#include "ros/wall_timer.h"

#include "msckf/msckf.h"
#include "msckf/datas.h"

using namespace std;

class Node {
private:
  ros::NodeHandle& n_;
  
  vector<ros::Subscriber>                 subscribers_;
  unordered_map<string, ros::Publisher>   publishers_;
  tf2_ros::TransformBroadcaster           tf_broadcaster_;
  ros::WallTimer                          wall_timer_;

  mutex mutex_;
  unique_ptr<MSCKF::Msckf> msckf_;
  MSCKF::FeatureManager    all_mature_features_ GUARDED_BY(mutex_);
  MSCKF::CameraWindow      all_camera_pose_     GUARDED_BY(mutex_);
  vector<MSCKF::ImuStatus> all_imu_pose_        GUARDED_BY(mutex_);

public:
  Node(ros::NodeHandle& n, string config_file);
  ~Node();

private:
  void infoCallBack(const MSCKF::ImuStatus& imu_status, const MSCKF::CameraWindow& cam_window, const MSCKF::FeatureManager& features);

  void imuCallBack(const sensor_msgs::Imu::ConstPtr& imu);
  
  void imageCallBack(const sensor_msgs::Image::ConstPtr& image);

  void walltimerCallBack(const ros::WallTimerEvent& event);

  bool launchSubscriber();

  bool launchPublisher();

  bool publishCamPath() EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool publishPointCloud() EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool publishImuPath() EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  
  bool publishRobotPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& p) EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};
