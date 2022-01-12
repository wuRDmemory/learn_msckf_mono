#include "node/node.h"

/**
 * ros topic names
 */
const char ref_frame[]         = "map";
const char imu_sub_name[]      = "imu";
const char img_sub_name[]      = "image";
const char img_pub_name[]      = "track_result";
const char cam_pub_path_name[] = "cam_path";
const char imu_pub_path_name[] = "imu_path";
const char imu_pub_odom_name[] = "imu_odom";
const char robot_pose_name[]   = "robot_pose";
const char point_cloud_name[]  = "feature";

Node::Node(ros::NodeHandle& n, string config_file) 
    : n_(n) {
  launchSubscriber();
  launchPublisher();
  wall_timer_ = n_.createWallTimer<Node>(ros::WallDuration(0.1), &Node::walltimerCallBack, this);
  msckf_.reset(new MSCKF::Msckf(config_file));
  msckf_->setCallBack(std::bind(&Node::infoCallBack, this, 
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}

Node::~Node() {
  if (!all_imu_pose_.empty()) {
    std::ofstream fp("/tmp/msckf_trajectory.csv");
    if (fp.is_open()) {
      for (const auto& state : all_camera_pose_) {
        double ts = state.second.ts;
        Eigen::Quaterniond Rwb = state.second.Rwc;
        Eigen::Vector3d pwb = state.second.pwc;
        fp << std::fixed << std::setprecision(6) 
            << ts << " " << pwb.x() << " " << pwb.y() << " " << pwb.z() << " "
            << Rwb.x() << " " << Rwb.y() << " " << Rwb.z() << " " << Rwb.w() << std::endl;
      }
    }
  }
  LOG(INFO) << "Finish node";
}

bool Node::launchSubscriber() {
  subscribers_.push_back(n_.subscribe<sensor_msgs::Imu>(imu_sub_name,   100, &Node::imuCallBack,   this));
  subscribers_.push_back(n_.subscribe<sensor_msgs::Image>(img_sub_name, 100, &Node::imageCallBack, this));
  return true;
}

bool Node::launchPublisher() {
  publishers_.insert(make_pair(img_pub_name,      n_.advertise<sensor_msgs::Image>(img_pub_name, 100)));
  publishers_.insert(make_pair(cam_pub_path_name, n_.advertise<nav_msgs::Path>(cam_pub_path_name, 100)));
  publishers_.insert(make_pair(point_cloud_name,  n_.advertise<sensor_msgs::PointCloud>(point_cloud_name, 100)));
  publishers_.insert(make_pair(imu_pub_path_name, n_.advertise<nav_msgs::Path>(imu_pub_path_name, 100)));
  publishers_.insert(make_pair(robot_pose_name,   n_.advertise<visualization_msgs::Marker>(robot_pose_name, 100)));
  publishers_.insert(make_pair(imu_pub_odom_name, n_.advertise<nav_msgs::Odometry>(imu_pub_odom_name, 100)));
  // ros::Publisher pub1 = n_.advertise<sensor_msgs::PointCloud>(point_cloud_name, 100);
  return true;
}

void Node::imageCallBack(const sensor_msgs::Image::ConstPtr& image) {
  // LOG(INFO) << "receive image data in " << image->header.stamp.toSec();
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::MONO8);
    cv::Mat img = cv_ptr->image;
    publishers_[img_pub_name].publish(cv_ptr);

    msckf_->feedImage(image->header.stamp.toSec(), img);
  } catch(exception e) {
    LOG(INFO) << e.what();
  }
}

void Node::imuCallBack(const sensor_msgs::Imu::ConstPtr& imu) {
  Eigen::Vector3f gyro(imu->angular_velocity.x,    imu->angular_velocity.y,    imu->angular_velocity.z);
  Eigen::Vector3f accl(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
  msckf_->feedImuData(imu->header.stamp.toSec(), accl, gyro);
}

void Node::infoCallBack(const MSCKF::ImuStatus& imu_status, const MSCKF::CameraWindow& cam_window, const MSCKF::FeatureManager& features)
{
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto& id_data : features) {
    all_mature_features_.insert(id_data);
  }

  for (auto& id_data : cam_window) {
    all_camera_pose_[id_data.first] = id_data.second;
  }

  all_imu_pose_.emplace_back(imu_status);
}

void Node::walltimerCallBack(const ros::WallTimerEvent& event) {
  unique_lock<mutex> lock(mutex_);
  publishCamPath();
  publishPointCloud();
  publishImuPath();
}

bool Node::publishCamPath() {
  nav_msgs::Path path;

  for (const auto& id_data: all_camera_pose_) {
    geometry_msgs::PoseStamped pose;
    const MSCKF::CameraStatus& cam = id_data.second;

    pose.header.frame_id = ref_frame;
    pose.header.seq      = id_data.first;
    pose.header.stamp    = ros::Time().fromSec(cam.ts);

    pose.pose.orientation.w = cam.Rwc.w();
    pose.pose.orientation.x = cam.Rwc.x();
    pose.pose.orientation.y = cam.Rwc.y();
    pose.pose.orientation.z = cam.Rwc.z();

    pose.pose.position.x = cam.pwc.x();
    pose.pose.position.y = cam.pwc.y();
    pose.pose.position.z = cam.pwc.z();

    path.poses.push_back(pose);
  }

  path.header.frame_id = ref_frame;
  path.header.seq      = 0; 
  path.header.stamp    = ros::Time().now();

  publishers_[cam_pub_path_name].publish(path);
  return true; 
}

bool Node::publishPointCloud() {
  sensor_msgs::PointCloud point_cloud;
  
  point_cloud.header.frame_id = ref_frame;
  point_cloud.header.seq = 0;
  
  for (const auto& id_data : all_mature_features_) {
    geometry_msgs::Point32 p;
    p.x = id_data.second.point_3d.x();
    p.y = id_data.second.point_3d.y();
    p.z = id_data.second.point_3d.z();
    point_cloud.points.push_back(p);
  }

  publishers_[point_cloud_name].publish(point_cloud);
  return true;
}

bool Node::publishImuPath() {
  nav_msgs::Path path;

  for (auto& imu_status : all_imu_pose_) {  
    geometry_msgs::PoseStamped pose;

    pose.header.frame_id = ref_frame;
    pose.header.seq      = imu_status.id;
    pose.header.stamp    = ros::Time().fromSec(imu_status.ts);

    pose.pose.orientation.w = imu_status.Rwb.w();
    pose.pose.orientation.x = imu_status.Rwb.x();
    pose.pose.orientation.y = imu_status.Rwb.y();
    pose.pose.orientation.z = imu_status.Rwb.z();

    pose.pose.position.x = imu_status.pwb.x();
    pose.pose.position.y = imu_status.pwb.y();
    pose.pose.position.z = imu_status.pwb.z();

    path.poses.push_back(pose);
  }

  path.header.frame_id = ref_frame;
  path.header.seq      = 0; 
  path.header.stamp    = ros::Time().now();

  publishers_[imu_pub_path_name].publish(path);

  if (!all_imu_pose_.empty()) {
    const MSCKF::ImuStatus& imu_status = all_imu_pose_.back();

    Eigen::Quaterniond qwc = imu_status.Rwb * imu_status.Rbc;
    Eigen::Vector3d pwc = imu_status.pwb + imu_status.Rwb*imu_status.pbc;
    publishRobotPose(qwc, pwc);

    nav_msgs::Odometry odom;
    odom.header = path.header;
    odom.child_frame_id = "robot";
    odom.pose.pose = path.poses.back().pose;
    odom.twist.twist.linear.x =  imu_status.g.x();
    odom.twist.twist.linear.y =  imu_status.g.y();
    odom.twist.twist.linear.z = -imu_status.g.z();

    publishers_[imu_pub_odom_name].publish(odom);
  }

  // if (!path.poses.empty()) {
  //   // boardcast tf information.
  //   geometry_msgs::TransformStamped transformStamped;
  //   transformStamped.header.stamp = ros::Time::now();
  //   transformStamped.header.frame_id = "world";
  //   transformStamped.child_frame_id  = "robot";
  //   transformStamped.transform.translation.x = path.poses.back().pose.position.x;
  //   transformStamped.transform.translation.y = path.poses.back().pose.position.y;
  //   transformStamped.transform.translation.z = path.poses.back().pose.position.z;
  //   transformStamped.transform.rotation = path.poses.back().pose.orientation;

  //   tf_broadcaster_.sendTransform(transformStamped);
  // }

  return true;
}

bool Node::publishRobotPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& p) {
  visualization_msgs::Marker marker;
  double scale = 0.2;
  double line_width = 0.01;

  marker.header.frame_id = ref_frame;
  marker.header.seq = 0;
  marker.header.stamp = ros::Time().now();

  marker.ns = "robot_pose_vis";
  marker.id = 1;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = line_width;

  marker.pose.position.x = 0.0;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;

  const Eigen::Vector3d imlt = Eigen::Vector3d(-1.0, -0.5, 1.0);
  const Eigen::Vector3d imrt = Eigen::Vector3d( 1.0, -0.5, 1.0);
  const Eigen::Vector3d imlb = Eigen::Vector3d(-1.0,  0.5, 1.0);
  const Eigen::Vector3d imrb = Eigen::Vector3d( 1.0,  0.5, 1.0);
  const Eigen::Vector3d lt0  = Eigen::Vector3d(-0.7, -0.5, 1.0);
  const Eigen::Vector3d lt1  = Eigen::Vector3d(-0.7, -0.2, 1.0);
  const Eigen::Vector3d lt2  = Eigen::Vector3d(-1.0, -0.2, 1.0);
  const Eigen::Vector3d oc   = Eigen::Vector3d(0.0, 0.0, 0.0);

  auto Eigen2Point = [](const Eigen::Vector3d& point) -> geometry_msgs::Point {
    geometry_msgs::Point p;
    p.x = point.x();
    p.y = point.y();
    p.z = point.z();
    return p;
  };

  geometry_msgs::Point pt_lt, pt_lb, pt_rt, pt_rb, pt_oc, pt_lt0, pt_lt1, pt_lt2;

  pt_lt  = Eigen2Point(q*(scale*imlt) + p);
  pt_lb  = Eigen2Point(q*(scale*imlb) + p);
  pt_rt  = Eigen2Point(q*(scale*imrt) + p);
  pt_rb  = Eigen2Point(q*(scale*imrb) + p);
  pt_lt0 = Eigen2Point(q*(scale*lt0 ) + p);
  pt_lt1 = Eigen2Point(q*(scale*lt1 ) + p);
  pt_lt2 = Eigen2Point(q*(scale*lt2 ) + p);
  pt_oc  = Eigen2Point(q*(scale*oc  ) + p);

  std_msgs::ColorRGBA image_boundary_color;
  image_boundary_color.a = 1;
  image_boundary_color.g = 1;
  image_boundary_color.r = 0;
  image_boundary_color.b = 0;

  std_msgs::ColorRGBA optical_center_connector_color;
  optical_center_connector_color.a = 1;
  optical_center_connector_color.g = 1;
  optical_center_connector_color.r = 0;
  optical_center_connector_color.b = 0;

  // image boundaries
  marker.points.push_back(pt_lt);
  marker.points.push_back(pt_lb);
  marker.colors.push_back(image_boundary_color);
  marker.colors.push_back(image_boundary_color);

  marker.points.push_back(pt_lb);
  marker.points.push_back(pt_rb);
  marker.colors.push_back(image_boundary_color);
  marker.colors.push_back(image_boundary_color);

  marker.points.push_back(pt_rb);
  marker.points.push_back(pt_rt);
  marker.colors.push_back(image_boundary_color);
  marker.colors.push_back(image_boundary_color);

  marker.points.push_back(pt_rt);
  marker.points.push_back(pt_lt);
  marker.colors.push_back(image_boundary_color);
  marker.colors.push_back(image_boundary_color);

  // top-left indicator
  marker.points.push_back(pt_lt0);
  marker.points.push_back(pt_lt1);
  marker.colors.push_back(image_boundary_color);
  marker.colors.push_back(image_boundary_color);

  marker.points.push_back(pt_lt1);
  marker.points.push_back(pt_lt2);
  marker.colors.push_back(image_boundary_color);
  marker.colors.push_back(image_boundary_color);

  // optical center connector
  marker.points.push_back(pt_lt);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(optical_center_connector_color);
  marker.colors.push_back(optical_center_connector_color);

  marker.points.push_back(pt_lb);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(optical_center_connector_color);
  marker.colors.push_back(optical_center_connector_color);

  marker.points.push_back(pt_rt);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(optical_center_connector_color);
  marker.colors.push_back(optical_center_connector_color);

  marker.points.push_back(pt_rb);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(optical_center_connector_color);
  marker.colors.push_back(optical_center_connector_color);

  publishers_[robot_pose_name].publish(marker);

  return true;
}
