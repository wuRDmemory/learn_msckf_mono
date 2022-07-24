#include "msckf.h"
#include "camera/camera.h"

namespace MSCKF {

Msckf::Msckf(string config_path)
{
  LOG(INFO) << "[MSKCF] Construct msckf.";
  Config::readConfig(config_path);
  param_ = Config::msckf_param;

  std::deque<ImuData>().swap(imu_buffer_);
  std::deque<CameraData>().swap(cam_buffer_);

  image_tracker_ = absl::make_unique<ImageTracker>(Config::track_param, Config::camera_param);
  sfm_ptr_ = absl::make_unique<SFM>(Config::sfm_param);

  for (int i = 1; i < 100; ++i) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_square_distribution_[i] = boost::math::quantile(chi_squared_dist, 0.05);
  }

  ofstream outf("/home/ubuntu/chi_square_dist.txt");
  for (auto iter : chi_square_distribution_) {
    outf << iter.first << ": " << iter.second << "," << endl;
  }
  outf.close();

  setup();

  is_stop_ = false;
  image_process_thread_ = thread(&Msckf::imageProcess, this);
}

Msckf::~Msckf() {
  LOG(INFO) << "[MSKCF] Deconstructing msckf...";
  is_stop_ = true;
  if (image_process_thread_.joinable()) {
    image_process_thread_.join();
  }
  LOG(INFO) << "[MSKCF] Msckf deconstructed.";
}

bool Msckf::setup()
{
  std::unique_lock<mutex> lock(mutex_);
  data_.imu_status.id = 0;
  data_.imu_status.ts = -1;
  data_.imu_status.ba.setZero();
  data_.imu_status.bg.setZero();
  data_.imu_status.Rwb.setIdentity();
  data_.imu_status.pwb.setZero();
  data_.imu_status.vwb.setZero();
  
  { // set extrinsic
    cv::Mat Rbc;
    cv::Mat pbc;
    Config::Tbc.colRange(0, 3).copyTo(Rbc);
    Config::Tbc.col(3).copyTo(pbc);
    
    Eigen::Matrix3d eigen_Rbc;
    Eigen::Vector3d eigen_pbc;
    cv::cv2eigen(Rbc, eigen_Rbc);
    cv::cv2eigen(pbc, eigen_pbc);
    data_.imu_status.Rbc = Eigen::Quaterniond(eigen_Rbc); 
    data_.imu_status.pbc = eigen_pbc;
    LOG(INFO) << "[MSCKF] Rbc: " << data_.imu_status.Rbc.toRotationMatrix().eulerAngles(2, 1, 0).transpose()*Deg2Rad;
    LOG(INFO) << "[MSCKF] pbc: " << data_.imu_status.pbc.transpose();
  }

  // set covariance
  data_.P_dx.resize(IMU_STATUS_DIM, IMU_STATUS_DIM); 
  data_.P_dx.setZero();
  data_.P_dx.block<3, 3>(J_BG, J_BG) = Eigen::Matrix3d::Identity()*0.01;
  data_.P_dx.block<3, 3>(J_V,  J_V)  = Eigen::Matrix3d::Identity()*0.25;
  data_.P_dx.block<3, 3>(J_BA, J_BA) = Eigen::Matrix3d::Identity()*0.01;
  // data_.P_dx.block<2, 2>(J_G,  J_G)  = Eigen::Matrix2d::Identity()*0.01;

  data_.Phi_test.setIdentity();

  // set noise covariance
  data_.Q_imu.setZero();
  data_.Q_imu.block<3, 3>(G_Ng , G_Ng ) = Eigen::Matrix3d::Identity()*param_.noise_gyro;
  data_.Q_imu.block<3, 3>(G_Nbg, G_Nbg) = Eigen::Matrix3d::Identity()*param_.noise_gyro_bias;
  data_.Q_imu.block<3, 3>(G_Na , G_Na ) = Eigen::Matrix3d::Identity()*param_.noise_accl;
  data_.Q_imu.block<3, 3>(G_Nba, G_Nba) = Eigen::Matrix3d::Identity()*param_.noise_accl_bias;

  // set observes
  data_.imu_status.observes.clear();

  return true;
}

int Msckf::feedImuData(double time, const Eigen::Vector3f& accl, const Eigen::Vector3f& gyro)
{
  {
    std::unique_lock<std::mutex> lock(imu_mutex_);
    imu_buffer_.push_back(ImuData{time, accl, gyro});
  }

  if (!is_initial_) {
    is_initial_ = initialization();
  }
  return (int)imu_buffer_.size();
}

int Msckf::feedImage(double time, const cv::Mat& image)
{
  if (!is_initial_) {
    return 0;
  }

  if (is_first_image_) {
    is_first_image_ = false;
  }

  std::unique_lock<std::mutex> lock(cam_mutex_);
  cam_buffer_.push_back(CameraData{time, image});
  return (int)cam_buffer_.size();

  // if (!image_tracker_->feedImage(time, image)) {
  //   return 0;
  // }

  // unique_lock<mutex> lock(mutex_);
  // return feedTrackResult(image_tracker_->fetchResult());
}

FeatureManager Msckf::clearLoseFeature()
{
  std::unique_lock<std::mutex> lock(mutex_);

  FeatureManager ret;
  FeatureManager::iterator iter = data_.features_.begin();
  while (iter != data_.features_.end()) {
    if (iter->second.status != FeatureStatus::WaitForDelete) {
      iter = next(iter);
      continue;
    }

    ret.insert(*iter);
    iter = data_.features_.erase(iter);
  }

  return ret;
}

void Msckf::imageProcess()
{
  LOG(INFO) << "[MSCKF] Image process begin...";
  while (!is_stop_) {
    if (cam_buffer_.empty()) {
      chrono::milliseconds dura(10);
      this_thread::sleep_for(dura);
      continue;
    }

    while (!cam_buffer_.empty()) {
      CameraData data;
      {
        std::unique_lock<std::mutex> lock(cam_mutex_);
        data = cam_buffer_.front();
        cam_buffer_.pop_front();
      }

      if (image_tracker_->feedImage(data.ts, data.image)) {
        std::unique_lock<std::mutex> lock(mutex_);
        feedTrackResult(image_tracker_->fetchResult());
      }
    }
  }

  LOG(INFO) << "[MSCKF] Image process done!!!";
}

bool Msckf::initialization()
{
  size_t N = 0;
  double timestamp = DBL_MIN;
  Eigen::Vector3f accl_sum(0, 0, 0), accl2_sum(0, 0, 0);
  Eigen::Vector3f gyro_sum(0, 0, 0), gyro2_sum(0, 0, 0);
  
  {
    std::unique_lock<mutex> lock(imu_mutex_);
    if (imu_buffer_.size() < Config::init_param.imu_cnt) {
      return false;
    }

    for (size_t i = 0; i < imu_buffer_.size(); ++i) {
      const ImuData& imu_data = imu_buffer_.at(i);
      timestamp = imu_data.ts;
      accl_sum  += imu_data.accl;
      gyro_sum  += imu_data.gyro;
      accl2_sum += imu_data.accl.cwiseAbs2();
      gyro2_sum += imu_data.gyro.cwiseAbs2();
      ++N;
    }
  }

  Eigen::Vector3f accl_mean  = accl_sum / N;
  Eigen::Vector3f gyro_mean  = gyro_sum / N;
  Eigen::Vector3f accl2_mean = accl2_sum / N;
  Eigen::Vector3f gyro2_mean = gyro2_sum / N;

  Eigen::Vector3f accl_cov = accl2_mean - accl_mean.cwiseAbs2();
  Eigen::Vector3f gyro_cov = gyro2_mean - gyro_mean.cwiseAbs2();

  if (Config::init_param.verbose) {
    LOG(INFO) << "[INIT] Accl mean: " << accl_mean.transpose();
    LOG(INFO) << "[INIT] Gyro mean: " << gyro_mean.transpose();
    LOG(INFO) << "[INIT] Accl covariance: " << accl_cov.transpose();
    LOG(INFO) << "[INIT] Gyro covariance: " << gyro_cov.transpose();
  }

  if (   accl_cov.maxCoeff() < Config::init_param.imu_accl_cov 
      && gyro_cov.maxCoeff() < Config::init_param.imu_gyro_cov) {
    data_.imu_status.bg = gyro_mean.cast<double>();
    data_.imu_status.ba = Eigen::Vector3d(-0.025266,0.136696,0.075593);
    
    data_.imu_status.g   = Eigen::Vector3d(0, 0, 9.81);
    data_.imu_status.Rwb = Eigen::Quaterniond::FromTwoVectors(accl_mean.cast<double>(), data_.imu_status.g);
    data_.imu_status.pwb.setZero();
    data_.imu_status.vwb.setZero();

    data_.imu_status.Rwb_nullspace = data_.imu_status.Rwb;
    data_.imu_status.pwb_nullspace = data_.imu_status.pwb;
    data_.imu_status.vwb_nullspace = data_.imu_status.vwb;

    data_.imu_status.ts  = timestamp;

    Eigen::Vector3d angles = data_.imu_status.Rwb.toRotationMatrix().eulerAngles(2, 1, 0);
    // data_.imu_status.Rwb = MATH_UTILS::eulerToQuaternion<double>(Eigen::Vector3d(0, angles[1], angles[2]));
    // angles = data_.imu_status.Rwb.toRotationMatrix().eulerAngles(2, 1, 0);

    if (Config::init_param.verbose) {
      LOG(INFO) << std::fixed << std::setprecision(4) << "[INIT] IMU initialization success in " << timestamp << "s.";
      LOG(INFO) << std::fixed << std::setprecision(4) << "[INIT] Gravity is:   " << data_.imu_status.g.transpose();
      LOG(INFO) << std::fixed << std::setprecision(4) << "[INIT] GYRO bias:    " << gyro_mean.transpose();
      LOG(INFO) << std::fixed << std::setprecision(4) << "[INIT] Initial Pose: " << angles.transpose()*Rad2Deg;
    }

    std::unique_lock<std::mutex> lock(imu_mutex_);
    while (!imu_buffer_.empty() && imu_buffer_.front().ts <= timestamp) {
      imu_buffer_.pop_front();
    }

    return true;
  }

  std::unique_lock<std::mutex> lock(imu_mutex_);
  for (size_t i = 0; i < imu_buffer_.size()/10; ++i) {
    imu_buffer_.pop_front();
  }

  return false;
}

bool Msckf::feedTrackResult(const TrackResult& track_result)
{
  std::vector<ImuData> imu_data;
  {
    unique_lock<mutex> lock(imu_mutex_);
    while (!imu_buffer_.empty() && imu_buffer_.front().ts <= track_result.ts) {
      imu_data.push_back(imu_buffer_.front());
      imu_buffer_.pop_front();
    }
  }

  // CHECK(!imu_data.empty()) << "This should never happen.";
  if (!imu_data.empty() && last_imu_.ts < 0) {
    last_imu_ = imu_data[0];
  }

  // save previous imu status for static check
  data_.last_imu_status = data_.imu_status;
  for (size_t i = 0; i < imu_data.size(); ++i) {
    const Eigen::Vector3f accl = 0.5*imu_data[i].accl + 0.5*last_imu_.accl;
    const Eigen::Vector3f gyro = 0.5*imu_data[i].gyro + 0.5*last_imu_.gyro;
    predictImuStatus(accl.cast<double>(), gyro.cast<double>(), imu_data[i].ts);
    last_imu_ = imu_data[i];
  }

  std::set<int> obs_set(track_result.point_id.begin(), track_result.point_id.end());
  data_.imu_status.observes.swap(obs_set);
  
  // LOG(INFO) << "[MSCKF] Move speed " << cv::norm(track_result.velocity_) << " pixel.";
  if (cv::norm(track_result.velocity_) < param_.static_vel) {
    // LOG(INFO) << "[MSCKF] velocity: " << track_result.velocity_;
    COMMON::TicToc tick;
    staticCorrect();
    if (param_.verbose) {
      LOG(INFO) << "[MSCKF] Update static speed " << tick.toc() << "s.";
    }
  } else {
    COMMON::TicToc tick;
    if (predictCamStatus(track_result)) {
      LOG(INFO) << "[MSCKF] Update feature Status";
      featureUpdateStatus();
      LOG(INFO) << "[MSCKF] Prune camera Status";
      pruneCameraStatus();
      if (param_.verbose) {
        LOG(INFO) << "[MSCKF] Update speed " << tick.toc() << "s.";
      }
    }
  }

  data_.imu_status.id++;
  
  if (callback_) {
    callback_(data_.imu_status, data_.cameras_, mature_features_);
  }
 
  return true;
}

bool Msckf::predictImuStatus(const Eigen::Vector3d& accl_m, const Eigen::Vector3d& gyro_m, double time)
{
  ImuStatus& imu_status = data_.imu_status;

  if (imu_status.ts < 0) {
    imu_status.ts = time;
    return false;
  }

  const double dt = time - imu_status.ts;
  imu_status.ts = time;

  Eigen::Vector3d accl = accl_m - imu_status.ba;
  Eigen::Vector3d gyro = gyro_m - imu_status.bg;
  
  // update covariance of IMU
  IMU_Matrix F = IMU_Matrix::Zero();
  F.block<3, 3>(J_R, J_R ) = -MATH_UTILS::skewMatrix<double>(gyro);
  F.block<3, 3>(J_R, J_BG) = -Eigen::Matrix3d::Identity();
  F.block<3, 3>(J_V, J_R ) = -imu_status.Rwb.toRotationMatrix()*MATH_UTILS::skewMatrix<double>(accl);
  F.block<3, 3>(J_V, J_BA) = -imu_status.Rwb.toRotationMatrix();
  // F.block<3, 2>(J_V, J_G)  =  MATH_UTILS::skewMatrix<double>(imu_status.g)*imu_status.S2Bx();
  F.block<3, 3>(J_P, J_V ) =  Eigen::Matrix3d::Identity();

  Drive_Matrix G = Drive_Matrix::Zero();
  G.block<3, 3>(J_R,  G_Ng ) = -Eigen::Matrix3d::Identity();
  G.block<3, 3>(J_BG, G_Nbg) =  Eigen::Matrix3d::Identity();
  G.block<3, 3>(J_V,  G_Na ) = -imu_status.Rwb.toRotationMatrix();
  G.block<3, 3>(J_BA, G_Nba) =  Eigen::Matrix3d::Identity();

  IMU_Matrix Fdtime  = F*dt;
  IMU_Matrix Fdtime2 = Fdtime*Fdtime;
  IMU_Matrix Fdtime3 = Fdtime2*Fdtime;
  
  IMU_Matrix Phi = IMU_Matrix::Identity() + Fdtime + 0.5*Fdtime2 + (1./6)*Fdtime3;
  IMU_Matrix V   = Phi*G*data_.Q_imu*G.transpose()*Phi.transpose()*dt;

  // update status of IMU
  Eigen::Quaterniond q_dt  = imu_status.Rwb*MATH_UTILS::rotateVecToQuaternion<double>(gyro*dt); q_dt.normalize();
  Eigen::Quaterniond q_2dt = imu_status.Rwb*MATH_UTILS::rotateVecToQuaternion<double>(gyro*dt/2); q_2dt.normalize();

  // k1 = f(t, x)
  Eigen::Vector3d delta_v_k1 = imu_status.Rwb*accl - data_.imu_status.g;
  Eigen::Vector3d delta_p_k1 = imu_status.vwb;

  // k2 = f(t+dt/2, x+k1*dt/2)
  Eigen::Vector3d delta_v_k2 = q_2dt*accl - data_.imu_status.g;
  Eigen::Vector3d delta_p_k2 = imu_status.vwb + delta_v_k1*dt/2;

  // k3 = f(t+dt/2, x+k2*dt/2)
  Eigen::Vector3d delta_v_k3 = q_2dt*accl - data_.imu_status.g;
  Eigen::Vector3d delta_p_k3 = imu_status.vwb + delta_v_k2*dt/2;

  // k4 = f(t+dt, x+k3*dt)
  Eigen::Vector3d delta_v_k4 = q_dt*accl - data_.imu_status.g;
  Eigen::Vector3d delta_p_k4 = imu_status.vwb + delta_v_k3*dt;

  imu_status.vwb = imu_status.vwb + (delta_v_k1 + 2*delta_v_k2 + 2*delta_v_k3 + delta_v_k4)*dt/6;
  imu_status.pwb = imu_status.pwb + (delta_p_k1 + 2*delta_p_k2 + 2*delta_p_k3 + delta_p_k4)*dt/6;
  // imu_status.pwb = imu_status.pwb + (imu_status.vwb + 0.5*(imu_status.Rwb*accl - data_.imu_status.g)*dt)*dt;
  // imu_status.vwb = imu_status.vwb + (imu_status.Rwb*accl - data_.imu_status.g)*dt;
  imu_status.Rwb = q_dt;

  if (dt > 0.01) {
    cout << dt << endl;
  }

  if (1) { // OC-KF 
    // A. change delta_theta
    Phi.block<3, 3>(J_R, J_R) = imu_status.Rwb.toRotationMatrix().transpose()*imu_status.Rwb_nullspace.toRotationMatrix();
    // B. change delta_v
    Eigen::Vector3d w = MATH_UTILS::skewMatrix<double>(imu_status.vwb_nullspace - imu_status.vwb)*data_.imu_status.g;
    Eigen::Vector3d u = imu_status.Rwb_nullspace.toRotationMatrix().transpose()*data_.imu_status.g;
    Eigen::Matrix3d Phi_v = Phi.block<3, 3>(J_V, J_R);
    Phi.block<3, 3>(J_V, J_R) = Phi_v - (Phi_v*u - w)*(u.transpose()*u).inverse()*u.transpose();

    Eigen::Matrix3d Phi_p = Phi.block<3, 3>(J_P, J_R);
    w = MATH_UTILS::skewMatrix<double>(imu_status.pwb_nullspace + imu_status.vwb_nullspace*dt - imu_status.pwb)*data_.imu_status.g;
    Phi.block<3, 3>(J_P, J_R) = Phi_p - (Phi_p*u - w)*(u.transpose()*u).inverse()*u.transpose();

    data_.imu_status.Rwb_nullspace = data_.imu_status.Rwb;
    data_.imu_status.pwb_nullspace = data_.imu_status.pwb;
    data_.imu_status.vwb_nullspace = data_.imu_status.vwb;
  }

  Eigen::MatrixXd P = data_.P_dx.block<IMU_STATUS_DIM, IMU_STATUS_DIM>(0, 0);
  data_.P_dx.block<IMU_STATUS_DIM, IMU_STATUS_DIM>(0, 0) = Phi*P*Phi.transpose() + V;
  
  if (data_.P_dx.size() > IMU_STATUS_DIM) {
    const int N = data_.P_dx.rows();
    Eigen::MatrixXd tmp = data_.P_dx.block(0, IMU_STATUS_DIM, IMU_STATUS_DIM, N-IMU_STATUS_DIM);
    data_.P_dx.block(0, IMU_STATUS_DIM, IMU_STATUS_DIM, N-IMU_STATUS_DIM) = Phi*tmp;
    data_.P_dx.block(IMU_STATUS_DIM, 0, N-IMU_STATUS_DIM, IMU_STATUS_DIM) = data_.P_dx.block(0, IMU_STATUS_DIM, IMU_STATUS_DIM, N-IMU_STATUS_DIM).transpose();
  }

  return 1;
}

bool Msckf::predictCamStatus(const TrackResult& track_result)
{
  if (!data_.cameras_.empty()) {
    const auto& imu_status = data_.imu_status;
    Eigen::Quaterniond qwc = imu_status.Rwb*imu_status.Rbc; qwc.normalize();
    Eigen::Vector3d pwc = imu_status.pwb + imu_status.Rwb*imu_status.pbc;

    Eigen::Quaterniond last_Rwc = std::prev(data_.cameras_.end())->second.Rwc;
    Eigen::Vector3d last_pwc = std::prev(data_.cameras_.end())->second.pwc;
    
    Eigen::Vector3d rotate = MATH_UTILS::quaternionToRotateVector<double>(last_Rwc.inverse()*qwc);
    if (rotate.norm() < 0.0523 && (pwc - last_pwc).norm() < 0.1) {
      return false;
    }
  }

  // add features
  int feature_cnt = data_.features_.size();
  int curr_track_feature = 0;

  const int cam_id = data_.imu_status.id;
  auto& features   = data_.features_;
  for (size_t i = 0; i < track_result.point_id.size(); ++i) {
    if (!features.count(track_result.point_id[i])) {
      features.insert(make_pair(track_result.point_id[i], Feature{}));
      Feature& feature = features.at(track_result.point_id[i]);
      feature.status   = FeatureStatus::NotInit;
      feature.point_3d.setConstant(-1);
      --curr_track_feature;
    }

    cv::Point2f point_f = track_result.point_f[i];
    Feature&    feature = features.at(track_result.point_id[i]);
    feature.observes.insert(make_pair(cam_id, Eigen::Vector3d(point_f.x, point_f.y, 1.)));
    ++curr_track_feature;
  }

  track_rate_ = (double)curr_track_feature/feature_cnt;

  // add new camera
  const ImuStatus& imu_status = data_.imu_status;
  if (!data_.cameras_.count(cam_id)) {
    data_.cameras_.insert(make_pair(cam_id, CameraStatus{}));
  }

  CameraStatus& camera_status = data_.cameras_.at(cam_id);
  camera_status.ts  = track_result.ts;
  camera_status.Rwc = imu_status.Rwb*imu_status.Rbc; camera_status.Rwc.normalize();
  camera_status.pwc = imu_status.pwb + imu_status.Rwb*imu_status.pbc;

  camera_status.Rwc_nullspace = camera_status.Rwc;
  camera_status.pwc_nullspace = camera_status.pwc;

  // covariance expand.
  const int old_row = data_.P_dx.rows();
  const int old_col = data_.P_dx.cols();

  Eigen::MatrixXd& P = data_.P_dx;
  Eigen::MatrixXd J_cam_imu = Eigen::MatrixXd::Zero(6, IMU_STATUS_DIM);
  J_cam_imu.block<3, 3>(C_R, J_R) =  imu_status.Rbc.toRotationMatrix().transpose();
  J_cam_imu.block<3, 3>(C_P, J_R) = -imu_status.Rwb.toRotationMatrix()*MATH_UTILS::skewMatrix(imu_status.pbc);
  J_cam_imu.block<3, 3>(C_P, J_P) =  Eigen::Matrix3d::Identity();

  P.conservativeResize(old_row + 6, old_col + 6);

  const Eigen::MatrixXd& P_ii = P.block(0,              0, IMU_STATUS_DIM, IMU_STATUS_DIM);
  const Eigen::MatrixXd& P_ic = P.block(0, IMU_STATUS_DIM, IMU_STATUS_DIM, old_col - IMU_STATUS_DIM);

  P.block(old_row, 0, 6, old_col) << J_cam_imu*P_ii, J_cam_imu*P_ic;
  P.block(0, old_col, old_row, 6) = P.block(old_row, 0, 6, old_col).transpose();
  P.block(old_row, old_col, 6, 6) = J_cam_imu*P_ii*J_cam_imu.transpose();

  Eigen::MatrixXd new_P = (P + P.transpose())/2.0; 
  P = new_P;

  return true;
}

bool Msckf::featureUpdateStatus()
{
  // A. find point which is losed
  std::vector<int> invalid_feature_id;
  std::vector<int> lose_feature_id;
  for (auto& id_ftr : data_.features_) {
    const int id = id_ftr.first;
    Feature& ftr = id_ftr.second;

    if (data_.imu_status.observes.count(id)) {
      continue;
    }

    if (ftr.observes.size() < 3) {
      invalid_feature_id.push_back(id);
      continue;
    }

    if (ftr.status == FeatureStatus::NotInit) {
      if (!(sfm_ptr_->checkMotion(ftr, data_.cameras_) && sfm_ptr_->initialFeature(ftr, data_.cameras_))) {
        invalid_feature_id.push_back(id);
        continue;
      }
      
      lose_feature_id.push_back(id);
    }
    else {
      lose_feature_id.push_back(id);    
    }
  }

  if (lose_feature_id.empty()) {
    return false;
  }

  if (param_.verbose) {
    LOG(INFO) << "[LOSE] Lose feature: " << lose_feature_id.size() << " | " << invalid_feature_id.size();
  }

  // B. use lose features to update all status.
  int jacobian_rows = 0;
  for (size_t i = 0; i < lose_feature_id.size(); ++i) {
    const Feature& ftr = data_.features_.at(lose_feature_id[i]);
    jacobian_rows += ftr.observes.size()*2 - 3;
  }

  if (!measurementUpdateIKF(lose_feature_id, jacobian_rows, "LOSE")) {
    if (param_.verbose) {
      LOG(WARNING) << "[LOSE] IKF update error!!!";
    }
    return false;
  }

  // D. remove invalid and lose feature.
  for (int ftr_id : lose_feature_id) {
    mature_features_[ftr_id] = data_.features_[ftr_id];
    data_.features_[ftr_id].status = FeatureStatus::Deleted;
    data_.features_.erase(ftr_id);
  }

  for (int ftr_id : invalid_feature_id) {
    data_.features_[ftr_id].status = FeatureStatus::Deleted;
    data_.features_.erase(ftr_id);
  }

  return true;
}

bool Msckf::pruneCameraStatus()
{
  if (data_.cameras_.size() < param_.sliding_window_lens) {
    return false;
  } 

  std::vector<int> remove_cam_id;
  findRedundanceCam(data_.cameras_, remove_cam_id);
  CHECK_EQ(remove_cam_id.size(), 2);

  int common_vis_cnt = 0;
  int valid_vis_cnt = 0;
  int motion_invalid_cnt = 0;
  int init_invalid_cnt = 0;
  int jacobian_rows = 0;
  for (auto& id_data : data_.features_) {
    Feature&  ftr = id_data.second;
    auto& observe = ftr.observes;

    std::vector<int> constraint_cam_id;
    for (int cam_id : remove_cam_id) {
      if (observe.count(cam_id)) {
        constraint_cam_id.push_back(cam_id);
      }
    }

    if (constraint_cam_id.size() == 0) {
      continue;
    }
    
    if (constraint_cam_id.size() == 1) {
      observe.erase(constraint_cam_id[0]);
      continue;
    }

    common_vis_cnt++;
    if (ftr.status == FeatureStatus::NotInit) {
      if (!sfm_ptr_->checkMotion(ftr, data_.cameras_)) {
        for (int cam_id : constraint_cam_id) {
          observe.erase(cam_id);
        }
        motion_invalid_cnt++;
        continue;
      }
      else if (!sfm_ptr_->initialFeature(ftr, data_.cameras_)) {
        for (int cam_id : constraint_cam_id) {
          observe.erase(cam_id);
        }
        init_invalid_cnt++;
        continue;       
      }
      valid_vis_cnt++;
      // jacobian_rows += (constraint_cam_id.size()*2 - 3);
      jacobian_rows += (observe.size()*2 - 3);
    }
    else {
      valid_vis_cnt++;
      // jacobian_rows += (constraint_cam_id.size()*2 - 3);
      jacobian_rows += (observe.size()*2 - 3);
    }
  }

  if (jacobian_rows != 0) {
    std::vector<int> valid_ftr_id;
    for (auto& id_data : data_.features_) {
      int ftr_id = id_data.first;
      Feature& ftr = id_data.second;
      auto& observe = ftr.observes;

      std::set<int> constraint_cam_id;
      for (int cam_id : remove_cam_id) {
        if (observe.count(cam_id)) {
          constraint_cam_id.insert(cam_id);
        }
      }

      if (constraint_cam_id.size() == 0) {
        continue;
      }

      valid_ftr_id.emplace_back(ftr_id);
    }

    measurementUpdateIKF(valid_ftr_id, jacobian_rows, "PRUNE");

    // remove cam in observe
    for (int id : valid_ftr_id) {
      for (int cam_id : remove_cam_id) {
        data_.features_[id].observes.erase(cam_id);
      }
    }
  }
  
  Eigen::MatrixXd& P = data_.P_dx;
  for (const int cam_id : remove_cam_id) {
    int cam_sequence     = std::distance(data_.cameras_.begin(), data_.cameras_.find(cam_id));
    int cam_status_start = IMU_STATUS_DIM + 6*cam_sequence;
    int cam_status_end   = cam_status_start + 6;

    if (cam_status_end < P.rows()) {
      const int old_rows = P.rows();
      const int old_cols = P.cols();
      P.block(cam_status_start, 0, old_rows - cam_status_end, old_cols) = P.block(cam_status_end, 0, old_rows - cam_status_end, old_cols);
      P.block(0, cam_status_start, old_rows, old_cols - cam_status_end) = P.block(0, cam_status_end, old_rows, old_cols - cam_status_end);
      P.conservativeResize(old_rows - 6, old_cols - 6);
    } else {
      const int old_rows = P.rows();
      const int old_cols = P.cols();
      P.conservativeResize(old_rows - 6, old_cols - 6);
    }

    data_.cameras_.erase(cam_id);
  }

  return true;
}

bool Msckf::buildProblemWithFeature(const std::vector<int>& ftr_id, Eigen::MatrixXd& H, Eigen::VectorXd &e)
{
  if (ftr_id.empty()) {
    return false;
  }

  int arows = 0;
  int rows = 0;
  for (int id : ftr_id) {
    const Feature& ftr = data_.features_.at(id);
    const auto& observe = ftr.observes;

    std::set<int> constraint_cam_id;
    for (auto& id_data : observe) {
      constraint_cam_id.insert(id_data.first);
    }

    Eigen::MatrixXd H_fj;
    Eigen::VectorXd e_fj;
    featureJacobian(ftr, data_.cameras_, constraint_cam_id, H_fj, e_fj);

    arows += H_fj.rows();
    if (gatingTest(H_fj, e_fj, constraint_cam_id.size())) {
      H.block(rows, 0, H_fj.rows(), H_fj.cols()) = H_fj;
      e.segment(rows,  e_fj.rows()) = e_fj;
      rows += H_fj.rows();
    }
  }

  if (rows == 0) {
    return false;
  }

  H.conservativeResize(rows, data_.cameras_.size()*6 + IMU_STATUS_DIM);
  e.conservativeResize(rows);

  return true;
}

bool Msckf::measurementUpdateIKF(const std::vector<int>& ftr_id, int measure_dim, std::string stage)
{
  if (ftr_id.empty()) {
    return false;
  }

  const ImuStatus    imu_prop = data_.imu_status;
  const CameraWindow cam_prop = data_.cameras_;

  const int N = data_.cameras_.size()*6 + IMU_STATUS_DIM;
  Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(N);
  Eigen::MatrixXd P = data_.P_dx;
  
  int iter = 0;
  bool converage = false;
  double init_loss = -1;
  double last_loss = -1;
  for (iter = 1; iter <= param_.ikf_iters; ++iter) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(measure_dim, N);
    Eigen::VectorXd e = Eigen::VectorXd::Zero(measure_dim);
    if (!buildProblemWithFeature(ftr_id, H, e)) {
      LOG(WARNING) << "[IKF] Can not get H and e, init measure dim: " << measure_dim;
      return false;
    }

    last_loss = e.norm();
    if (init_loss < 0) {
      init_loss = last_loss;
    }

    Eigen::MatrixXd K = kalmanGain(P, H, e);
    Eigen::VectorXd iter_delta_x = K*(e + H * delta_x) - delta_x;
    if (!updateStates(iter_delta_x)) {
      return false;
    }

    converage = iter_delta_x.maxCoeff() < 0.001;
    if (converage || iter == param_.ikf_iters) {
      Eigen::MatrixXd new_P = (Eigen::MatrixXd::Identity(K.rows(), H.cols()) - K*H)*data_.P_dx;
      data_.P_dx = (new_P + new_P.transpose()) / 2.0;
      break;
    }

    // build delta_x
    delta_x.head<IMU_STATUS_DIM>() = data_.imu_status.boxMinus(imu_prop);
    int rows = 0;
    for (const auto& id_data : data_.cameras_) {
      const int id = id_data.first;
      const CameraStatus& cam = id_data.second;
      delta_x.segment<6>(IMU_STATUS_DIM + rows) = cam.boxMinus(cam_prop.at(id));
      rows += 6;
    }
  }

  if (param_.verbose) {
    LOG(INFO) << "[" << stage << "-IKF] Iterator times : " << iter << ", coverage: " << (converage ? "True" : "False") << ", Loss from " << init_loss << " -> " << last_loss;
  }
  LOG(INFO) << "[" << stage << "-IKF] Gravity: " << imu_prop.g.transpose() << " -> " << data_.imu_status.g.transpose() << ", norm: " << data_.imu_status.g.norm();

  return true;
}

bool Msckf::featureJacobian(const Feature& ftr, const CameraWindow& cam_window, const set<int>& constraint_cam_id, 
    Eigen::MatrixXd& H_fj, Eigen::VectorXd& e_fj)
{
  if (ftr.status == FeatureStatus::NotInit) {
    return false;
  }

  Eigen::MatrixXd H_status = Eigen::MatrixXd::Zero(constraint_cam_id.size()*2, cam_window.size()*6 + IMU_STATUS_DIM);
  Eigen::MatrixXd H_Pfj    = Eigen::MatrixXd::Zero(constraint_cam_id.size()*2, 3);
  Eigen::VectorXd e        = Eigen::VectorXd::Zero(constraint_cam_id.size()*2);

  auto getJacobian = [](const Eigen::Vector3d& p_fj, const Eigen::Vector3d& P_fj, const CameraStatus& cam_status, 
                            Eigen::Matrix<double, 2, 6>& J_status, Eigen::Matrix<double, 2, 3>& J_Pfj) {
    
    Eigen::Vector3d p_fj_ = cam_status.Rwc.inverse()*(P_fj - cam_status.pwc);
    Eigen::Vector2d res(p_fj(0) - p_fj_(0)/p_fj_(2), p_fj(1) - p_fj_(1)/p_fj_(2));
    
    Eigen::Matrix<double, 2, 3> J_cvt;
    J_cvt << 1./p_fj_(2), 0, -p_fj_(0)/(p_fj_(2)*p_fj_(2)), 
             0, 1./p_fj_(2), -p_fj_(1)/(p_fj_(2)*p_fj_(2));

    Eigen::Matrix<double, 3, 6> J_status_;
    J_status_.block<3, 3>(0, 0) = MATH_UTILS::skewMatrix<double>(p_fj_);
    J_status_.block<3, 3>(0, 3) = -1*cam_status.Rwc.toRotationMatrix().transpose();
    J_status = J_cvt*J_status_;

    J_Pfj = J_cvt*cam_status.Rwc.toRotationMatrix().transpose();
    
    return res;
  };

  int i = 0;
  for (const auto& id_data : ftr.observes) {
    const int       cam_id = id_data.first;
    if (!constraint_cam_id.count(cam_id)) {
      continue;
    }

    Eigen::Vector3d cam_ob = id_data.second;
    const CameraStatus& cam_status = cam_window.at(cam_id);

    Eigen::Matrix<double, 2, 6> J_status;
    Eigen::Matrix<double, 2, 3> J_Pfj;
    
    Eigen::Vector2d res = getJacobian(cam_ob, ftr.point_3d, cam_status, J_status, J_Pfj);

    if (1) { // OC-KF
      Eigen::Matrix<double, 6, 1> u = Eigen::Matrix<double, 6, 1>::Zero();
      u.block<3, 1>(0, 0) = cam_status.Rwc_nullspace.toRotationMatrix().transpose()*data_.imu_status.g;
      u.block<3, 1>(3, 0) = MATH_UTILS::skewMatrix<double>(ftr.point_3d - cam_status.pwc_nullspace)*data_.imu_status.g;
      J_status = J_status - J_status*u*(u.transpose()*u).inverse()*u.transpose();
      J_Pfj    = -J_status.block<2, 3>(0, 3);
    }

    int cam_iter_cnt = std::distance(cam_window.begin(), cam_window.find(cam_id));

    H_status.block<2, 6>(i, cam_iter_cnt*6 + IMU_STATUS_DIM) = J_status;
    H_Pfj.block<2, 3>(i, 0) = J_Pfj;
    e.segment<2>(i) = res;

    i += 2;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_Pfj, Eigen::ComputeFullU | Eigen::ComputeThinV);
  Eigen::MatrixXd A = svd.matrixU().rightCols(2*constraint_cam_id.size() - 3);

  H_fj = A.transpose()*H_status;
  e_fj = A.transpose()*e;

  return true;
}

bool Msckf::findRedundanceCam(CameraWindow& cameras, vector<int>& remove_cam_id)
{
  
  auto key_cam_state_iter = cameras.end();
  for (int i = 0; i < 4; ++i)
    --key_cam_state_iter;
  
  auto cam_state_iter = key_cam_state_iter;
  ++cam_state_iter;
  
  auto first_cam_state_iter = cameras.begin();

  // Pose of the key camera state.
  const Eigen::Vector3d key_position = key_cam_state_iter->second.pwc;
  const Eigen::Matrix3d key_rotation = key_cam_state_iter->second.Rwc.toRotationMatrix();

  remove_cam_id.clear();
  for (int i = 0; i < 2; ++i) {
    const Eigen::Vector3d position = cam_state_iter->second.pwc;
    const Eigen::Matrix3d rotation = cam_state_iter->second.Rwc.toRotationMatrix();

    double distance = (position-key_position).norm();
    double angle = Eigen::AngleAxisd(rotation*key_rotation.transpose()).angle();

    if (   angle < param_.angle_threshold 
        && distance < param_.distance_threshold
        && track_rate_ > 0.5) {
      remove_cam_id.push_back(cam_state_iter->first);
      ++cam_state_iter;
    } 
    else {
      remove_cam_id.push_back(first_cam_state_iter->first);
      ++first_cam_state_iter;
    }
  }

  sort(remove_cam_id.begin(), remove_cam_id.end());

  return true;
}

bool Msckf::measurementUpdateStatus(Eigen::MatrixXd& H, Eigen::VectorXd& e)
{
  Eigen::MatrixXd K = kalmanGain(data_.P_dx, H, e);
  Eigen::VectorXd delta_x = K*e;
  Eigen::MatrixXd I_KF  = Eigen::MatrixXd::Identity(K.rows(), H.cols()) - K*H;
  Eigen::MatrixXd new_P = I_KF*data_.P_dx;
  data_.P_dx = (new_P + new_P.transpose()) / 2.0;

  const Eigen::VectorXd& delta_imu = delta_x.head(IMU_STATUS_DIM);
  if (   delta_imu.segment<3>(J_V).norm() > 0.5 
      || delta_imu.segment<3>(J_P).norm() > 1.0) {
    LOG(WARNING) << "[MeasureUpdate] Velocity update: " << delta_imu.segment<3>(J_V).transpose() << ". norm: " << delta_imu.segment<3>(J_V).norm();
    LOG(WARNING) << "[MeasureUpdate] Position update: " << delta_imu.segment<3>(J_P).transpose() << ". norm: " << delta_imu.segment<3>(J_P).norm();
    LOG(WARNING) << "[MeasureUpdate] Update delta is too large.";
  }
  
  data_.imu_status.boxPlus(delta_imu);

  int rows = 0;
  for (auto& id_data : data_.cameras_) {
    CameraStatus& cam_status = id_data.second;
    const Eigen::VectorXd& delta_cam = delta_x.segment<6>(IMU_STATUS_DIM + rows);
    cam_status.boxPlus(delta_cam);
    rows += 6;
  }

  return true;
}

bool Msckf::gatingTest(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, const int v)
{
  Eigen::MatrixXd P1 = H*data_.P_dx*H.transpose();
  Eigen::MatrixXd P2 = param_.noise_observation*Eigen::MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

  if (gamma < chi_square_distribution_[v]) {
    return true;
  } else {
    return false;
  }
}

Eigen::MatrixXd Msckf::kalmanGain(const Eigen::MatrixXd& P, Eigen::MatrixXd& H, Eigen::VectorXd& e)
{
#if 1
  if (H.rows() > H.cols()) {
    Eigen::SparseMatrix<double> H_sparse = H.sparseView();
    Eigen::SPQR<Eigen::SparseMatrix<double>> spqr_solver;
    spqr_solver.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_solver.compute(H_sparse);

    Eigen::MatrixXd H_dense;
    Eigen::VectorXd e_dense;
    (spqr_solver.matrixQ().transpose()*H).evalTo(H_dense);
    (spqr_solver.matrixQ().transpose()*e).evalTo(e_dense);

    H = H_dense.topRows(H.cols());
    e = e_dense.head(H.cols());
  }

  Eigen::MatrixXd S = H*P*H.transpose() + Eigen::MatrixXd::Identity(H.rows(), H.rows())*param_.noise_observation;
  return (S.ldlt().solve(H*P)).transpose();
#else
  Eigen::MatrixXd K;
  if (H.rows() > H.cols()) {
    // when measurement dimension is large than states
    // H : N * M
    // P : M * M
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(H.rows(), H.rows())*param_.noise_observation;
    Eigen::MatrixXd inv_P = P.inverse(); // M * M
    Eigen::MatrixXd HTRH = H.transpose() * R * H; // M * M
    Eigen::MatrixXd S = HTRH + inv_P; // M * M
    K = S.ldlt().solve(H.transpose() * R); // M * N
  }
  else {
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(H.rows(), H.rows())*param_.noise_observation;
    Eigen::MatrixXd S = H*P*H.transpose() + R;
    K = (S.ldlt().solve(H*P)).transpose(); // M * N
  }
  return K;
#endif
}

bool Msckf::updateStates(Eigen::VectorXd& delta_x)
{
  if (delta_x.rows() < IMU_STATUS_DIM) {
    LOG(ERROR) << "[MSCKF]: Update delta dimension is wrong, " << delta_x.rows();
    return false;
  }

  if (delta_x.hasNaN()) {
    LOG(ERROR) << "[MSCKF]: Update delta has nan, " << delta_x.transpose();
    return false;
  }

  // update state
  bool update = true;
  Eigen::VectorXd delta_imu = delta_x.head(IMU_STATUS_DIM);
  if (   delta_imu.segment<3>(J_V).norm() > 10.0 
      || delta_imu.segment<3>(J_P).norm() > 10.0) {
    LOG(WARNING) << "[MSCKF] Velocity update: " << delta_imu.segment<3>(J_V).transpose() << ". norm: " << delta_imu.segment<3>(J_V).norm();
    LOG(WARNING) << "[MSCKF] Position update: " << delta_imu.segment<3>(J_P).transpose() << ". norm: " << delta_imu.segment<3>(J_P).norm();
    LOG(WARNING) << "[MSCKF] Update delta is too large.";
    delta_imu.segment<3>(J_V) = Eigen::Vector3d::Zero();
    delta_imu.segment<3>(J_P) = Eigen::Vector3d::Zero(); 
    delta_imu.segment<3>(J_R) = Eigen::Vector3d::Zero(); // TODO: check
    update = false;
  }

  data_.imu_status.boxPlus(delta_imu);

  if (!update) {
    return false;
  }

  int rows = 0;
  for (auto& id_data : data_.cameras_) {
    const Eigen::VectorXd& delta_cam = delta_x.segment<6>(IMU_STATUS_DIM + rows);
    id_data.second.boxPlus(delta_cam);
    rows += 6;
  }

  return true;
}

bool Msckf::staticCorrect()
{
  Eigen::MatrixXd H(9, IMU_STATUS_DIM); H.setZero();
  H.block<3, 3>(0, J_R) = 0.5 * Eigen::Matrix3d::Identity();
  H.block<3, 3>(3, J_V).setIdentity();
  H.block<3, 3>(6, J_P).setIdentity();

  Eigen::VectorXd e(9); e.setZero();
  auto dq = data_.last_imu_status.Rwb * data_.imu_status.Rwb.inverse();
  e.segment<3>(0) = Eigen::Vector3d(dq.x(), dq.y(), dq.z());
  e.segment<3>(3) = -data_.imu_status.vwb;
  e.segment<3>(6) = data_.last_imu_status.pwb - data_.imu_status.pwb;

  Eigen::MatrixXd P = data_.P_dx.block<IMU_STATUS_DIM, IMU_STATUS_DIM>(0, 0);
  Eigen::MatrixXd S = H*P*H.transpose() + Eigen::MatrixXd::Identity(H.rows(), H.rows())*0.001f;
  Eigen::VectorXd K = (S.ldlt().solve(H*P)).transpose();

  Eigen::VectorXd dx = K*e;
  if (dx.hasNaN()) {
    LOG(ERROR) << "[STATIC CHECK] has nan: " << e.transpose();
    return false;
  }

  Eigen::MatrixXd I_KF  = Eigen::MatrixXd::Identity(K.rows(), H.cols()) - K*H;
  Eigen::MatrixXd new_P = I_KF*data_.P_dx;
  // data_.P_dx.block<IMU_STATUS_DIM, IMU_STATUS_DIM>(0, 0) = (new_P + new_P.transpose()) / 2.0;

  // Eigen::Quaterniond Rwb = data_.imu_status.Rwb * MATH_UTILS::rotateVecToQuaternion<double>(dx.segment<3>(J_R));
  // Eigen::Vector3d vwb = data_.imu_status.vwb + dx.segment<3>(J_V);
  // Eigen::Vector3d pwb = data_.imu_status.pwb + dx.segment<3>(J_P);

  // LOG(INFO) << "[MSCKF] Static update : " << dx.transpose();
  // LOG(INFO) << "[MSCKF] New Rwb : " << Rwb.coeffs().transpose();
  // LOG(INFO) << "[MSCKF] New Vwb : " << vwb.transpose();
  // LOG(INFO) << "[MSCKF] New Pwb : " << pwb.transpose();

  // data_.imu_status = data_.last_imu_status;
  data_.imu_status.Rwb = data_.last_imu_status.Rwb;
  data_.imu_status.pwb = data_.last_imu_status.pwb;
  data_.imu_status.vwb.setZero();
  data_.imu_status.ba  += dx.segment<3>(J_BA);
  data_.imu_status.bg  += dx.segment<3>(J_BG);

  // LOG(INFO) << "[MSCKF] New ba : " << data_.imu_status.ba.transpose();
  // LOG(INFO) << "[MSCKF] New bg : " << data_.imu_status.bg.transpose();

  return true;
}

} // namespace MSCKF
