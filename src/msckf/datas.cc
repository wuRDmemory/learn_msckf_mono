#include "msckf/datas.h"
#include "msckf/config.h"
#include "glog/logging.h"

namespace MSCKF {

namespace LOSE_FUNCTION {
double huber(const double e, const double epsilon) {
  if (e < epsilon) {
    return 1;
  }
  else {
    return sqrt(2.0*epsilon/e);
  }
}
};

std::ostream& operator<<(std::ostream& os, const ImuStatus& imu) {
  os << "Imu Status:" << endl;
  os << "|- qwb:    " << imu.Rwb.coeffs().transpose() << endl;
  os << "|- vwb:    " << imu.vwb.transpose() << endl;
  os << "|- pwb:    " << imu.pwb.transpose() << endl;
  os << "|- bias_a: " << imu.ba.transpose() << endl;
  os << "|- bias_w: " << imu.bg.transpose() << endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const CameraStatus& cam) {
  os << setprecision(15) 
     << cam.ts << ", " \
     << cam.pwc.x() << ", " << cam.pwc.y() << ", " << cam.pwc.z() << ", " \
     << cam.Rwc.w() << ", " << cam.Rwc.x() << ", " << cam.Rwc.y() << ", " << cam.Rwc.z();
  return os;
}

bool initialFeature(Feature& ftr, CameraWindow& cams) {
  if (ftr.status != FeatureStatus::NotInit) {
    return true;
  }

  // 1. init position, use last and recent measurement.
  const int c0_id = ftr.observes.begin()->first;
  const int c1_id = prev(ftr.observes.end())->first;

  const Eigen::Vector3d f_c0 = ftr.observes.begin()->second;
  const Eigen::Vector3d f_c1 = prev(ftr.observes.end())->second;

  if (!cams.count(c0_id) || !cams.count(c1_id)) {
    LOG(WARNING) << "This Should never happen.";
    return false;
  }

  Eigen::Quaterniond R_w_c0 = cams.at(c0_id).Rwc;
  Eigen::Vector3d    t_w_c0 = cams.at(c0_id).pwc;

  Eigen::Quaterniond R_w_c1 = cams.at(c1_id).Rwc;
  Eigen::Vector3d    t_w_c1 = cams.at(c1_id).pwc;

  Eigen::Quaterniond R_c1_c0 = R_w_c1.inverse()*R_w_c0;
  Eigen::Vector3d    t_c1_c0 = R_w_c1.inverse()*(t_w_c0 - t_w_c1);

  Eigen::Matrix<double, 3, 2> A;
  A << R_c1_c0*f_c0, -f_c1;

  Eigen::Vector2d depth = (A.transpose()*A).ldlt().solve(A.transpose()*t_c1_c0*-1);
  if (depth.hasNaN()) {
     return false;
  }

  Eigen::Vector3d w_point_c0 = R_w_c0*f_c0*depth(0) + t_w_c0;
  Eigen::Vector3d w_point_c1 = R_w_c1*f_c1*depth(1) + t_w_c1;
  Eigen::Vector3d init_postion = (w_point_c0 + w_point_c1)/2;

  // TODO: do BA-optimization with all measurement.
  int iter_cnt = 0;
  bool converge = false;
  
  Eigen::Vector3d old_x = init_postion; //(init_postion(0)/init_postion(2), init_postion(1)/init_postion(2), 1./init_postion(2));
  Eigen::Matrix3d JtJ;
  Eigen::Vector3d Jtb;
  double residual = 0;
  double lambda   = 1.e-3;

  for (const auto& cam_id_obs : ftr.observes) {
    const int cam_id = cam_id_obs.first;
    if (!cams.count(cam_id)) {
      continue;
    }

    Eigen::Vector2d res = evaluate(old_x, cams.at(cam_id), cam_id_obs.second, NULL);
    residual += res.squaredNorm();
  }

  do {
    JtJ.setZero();
    Jtb.setZero();

    for (const auto& cam_id_obs : ftr.observes) {
      const int cam_id = cam_id_obs.first;
      if (!cams.count(cam_id)) {
        continue;
      }

      Eigen::Matrix<double, 2, 3> Jacobian;
      Eigen::Vector2d res = evaluate(old_x, cams.at(cam_id), cam_id_obs.second, &Jacobian);
      double w = LOSE_FUNCTION::huber(res.norm(), 0.01);

      JtJ.noalias() += w*w*Jacobian.transpose()*Jacobian;
      Jtb.noalias() -= w*w*Jacobian.transpose()*res;
    }
    
    int   try_solve_cnt = 0;
    double new_residual = 0;
    do {
      Eigen::Vector3d delta = (lambda*Eigen::Matrix3d::Identity()+JtJ).ldlt().solve(Jtb);
      Eigen::Vector3d new_x = old_x + delta;
      converge = (delta.norm() < Config::feature_config.converge_threshold);

      new_residual = 0;
      for (const auto& cam_id_obs : ftr.observes) {
        const int cam_id = cam_id_obs.first;
        if (!cams.count(cam_id)) {
          continue;
        }

        Eigen::Vector2d res = evaluate(new_x, cams.at(cam_id), cam_id_obs.second, NULL);
        new_residual += res.squaredNorm();
      }

      if (new_residual < residual) {
        old_x    = new_x;
        residual = new_residual;

        lambda  *= 0.1;
        lambda   = max(lambda, 1.e-10);
      }
      else {
        // old_x = old_x;

        lambda *= 10;
        lambda  = min(lambda, 1.e10);
      }
    } while (try_solve_cnt++ < Config::feature_config.max_try_cnt && new_residual >= residual);

  } while (!converge && iter_cnt++ < Config::feature_config.max_iter_cnt);

  bool valid_feature = true;
  Eigen::Vector3d position = old_x;
  for (const auto& cam_id_obs : ftr.observes) {
    const int cam_id = cam_id_obs.first;
    if (!cams.count(cam_id)) {
      continue;
    }

    CameraStatus& cam_status = cams[cam_id];
    Eigen::Vector3d p = cam_status.Rwc.inverse()*(position - cam_status.pwc);
    if (p.z() < 0) {
      valid_feature = false;
      break;
    }
  }

  ftr.point_3d = position;
  ftr.status   = valid_feature ? FeatureStatus::Inited : FeatureStatus::NotInit;

  return valid_feature;
}

bool checkMotion(Feature& ftr, CameraWindow& cams) {
  if (ftr.status != FeatureStatus::NotInit) {
    return true;
  }

  const int c0_id = ftr.observes.begin()->first;
  const int c1_id = prev(ftr.observes.end())->first;

  const Eigen::Vector3d f_c0 = (ftr.observes.begin()->second).normalized();
  const Eigen::Vector3d f_c1 = (prev(ftr.observes.end())->second).normalized();

  if (!cams.count(c0_id) || !cams.count(c1_id)) {
    CHECK(false) << "[CheckMotion] This Should never happen.";
    return false;
  }

  Eigen::Quaterniond R_w_c0 = cams.at(c0_id).Rwc;
  Eigen::Vector3d    t_w_c0 = cams.at(c0_id).pwc;

  Eigen::Quaterniond R_w_c1 = cams.at(c1_id).Rwc;
  Eigen::Vector3d    t_w_c1 = cams.at(c1_id).pwc;

  Eigen::Quaterniond R_c0_c1 = R_w_c0.inverse()*R_w_c1;

  const Eigen::Vector3d c0_f_c1 = (R_c0_c1*f_c1).normalized();

  // 1. check direction's angle
  // double cos_angle = abs(c0_f_c1.dot(f_c0));
  // if (cos_angle > Config::feature_config.min_disparity_angle) {
  //   return false;
  // }

  // 2. check distance
  Eigen::Vector3d t_c0_c1 = R_w_c0.inverse()*(t_w_c1 - t_w_c0); // project V_c0_c1 in world cooridation to c0's cooridation.
  Eigen::Vector3d orthogonal_vec = t_c0_c1 - t_c0_c1.dot(f_c0)*f_c0;
  double orthogonal_distance = orthogonal_vec.norm();
  // LOG(INFO) << "cam " << c0_id << "->" << c1_id << " orthogonal distance " << orthogonal_distance;
  return orthogonal_distance >= Config::feature_config.min_disparity_distance;
}

Eigen::Vector2d 
evaluate(const Eigen::Vector3d& Pj_w, const CameraStatus& T_ci, const Eigen::Vector3d& obs_ci, Eigen::Matrix<double, 2, 3>* jacobian) {
  const Eigen::Quaterniond R_w_ci = T_ci.Rwc;
  const Eigen::Vector3d    t_w_ci = T_ci.pwc;

  Eigen::Vector3d Pj_ci = R_w_ci.inverse()*(Pj_w - t_w_ci);
  Eigen::Vector3d pj_ci = Pj_ci/Pj_ci(2);

  Eigen::Vector2d residual = pj_ci.head<2>() - obs_ci.head<2>();

  if (jacobian) {
    Eigen::Matrix<double, 2, 3> J_p_P;
    J_p_P << 1./Pj_ci(2), 0, -Pj_ci(0)/(Pj_ci(2)*Pj_ci(2)),
             0, 1./Pj_ci(2), -Pj_ci(1)/(Pj_ci(2)*Pj_ci(2));

    *jacobian = J_p_P*R_w_ci.inverse().toRotationMatrix();
  }

  return residual;
}



}
