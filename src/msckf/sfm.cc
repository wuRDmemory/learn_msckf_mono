#include "sfm.h"
#include "math_utils.h"

namespace MSCKF {

namespace LOSE_FUNCTION {
double huber(const double e, const double epsilon)
{
  if (e < epsilon) {
    return 1;
  } else {
    return sqrt(2.0*epsilon/e);
  }
}
};

bool SFM::tryToInit(Feature& ftr, CameraWindow& cams, Eigen::Vector3d& position)
{
  // Our linear system matrices
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();

  const int anchor_cam_id = ftr.observes.begin()->first;
  const Eigen::Quaterniond Rwa = cams.at(anchor_cam_id).Rwc;
  const Eigen::Vector3d pwa = cams.at(anchor_cam_id).pwc;

  for (const auto& pair : ftr.observes) {
    const int cam_id = pair.first;
    // if (cam_id == anchor_cam_id) {
    //   continue;
    // }

    Eigen::Vector3d fc = pair.second;
    Eigen::Matrix3d Rac = (Rwa.inverse() * cams.at(cam_id).Rwc).toRotationMatrix();
    Eigen::Vector3d pac = Rwa.inverse()*(cams.at(cam_id).pwc - pwa);
    Eigen::Vector3d afc = (Rac * fc).normalized();
    
    Eigen::Matrix3d Nafc = MATH_UTILS::skewMatrix(afc);
    Eigen::Matrix3d NNT = Nafc.transpose()*Nafc;

    A.noalias() += NNT;
    b.noalias() += NNT * pac;
  }

  // Solve the linear system
  Eigen::Vector3d p_f = A.colPivHouseholderQr().solve(b);

  // Check A and p_f
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Vector3d singularValues = svd.singularValues();
  double condA = singularValues(0) / singularValues(2);
  if (   std::abs(condA) > param_.max_cond_number 
      || p_f(2, 0) < param_.min_dist 
      || p_f(2, 0) > param_.max_dist 
      || std::isnan(p_f.norm())) {
      LOG(WARNING) << "cond number: " << condA;
      LOG(WARNING) << "point: " << p_f.transpose();
    return false;
  }

  // Store it in our feature object
  position = Rwa.toRotationMatrix()*p_f + pwa;
  return true;
}

bool SFM::initialFeature(Feature& ftr, CameraWindow& cams)
{
  if (ftr.status != FeatureStatus::NotInit) {
    return true;
  }

  // 1. init position, use last and recent measurement.
  const int c0_id = ftr.observes.begin()->first;
  const int c1_id = prev(ftr.observes.end())->first;

  const Eigen::Vector3d f_c0 = ftr.observes.begin()->second;
  const Eigen::Vector3d f_c1 = prev(ftr.observes.end())->second;

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

  Eigen::Vector3d init_postion1(0, 0, 0);
  if (!tryToInit(ftr, cams, init_postion1)) {
    ;
  }

  int iter_cnt = 0;
  bool converge = false;
  
  Eigen::Vector3d old_x(init_postion(0)/init_postion(2), init_postion(1)/init_postion(2), 1./init_postion(2));
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
    
    int try_solve_cnt = 0;
    double new_residual = 0;
    do {
      Eigen::Vector3d delta = (lambda*Eigen::Matrix3d::Identity()+JtJ).ldlt().solve(Jtb);
      Eigen::Vector3d new_x = old_x + delta;
      converge = (delta.norm() < param_.converge_threshold);

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
        lambda *= 10;
        lambda  = min(lambda, 1.e10);
      }
    } while (try_solve_cnt++ < param_.max_try_cnt && new_residual >= residual);

  } while (!converge && iter_cnt++ < param_.max_iter_cnt);

  bool valid_feature = true;
  Eigen::Vector3d position(old_x.x()/old_x.z(), old_x.y()/old_x.z(), 1/old_x.z());
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

  // if (param_.verbose) {
    LOG(INFO) << "init postion: " << init_postion.transpose();
    LOG(INFO) << "init1 postion: " << init_postion1.transpose();
    LOG(INFO) << "opti postion: " << position.transpose();
    LOG(INFO) << "point valid: " << valid_feature;
    // TODO: report some optimization information like ceres
  // }

  return valid_feature;
}

bool SFM::checkMotion(Feature& ftr, CameraWindow& cams)
{
  if (ftr.status != FeatureStatus::NotInit) {
    return true;
  }

  const int c0_id = ftr.observes.begin()->first;
  const int c1_id = prev(ftr.observes.end())->first;

  const Eigen::Vector3d f_c0 = (ftr.observes.begin()->second).normalized();
  const Eigen::Vector3d f_c1 = (prev(ftr.observes.end())->second).normalized();

  Eigen::Quaterniond R_w_c0 = cams.at(c0_id).Rwc;
  Eigen::Vector3d    t_w_c0 = cams.at(c0_id).pwc;

  Eigen::Quaterniond R_w_c1 = cams.at(c1_id).Rwc;
  Eigen::Vector3d    t_w_c1 = cams.at(c1_id).pwc;

  Eigen::Quaterniond R_c0_c1 = R_w_c0.inverse()*R_w_c1;

  const Eigen::Vector3d c0_f_c1 = (R_c0_c1*f_c1).normalized();

  // 1. check direction's angle
  double cos_angle = abs(c0_f_c1.dot(f_c0));
  if (cos_angle > param_.min_disparity_angle) {
    return false;
  }

  // 2. check distance
  Eigen::Vector3d t_c0_c1 = R_w_c0.inverse()*(t_w_c1 - t_w_c0); // project V_c0_c1 in world cooridation to c0's cooridation.
  Eigen::Vector3d orthogonal_vec = t_c0_c1 - t_c0_c1.dot(f_c0)*f_c0;
  double orthogonal_distance = orthogonal_vec.norm();
  // LOG(INFO) << "cam " << c0_id << "->" << c1_id << " orthogonal distance " << orthogonal_distance;
  return orthogonal_distance >= param_.min_disparity_distance;
}

Eigen::Vector2d SFM::evaluate(const Eigen::Vector3d& Pj_w, const CameraStatus& T_ci, 
    const Eigen::Vector3d& obs_ci, Eigen::Matrix<double, 2, 3>* jacobian)
{
  const Eigen::Quaterniond R_w_ci = T_ci.Rwc;
  const Eigen::Vector3d    t_w_ci = T_ci.pwc;

  const double alpha = Pj_w.x();
  const double beta  = Pj_w.y();
  const double rho   = Pj_w.z();

  const Eigen::Matrix3d R_ci_w = R_w_ci.toRotationMatrix().transpose();
  const Eigen::Vector3d t_ci_w = -R_ci_w*t_w_ci;

  Eigen::Vector3d Pj_ci = R_ci_w*Eigen::Vector3d(alpha, beta, 1) + rho*t_ci_w;
  Eigen::Vector3d pj_ci = Pj_ci/Pj_ci(2);

  Eigen::Vector2d residual = pj_ci.head<2>() - obs_ci.head<2>();

  if (jacobian) {
    Eigen::Matrix<double, 2, 3> J_p_P;
    J_p_P << 1./Pj_ci(2), 0, -Pj_ci(0)/(Pj_ci(2)*Pj_ci(2)),
             0, 1./Pj_ci(2), -Pj_ci(1)/(Pj_ci(2)*Pj_ci(2));

    Eigen::Matrix3d J_P_T;
    J_P_T.leftCols(2)  = R_ci_w.leftCols(2);
    J_P_T.rightCols(1) = t_ci_w;
    *jacobian = J_p_P*J_P_T;
  }

  return residual;
}

} // namespace MSCKF