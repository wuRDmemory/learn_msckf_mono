#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Core"
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "ceres/ceres.h"
#include "../src/msckf/common.h"
#include "../src/msckf/datas.h"
#include "../src/msckf/math_utils.h"

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        Eigen::Map<const Eigen::Quaterniond> _q(x);
        Eigen::Map<const Eigen::Vector3d> _dq(delta);
        Eigen::Quaterniond dq = MATH_UTILS::rotateVecToQuaternion<double>(_dq);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta);
        q = (_q * dq).normalized();
        return true;
    }

    virtual bool ComputeJacobian(const double *x, double *jacobian) const
    {
      Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
      j.topRows<3>().setIdentity();
      j.bottomRows<1>().setZero();

      return true;
    }
    virtual int GlobalSize() const { return 4; };
    virtual int LocalSize() const { return 3; };
};

class LineFeatureProjFactor {
private:
  Eigen::Vector3d sp_, ep_;
  Eigen::Vector3d norm_vec_, direct_vec_;
  Eigen::Matrix3d K_;
  Eigen::Matrix3d KK_;

public:
  LineFeatureProjFactor(const Eigen::Vector3d sp, const Eigen::Vector3d ep, 
    const Eigen::Vector3d norm_vec, const Eigen::Vector3d direct_vec, const Eigen::Matrix3d K)
      : sp_(sp), ep_(ep), norm_vec_(norm_vec), direct_vec_(direct_vec), K_(K) {
        const double fx = K_(0, 0), cx = K_(0, 2);
        const double fy = K_(1, 1), cy = K_(1, 2);
        KK_ << fy, 0, 0, 0, fx, 0, -fy*cx, -fx*cy, fx*fy;
      }

  template <typename T>
  bool operator()(const T* const rotate, const T* const trans, T* residuals) const {
    Eigen::Matrix<T, 3, 3> Rwc = Eigen::Quaternion<T>(rotate[3], rotate[0], rotate[1], rotate[2]).toRotationMatrix();
    Eigen::Matrix<T, 3, 1> twc{trans[0], trans[1], trans[2]};

    Eigen::Matrix<T, 3, 3> Rcw = Rwc.transpose();
    Eigen::Matrix<T, 3, 3> twc_x = MATH_UTILS::skewMatrix<T>(twc);

    Eigen::Matrix<T, 3, 1> norm_vec_c = Rcw * norm_vec_.cast<T>() + Rcw * twc_x * direct_vec_.cast<T>();
    Eigen::Matrix<T, 3, 1> direct_vec_c = Rcw * direct_vec_.cast<T>();

    Eigen::Matrix<T, 3, 1> im_line = KK_ * norm_vec_c;
    // std::cout << "line: " << im_line.transpose() << std::endl;
    T im_line_norm = ceres::sqrt(im_line(0)*im_line(0) + im_line(1)*im_line(1));

    residuals[0] = (im_line.transpose() * sp_.cast<T>())(0, 0) / im_line_norm;
    residuals[1] = (im_line.transpose() * ep_.cast<T>())(0, 0) / im_line_norm;

    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const Eigen::Vector3d sp, const Eigen::Vector3d ep, 
      const Eigen::Vector3d norm_vec, const Eigen::Vector3d direct_vec, const Eigen::Matrix3d K) {
     return (new ceres::AutoDiffCostFunction<LineFeatureProjFactor, 2, 4, 3>(
                 new LineFeatureProjFactor(sp, ep, norm_vec, direct_vec, K)));
   }
};

class LineFeatureFactor : public ceres::SizedCostFunction<2, 4, 3>
{
protected:
  Eigen::Vector3d sp_, ep_;
  Eigen::Vector3d norm_vec_, direct_vec_;
  Eigen::Matrix3d K_;
  Eigen::Matrix3d KK_;

public:
  LineFeatureFactor(const Eigen::Vector3d sp, const Eigen::Vector3d ep, 
    const Eigen::Vector3d norm_vec, const Eigen::Vector3d direct_vec, const Eigen::Matrix3d K) 
      : sp_(sp), ep_(ep), norm_vec_(norm_vec), direct_vec_(direct_vec), K_(K) {
    const double fx = K_(0, 0), cx = K_(0, 2);
    const double fy = K_(1, 1), cy = K_(1, 2);
    KK_ << fy, 0, 0, 0, fx, 0, -fy*cx, -fx*cy, fx*fy;
  }

  virtual bool Evaluate(double const *const *params, double *residuals, double **jacobians) const
  {
    const Eigen::Quaterniond qwc(params[0][3], params[0][0], params[0][1], params[0][2]);
    const Eigen::Vector3d twc(params[1][0], params[1][1], params[1][2]);
    Eigen::Matrix3d Rwc = qwc.toRotationMatrix();

    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Matrix3d twc_x = MATH_UTILS::skewMatrix<double>(twc);

    Eigen::Vector3d norm_vec_c = Rcw * norm_vec_ + Rcw * twc_x * direct_vec_;
    Eigen::Vector3d direct_vec_c = Rcw * direct_vec_;

    Eigen::Vector3d im_line = KK_ * norm_vec_c;
    double im_line_norm = std::sqrt(im_line(0)*im_line(0) + im_line(1)*im_line(1));

    residuals[0] = (im_line.transpose() * sp_)(0, 0) / im_line_norm;
    residuals[1] = (im_line.transpose() * ep_)(0, 0) / im_line_norm;

    if (!jacobians) {
      return true;
    }

    double im_line_norm2 = im_line(0)*im_line(0) + im_line(1)*im_line(1);
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_r_imL; J_r_imL.setZero();
    J_r_imL.row(0) << -(im_line(0)*(im_line.transpose()*sp_)(0, 0))/im_line_norm2 + sp_(0), -(im_line(1)*(im_line.transpose()*sp_)(0, 0))/im_line_norm2 + sp_(1), 1;
    J_r_imL.row(1) << -(im_line(0)*(im_line.transpose()*ep_)(0, 0))/im_line_norm2 + ep_(0), -(im_line(1)*(im_line.transpose()*ep_)(0, 0))/im_line_norm2 + ep_(1), 1;
    J_r_imL /= im_line_norm;

    Eigen::Matrix<double, 3, 6> J_imL_camL; J_imL_camL.setZero(); 
    J_imL_camL.leftCols(3) = KK_; 

    if (jacobians[0]) {
      Eigen::Matrix<double, 6, 3> J_camL_rot; J_camL_rot.setZero();
      J_camL_rot.topRows(3) = MATH_UTILS::skewMatrix<double>(norm_vec_c);
      J_camL_rot.bottomRows(3) = MATH_UTILS::skewMatrix<double>(direct_vec_c);

      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jacobians[0]); J.setZero();
      J.leftCols(3) = J_r_imL * J_imL_camL * J_camL_rot;
    }

    if (jacobians[1]) {
      Eigen::Matrix<double, 6, 3> J_camL_tran; J_camL_tran.setZero();
      J_camL_tran.topRows(3) = Rcw * MATH_UTILS::skewMatrix<double>(direct_vec_) * -1.0;
      J_camL_tran.bottomRows(3).setZero();

      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
      J = J_r_imL * J_imL_camL * J_camL_tran;
    }
    
    return true;
  }
};

double sampleUniformDouble( double start, double end )
{
  static boost::mt19937 rng( static_cast< unsigned >( std::time( 0 ) ) );
  boost::uniform_real< > uni_dist( start, end );
  return uni_dist( rng );
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> 
fakeMeasure(const Eigen::Vector3d& sP, const Eigen::Vector3d& eP, Eigen::Quaterniond &Rwb, Eigen::Vector3d& pwb, Eigen::Matrix3d& K)
{
  // start point
  double start_n = -0.5;
  double end_n   =  0.5;
  Eigen::Vector3d nsp = Rwb.inverse()*(sP - pwb); nsp /= nsp(2);
  Eigen::Vector3d sp = K*nsp; // + Eigen::Vector3d{sampleUniformDouble(start_n, end_n), sampleUniformDouble(start_n, end_n), 0};

  // end point
  Eigen::Vector3d nep = Rwb.inverse()*(eP - pwb); nep /= nep(2);
  Eigen::Vector3d ep = K*nep; // + Eigen::Vector3d{sampleUniformDouble(start_n, end_n), sampleUniformDouble(start_n, end_n), 0};
  
  return std::make_pair(sp, ep);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d>
getLineParam(const Eigen::Vector3d& sP, const Eigen::Vector3d& eP)
{
  Eigen::Vector3d direct_vec = (eP - sP).normalized();
  Eigen::Vector3d norm_vec   = (eP.cross(sP)).normalized();
  const double d = (eP.cross(sP)).norm() / (eP - sP).norm();
  norm_vec *= d;

  return std::make_pair(norm_vec, direct_vec);
}

TEST(LineFeatureTest, LineFeatureOpt)
{
  /* true position of line feature and camera */
  // Line Feature's ends
  Eigen::Vector3d sP1{-2, 0, 3};
  Eigen::Vector3d eP1{ 2, 0, 3};
  
  Eigen::Vector3d sP2{-2, 1, 3};
  Eigen::Vector3d eP2{ 2, -2, 5};
  
  Eigen::Vector3d sP3{-2, -4, 5};
  Eigen::Vector3d eP3{ 2,  3, 3};

  // c1 pose
  Eigen::Quaterniond rot0{1, 0, 0, 0};
  Eigen::Vector3d tran0 = Eigen::Vector3d::Zero();

  // c2 pose
  Eigen::Quaterniond rot1{1, 0, 0, 0};
  Eigen::Vector3d tran1{0, -0.8, 0};

  // c3 pose
  Eigen::Quaterniond rot2 = MATH_UTILS::eulerToQuaternion<double>(Eigen::Vector3d{10, 0, 0} * MSCKF::Deg2Rad);
  Eigen::Vector3d tran2{0, -0.4, 0};

  // intrinsic
  Eigen::Matrix3d K;
  K << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1.0;

  // line feature parameter
  auto line1 = getLineParam(sP1, eP1);
  auto line2 = getLineParam(sP2, eP2);
  auto line3 = getLineParam(sP3, eP3);

  {
    // measurements
    std::pair<Eigen::Vector3d, Eigen::Vector3d> m1 = fakeMeasure(sP1, eP1, rot1, tran1, K);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> m2 = fakeMeasure(sP2, eP2, rot1, tran1, K);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> m3 = fakeMeasure(sP3, eP3, rot1, tran1, K);

    /* estimate */
    double est_rot[] = {0, 0, 0, 1};
    double est_tran[] = {0, 0, 0};

    ceres::Problem problem;
    ceres::LocalParameterization* quat_param = new PoseLocalParameterization();
    problem.AddParameterBlock(est_rot, 4, quat_param);
    problem.AddParameterBlock(est_tran, 3);
    problem.AddResidualBlock(new LineFeatureFactor(m1.first, m1.second, line1.first, line1.second, K), nullptr, est_rot, est_tran);
    problem.AddResidualBlock(new LineFeatureFactor(m2.first, m2.second, line2.first, line2.second, K), nullptr, est_rot, est_tran);
    problem.AddResidualBlock(new LineFeatureFactor(m3.first, m3.second, line3.first, line3.second, K), nullptr, est_rot, est_tran);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;  

    Eigen::Map<Eigen::Quaterniond> est_rot1(est_rot);
    Eigen::Map<Eigen::Vector3d> est_tran1(est_tran);
    Eigen::Vector3d delta_rot = MATH_UTILS::quaternionToRotateVector<double>(est_rot1.inverse()*rot1);
    Eigen::Vector3d delta_trans = est_tran1 - tran1;

    std::cout << "delta rot : " << delta_rot.norm() << " , axis : " << delta_rot.normalized().transpose() << std::endl;
    std::cout << "delta trans : " << delta_trans.norm() << std::endl;

    EXPECT_LE(delta_rot.norm(), 3 * MSCKF::Deg2Rad);
    EXPECT_LE(delta_trans.norm(), 0.3);
  }

  {
    // measurements
    std::pair<Eigen::Vector3d, Eigen::Vector3d> m1 = fakeMeasure(sP1, eP1, rot2, tran2, K);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> m2 = fakeMeasure(sP2, eP2, rot2, tran2, K);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> m3 = fakeMeasure(sP3, eP3, rot2, tran2, K);

    /* estimate */
    double est_rot[] = {0, 0, 0, 1};
    double est_tran[] = {0, 0, 0};

    ceres::Problem problem;
    ceres::LocalParameterization* quat_param = new PoseLocalParameterization();
    problem.AddParameterBlock(est_rot, 4, quat_param);
    problem.AddParameterBlock(est_tran, 3);
    problem.AddResidualBlock(new LineFeatureFactor(m1.first, m1.second, line1.first, line1.second, K), nullptr, est_rot, est_tran);
    problem.AddResidualBlock(new LineFeatureFactor(m2.first, m2.second, line2.first, line2.second, K), nullptr, est_rot, est_tran);
    problem.AddResidualBlock(new LineFeatureFactor(m3.first, m3.second, line3.first, line3.second, K), nullptr, est_rot, est_tran);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;  

    Eigen::Map<Eigen::Quaterniond> est_rot2(est_rot); est_rot2.normalize();
    Eigen::Map<Eigen::Vector3d> est_tran2(est_tran);
    Eigen::Vector3d delta_rot = MATH_UTILS::quaternionToRotateVector<double>(est_rot2.inverse()*rot2);
    Eigen::Vector3d delta_trans = est_tran2 - tran2;

    std::cout << "delta rot : " << delta_rot.norm() << " , axis : " << delta_rot.normalized().transpose() << std::endl;
    std::cout << "delta trans : " << delta_trans.norm() << std::endl;

    EXPECT_LE(delta_rot.norm(), 3 * MSCKF::Deg2Rad);
    EXPECT_LE(delta_trans.norm(), 0.3);
  }
}
