#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "helpers/fake_pose.h"

struct LineFeature {
  cv::Point2f sp;
  cv::Point2f ep;
};

class LineFeatureTest : public ::testing::Test {
protected:
  Eigen::Vector3d sP_, eP_;
  Eigen::Quaterniond Rcb_;
  Eigen::Vector3d pcb_;
  std::unique_ptr<TESTING::FakePose> fake_pose_ptr_;

public:
  LineFeatureTest() = default;
  ~LineFeatureTest() override {}

  void SetUp() override {
    // fake pose
    // Eigen::Quaterniond start_q = Eigen::Quaterniond::Identity();
    // Eigen::Vector3d    start_p = Eigen::Vector3d::Zero();
    // Eigen::Quaterniond end_q   = MATH_UTILS::eulerToQuaternion(Eigen::Vector3d{10*MSCKF::Deg2Rad, 20*MSCKF::Deg2Rad, 15*MSCKF::Deg2Rad});
    // Eigen::Vector3d    end_p   = Eigen::Vector3d{0.0, -5.0, 0.0};
    // fake_pose_ptr_.reset(new TESTING::FakePose(6, start_p, end_p, start_q, end_q));

    // fake line feature
    sP_ = Eigen::Vector3d{-2, 0, 3};
    eP_ = Eigen::Vector3d{ 2, 0, 5};

    // fake extrinsic
    Rcb_ = MATH_UTILS::eulerToQuaternion(Eigen::Vector3d{-M_PI/2, 0, -M_PI/2}); //Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d{1, 2, 3}, Eigen::Vector3d{-2, -3, 1});
    pcb_ = Eigen::Vector3d::Zero();

    // std::cout << "Rotate [1, 0, 0]: " << (Rcb_.inverse()*Eigen::Vector3d{1, 0, 0}).transpose() << std::endl;
    // std::cout << "Rotate [0, 1, 0]: " << (Rcb_.inverse()*Eigen::Vector3d{0, 1, 0}).transpose() << std::endl;
    // std::cout << "Rotate [0, 0, 1]: " << (Rcb_.inverse()*Eigen::Vector3d{0, 0, 1}).transpose() << std::endl;
  }

  void TearDown() override {
    fake_pose_ptr_.reset(nullptr);
  }

  double sampleUniformDouble( double start, double end ) const {
      static boost::mt19937 rng( static_cast< unsigned >( std::time( 0 ) ) );
      boost::uniform_real< > uni_dist( start, end );
      return uni_dist( rng );
  }

  std::pair<Eigen::Vector3d, Eigen::Vector3d> fakeMeasure(Eigen::Quaterniond &Rwb, Eigen::Vector3d& pwb) const
  {

    // start point
    double start_n = -0.01;
    double end_n = 0.01;
    Eigen::Vector3d nSP = sP_ + Eigen::Vector3d{sampleUniformDouble(start_n, end_n), sampleUniformDouble(start_n, end_n), sampleUniformDouble(start_n, end_n)};
    Eigen::Vector3d sp = Rwb.inverse()*(nSP - pwb);
    sp /= sp(2);

    // end point
    Eigen::Vector3d nEP = eP_ + Eigen::Vector3d{sampleUniformDouble(start_n, end_n), sampleUniformDouble(start_n, end_n), sampleUniformDouble(start_n, end_n)};
    Eigen::Vector3d ep = Rwb.inverse()*(nEP - pwb);
    ep /= ep(2);

    return std::make_pair(sp, ep);
  }

  void initFakePose(const int N, const Eigen::Vector3d& end_euler, const Eigen::Vector3d& end_trans, 
    const Eigen::Vector3d& start_euler = Eigen::Vector3d::Zero(), const Eigen::Vector3d& start_trans = Eigen::Vector3d::Zero())
  {
    Eigen::Quaterniond start_q = MATH_UTILS::eulerToQuaternion<double>(start_euler * MSCKF::Deg2Rad);
    Eigen::Vector3d    start_p = start_trans;
    Eigen::Quaterniond end_q   = MATH_UTILS::eulerToQuaternion<double>(end_euler * MSCKF::Deg2Rad);
    Eigen::Vector3d    end_p   = end_trans;
    fake_pose_ptr_.reset(new TESTING::FakePose(N, start_p, end_p, start_q, end_q));
  }
};

TEST_F(LineFeatureTest, LineInitFeature)
{
  /* ground truth */
  Eigen::Vector3d true_direct_vec = (eP_ - sP_).normalized();
  Eigen::Vector3d true_norm_vec = eP_.normalized().cross(sP_.normalized());
  const double true_d = (eP_.cross(sP_)).norm() / (eP_ - sP_).norm();

  /* estimate */
  initFakePose(5, Eigen::Vector3d{10, 20, 15}, Eigen::Vector3d{0, -5, 0});
  std::vector<Eigen::Quaterniond> rotates = fake_pose_ptr_->getRotate();
  std::vector<Eigen::Vector3d>    transs  = fake_pose_ptr_->getTranslate();

  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> line_measures;
  for (size_t i = 0; i < rotates.size(); ++i) {
    auto rotate  = rotates[i];
    auto trans   = transs[i];
    auto measure = fakeMeasure(rotate, trans);
    line_measures.emplace_back(measure);
  }

  // initial line feature's orientation
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(static_cast<int>(line_measures.size()), 3);
  for (size_t i = 0; i < line_measures.size(); ++i) {
    // calculate normal vector
    auto line_end = line_measures[i];
    Eigen::Vector3d sp = line_end.first;
    Eigen::Vector3d ep = line_end.second;
    Eigen::Vector3d norm_vec = sp.cross(ep);
    norm_vec.normalize();

    // calculate A matrix
    Eigen::Quaterniond Rcic0 = rotates[i].inverse()*rotates[0];
    A.row(i) = norm_vec.transpose() * Rcic0.toRotationMatrix();
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeFullV);
  std::cout << "Normal case: " << svd.singularValues().transpose() << std::endl;
  Eigen::VectorXd direct_vec = svd.matrixV().rightCols(1);
  direct_vec.normalize();

  const double res_angle = acos(direct_vec.dot(true_direct_vec));
  std::cout << "Residual angle: " << res_angle * MSCKF::Rad2Deg << " Deg." << std::endl;
  EXPECT_LE(res_angle, 1.0 * MSCKF::Deg2Rad);

  // init line feature's distance
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(static_cast<int>(line_measures.size()), 1);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(static_cast<int>(line_measures.size()), 1);
  Eigen::Vector3d direct_vec0 = direct_vec;
  Eigen::Vector3d norm_vec0;
  for (size_t i = 0; i < line_measures.size(); ++i) {
    // calculate normal vector
    auto line_end = line_measures[i];
    Eigen::Vector3d sp = line_end.first;
    Eigen::Vector3d ep = line_end.second;
    Eigen::Vector3d norm_vec = sp.cross(ep);
    norm_vec.normalize();

    if (i == 0) {
      norm_vec0 = norm_vec;
    }

    // rotate
    Eigen::Quaterniond Rc0ci = rotates[0].inverse()*rotates[i];
    Eigen::Vector3d tc0ci = rotates[0].inverse()*(transs[i] - transs[0]);
    Eigen::Vector3d norm_vec0_i = Rc0ci*norm_vec;
    Eigen::Vector3d bi = direct_vec0.cross(norm_vec0_i);

    B.row(i) = bi.transpose() * norm_vec0;
    C.row(i) = bi.transpose() * tc0ci.cross(direct_vec0);
  }
  const double d = (B.transpose() * C)(0, 0) / (B.transpose() * B)(0, 0);
  std::cout << "Distance: " << d << ", True distance: " << true_d << std::endl;
  EXPECT_LE(std::abs(d - true_d), 0.1);
}

TEST_F(LineFeatureTest, MotionDegenerateCase)
{
  /* ground truth */
  Eigen::Vector3d true_direct_vec = (eP_ - sP_).normalized();
  Eigen::Vector3d true_norm_vec = eP_.normalized().cross(sP_.normalized());
  const double true_d = (eP_.cross(sP_)).norm() / (eP_ - sP_).norm();

  /* estimate */
  initFakePose(10, Eigen::Vector3d{0, 0, 0}, Eigen::Vector3d{0, 0, -4});
  std::vector<Eigen::Quaterniond> rotates = fake_pose_ptr_->getRotate();
  std::vector<Eigen::Vector3d>    transs  = fake_pose_ptr_->getTranslate();
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> line_measures;
  for (size_t i = 0; i < rotates.size(); ++i) {
    auto rotate  = rotates[i];
    auto trans   = transs[i];
    auto measure = fakeMeasure(rotate, trans);
    line_measures.emplace_back(measure);
  }

  // initial line feature's orientation
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(static_cast<int>(line_measures.size()), 3);
  for (size_t i = 0; i < line_measures.size(); ++i) {
    // calculate normal vector
    auto line_end = line_measures[i];
    Eigen::Vector3d sp = line_end.first;
    Eigen::Vector3d ep = line_end.second;
    Eigen::Vector3d norm_vec = sp.cross(ep);
    norm_vec.normalize();

    // calculate A matrix
    Eigen::Quaterniond Rcic0 = rotates[i].inverse()*rotates[0];
    A.row(i) = norm_vec.transpose() * Rcic0.toRotationMatrix();
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeFullV);
  Eigen::VectorXd singulars = svd.singularValues();
  std::cout << "Degenerate case: " << singulars.transpose() << std::endl;
  EXPECT_LE(singulars(1), 0.01);
}

TEST_F(LineFeatureTest, RotateDegenerateCase)
{
  /* ground truth */
  Eigen::Vector3d true_direct_vec = (eP_ - sP_).normalized();
  Eigen::Vector3d true_norm_vec = eP_.normalized().cross(sP_.normalized());
  const double true_d = (eP_.cross(sP_)).norm() / (eP_ - sP_).norm();

  /* estimate */
  initFakePose(10, Eigen::Vector3d{10, 20, 15}, Eigen::Vector3d{0, 0, 0});
  std::vector<Eigen::Quaterniond> rotates = fake_pose_ptr_->getRotate();
  std::vector<Eigen::Vector3d>    transs  = fake_pose_ptr_->getTranslate();
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> line_measures;
  for (size_t i = 0; i < rotates.size(); ++i) {
    auto rotate  = rotates[i];
    auto trans   = transs[i];
    auto measure = fakeMeasure(rotate, trans);
    line_measures.emplace_back(measure);
  }

  // initial line feature's orientation
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(static_cast<int>(line_measures.size()), 3);
  for (size_t i = 0; i < line_measures.size(); ++i) {
    // calculate normal vector
    auto line_end = line_measures[i];
    Eigen::Vector3d sp = line_end.first;
    Eigen::Vector3d ep = line_end.second;
    Eigen::Vector3d norm_vec = sp.cross(ep);
    norm_vec.normalize();

    // calculate A matrix
    Eigen::Quaterniond Rcic0 = rotates[i].inverse()*rotates[0];
    A.row(i) = norm_vec.transpose() * Rcic0.toRotationMatrix();
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeFullV);
  Eigen::VectorXd singulars = svd.singularValues();
  std::cout << "Degenerate case: " << singulars.transpose() << std::endl;
  EXPECT_LE(singulars(1), 0.01);
}

TEST_F(LineFeatureTest, CompDegenerateCase)
{
  /* ground truth */
  Eigen::Vector3d true_direct_vec = (eP_ - sP_).normalized();
  Eigen::Vector3d true_norm_vec = eP_.normalized().cross(sP_.normalized());
  const double true_d = (eP_.cross(sP_)).norm() / (eP_ - sP_).norm();

  /* estimate */
  std::vector<Eigen::Quaterniond> rotates;
  std::vector<Eigen::Vector3d> transs;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> line_measures;
  { // degenerate case
    initFakePose(10, Eigen::Vector3d{10, 20, 15}, Eigen::Vector3d{0, 0, 0});
    auto rotate = fake_pose_ptr_->getRotate();
    auto trans  = fake_pose_ptr_->getTranslate();
    rotates.insert(rotates.end(), rotate.begin(), rotate.end());
    transs.insert(transs.end(), trans.begin(), trans.end());
  }
  { // normal case
    initFakePose(10, Eigen::Vector3d{0, 0, 0}, Eigen::Vector3d{0, -6, 0}, Eigen::Vector3d{10, 20, 15});
    auto rotate = fake_pose_ptr_->getRotate();
    auto trans  = fake_pose_ptr_->getTranslate();
    rotates.insert(rotates.end(), rotate.begin(), rotate.end());
    transs.insert(transs.end(), trans.begin(), trans.end());
  }

  for (size_t i = 0; i < rotates.size(); ++i) {
    auto rotate  = rotates[i];
    auto trans   = transs[i];
    auto measure = fakeMeasure(rotate, trans);
    line_measures.emplace_back(measure);
  }

  // initial line feature's orientation
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(static_cast<int>(line_measures.size()), 3);
  for (size_t i = 0; i < line_measures.size(); ++i) {
    // calculate normal vector
    auto line_end = line_measures[i];
    Eigen::Vector3d sp = line_end.first;
    Eigen::Vector3d ep = line_end.second;
    Eigen::Vector3d norm_vec = sp.cross(ep);
    norm_vec.normalize();

    // calculate A matrix
    Eigen::Quaterniond Rcic0 = rotates[i].inverse()*rotates[0];
    A.row(i) = norm_vec.transpose() * Rcic0.toRotationMatrix();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeFullV);
  Eigen::VectorXd singulars = svd.singularValues();
  std::cout << "Half degenerate case: " << singulars.transpose() << std::endl;
  EXPECT_GE(singulars(1), 0.01);

  Eigen::VectorXd direct_vec = svd.matrixV().rightCols(1);
  direct_vec.normalize();

  const double res_angle = acos(direct_vec.dot(true_direct_vec));
  std::cout << "Residual angle: " << res_angle * MSCKF::Rad2Deg << " Deg." << std::endl;
  EXPECT_LE(res_angle, 1.0 * MSCKF::Deg2Rad);

  // init line feature's distance
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(static_cast<int>(line_measures.size()), 1);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(static_cast<int>(line_measures.size()), 1);
  Eigen::Vector3d direct_vec0 = direct_vec;
  Eigen::Vector3d norm_vec0;
  for (size_t i = 0; i < line_measures.size(); ++i) {
    // calculate normal vector
    auto line_end = line_measures[i];
    Eigen::Vector3d sp = line_end.first;
    Eigen::Vector3d ep = line_end.second;
    Eigen::Vector3d norm_vec = sp.cross(ep);
    norm_vec.normalize();

    if (i == 0) {
      norm_vec0 = norm_vec;
    }

    // rotate
    Eigen::Quaterniond Rc0ci = rotates[0].inverse()*rotates[i];
    Eigen::Vector3d tc0ci = rotates[0].inverse()*(transs[i] - transs[0]);
    Eigen::Vector3d norm_vec0_i = Rc0ci*norm_vec;
    Eigen::Vector3d bi = direct_vec0.cross(norm_vec0_i);

    B.row(i) = bi.transpose() * norm_vec0;
    C.row(i) = bi.transpose() * tc0ci.cross(direct_vec0);
  }
  const double d = (B.transpose() * C)(0, 0) / (B.transpose() * B)(0, 0);
  std::cout << "Distance: " << d << ", True distance: " << true_d << std::endl;
  EXPECT_LE(std::abs(d - true_d), 0.1);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
