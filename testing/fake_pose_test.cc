#include <iostream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include "helpers/fake_pose.h"

class FakePoseTest : public ::testing::Test {
protected:
  Eigen::Quaterniond start_q_, end_q_;
  Eigen::Vector3d    start_p_, end_p_;
  std::unique_ptr<TESTING::FakePose> fake_pose_ptr_ = nullptr;

public:
  FakePoseTest() = default;
  ~FakePoseTest() override {}

  void SetUp() override {
    start_q_ = Eigen::Quaterniond::Identity();
    start_p_ = Eigen::Vector3d::Zero();
    end_q_ = MATH_UTILS::eulerToQuaternion(Eigen::Vector3d{10*MSCKF::Deg2Rad, 20*MSCKF::Deg2Rad, 15*MSCKF::Deg2Rad});
    end_p_ = Eigen::Vector3d{0.0, -4.0, 5.0};
    fake_pose_ptr_.reset(new TESTING::FakePose(10, start_p_, end_p_, start_q_, end_q_));
  }

  void TearDown() override {
    fake_pose_ptr_.reset(nullptr);
  }
};

TEST_F(FakePoseTest, CheckEndPoint) {
  std::vector<Eigen::Quaterniond> rotates = fake_pose_ptr_->getRotate();
  std::vector<Eigen::Vector3d> translate  = fake_pose_ptr_->getTranslate();

  Eigen::Quaterniond end_q = rotates.back();
  Eigen::Vector3d end_p = translate.back();
  Eigen::Vector3d res_rotate = MATH_UTILS::quaternionToRotateVector(end_q.inverse()*end_q_);
  Eigen::Vector3d res_trans  = end_p - end_p_;

  const double res_angle = res_rotate.norm();
  const double res_dist  = res_trans.norm();

  EXPECT_LE(res_angle, 1*MSCKF::Deg2Rad); // 5 deg
  EXPECT_LE(res_dist, 0.1); // 0.1m
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



