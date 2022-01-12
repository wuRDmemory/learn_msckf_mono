#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Core"

#include "../../src/msckf/common.h"
#include "../../src/msckf/math_utils.h"

namespace TESTING {

class FakePose {
private:
  std::vector<Eigen::Quaterniond> rotates_;
  std::vector<Eigen::Vector3d> trans_;

public:
  FakePose(const int N, Eigen::Vector3d& start_trans, Eigen::Vector3d& end_trans, Eigen::Quaterniond& start_rotate, Eigen::Quaterniond& end_rotate) {
    Eigen::Vector3d delta_trans = end_trans - start_trans;
    Eigen::Vector3d delta_rotate = MATH_UTILS::quaternionToRotateVector(start_rotate.inverse()*end_rotate);

    Eigen::Vector3d step_trans = delta_trans / N;
    Eigen::Vector3d step_rotate = delta_rotate / N;

    int i = 0;
    Eigen::Quaterniond q = start_rotate;
    Eigen::Vector3d p = start_trans;
    do {
      Eigen::Quaterniond q = start_rotate.slerp((float)i/N, end_rotate);
      Eigen::Vector3d p = start_trans + i*step_trans;
      rotates_.emplace_back(q);
      trans_.emplace_back(p);
    } while (++i <= N);
  }

  const std::vector<Eigen::Quaterniond>& getRotate() const {
    return rotates_;
  }

  const std::vector<Eigen::Vector3d>& getTranslate() const {
    return trans_;
  }
};

} 