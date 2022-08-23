#pragma once

#include "common.h"
#include "config.h"
#include "datas.h"

namespace MSCKF {

class SFM {
private:
  const SFMParam &param_;

public:
  SFM(const SFMParam &param) : param_(param) {}
  ~SFM() = default;

  SFM(const SFM &sfm) = delete;
  SFM &operator=(const SFM &sfm) = delete;

  bool initialFeature(Feature& ftr, CameraWindow& cams);
  bool checkMotion(Feature& ftr, CameraWindow& cams);
  Eigen::Vector2d evaluate(const Eigen::Vector3d& Pj_w, const CameraStatus& T_ci, const Eigen::Vector3d& obs_ci, Eigen::Matrix<double, 2, 3>* jacobian);

private:
  bool tryToInit(Feature& ftr, CameraWindow& cams, Eigen::Vector3d& position);
};

} // namespace MSCKF
