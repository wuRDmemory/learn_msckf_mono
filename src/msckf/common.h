#pragma once

#include <iostream>
#include <queue>
#include <list>
#include <cfloat>
#include <limits>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <functional>
#include <fstream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <condition_variable>
#include <thread>
#include <boost/math/distributions/chi_squared.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#include <opencv2/opencv.hpp>
#include <opencv/cxeigen.hpp>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "glog/logging.h"

namespace MSCKF {

constexpr double Deg2Rad = M_PI / 180;
constexpr double Rad2Deg = 180 / M_PI;

constexpr int IMU_STATUS_NUM = 5;
constexpr int IMU_NOISE_NUM  = 4;
constexpr int IMU_STATUS_DIM = 3*IMU_STATUS_NUM;
constexpr int IMU_NOISE_DIM  = 3*IMU_NOISE_NUM;
constexpr int J_R  = 0;
constexpr int J_BG = 3;
constexpr int J_V  = 6;
constexpr int J_BA = 9;
constexpr int J_P  = 12;
constexpr int G_Ng  = 0;
constexpr int G_Nbg = 3;
constexpr int G_Na  = 6;
constexpr int G_Nba = 9;
constexpr int C_R = 0;
constexpr int C_P = 3;

} // namespace MSCKF
