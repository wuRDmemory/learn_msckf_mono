#include "iostream"
#include "unordered_map"
#include "memory"
#include "functional"

#include "boost/functional.hpp"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "gflags/gflags.h"

#include "ros/ros.h"

#include "node/node.h"
#include "msckf/config.h"

using namespace std;

/**
 * program inputs
**/
DEFINE_string(config_file, "", "Config file path.");

/**
 * configuration functions.
 */
bool setup(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Config file: " << FLAGS_config_file;
  if (FLAGS_config_file.empty()) {
    CHECK(false) << "Can not open config file.";
    return false;
  }

  LOG(INFO) << "Setup down.";
  return true;
}

/**
 * main function.
 */
int main(int argc, char** argv) {
  // system setup
  if (!setup(argc, argv)) {
    LOG(FATAL) << "System setup failed. Please check.";
    return 0;
  }

  // ros setup
  ros::init(argc, argv, "msckf_node");
  ros::NodeHandle n;

  Node node(n, FLAGS_config_file);

  ros::spin();

  return 1;
}
