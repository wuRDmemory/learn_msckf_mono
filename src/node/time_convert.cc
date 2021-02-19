#include "time_convert.h"

constexpr int64_t kUtsEpochOffsetFromUnixEpochInSeconds = (719162ll * 24ll * 60ll * 60ll);

ros::Time toRos(const COMMON::Time& time) {
  int64_t uts_timestamp = COMMON::toUniversal(time);
  int64_t ns_since_unix_epoch = (uts_timestamp - kUtsEpochOffsetFromUnixEpochInSeconds*10000000ll)*100ll;
  ros::Time ros_time;
  ros_time.fromNSec(ns_since_unix_epoch);
  return ros_time;
}

COMMON::Time fromRos(const ros::Time& time) {
  return COMMON::fromUniversal(
      (time.sec + kUtsEpochOffsetFromUnixEpochInSeconds)*10000000ll +
      (time.nsec + 50) / 100);
}
