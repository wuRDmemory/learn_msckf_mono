#pragma once

#include "iostream"
#include "vector"
#include "algorithm"
#include "chrono"

#include "ros/ros.h"

#include "msckf/utils.h"

using namespace std;

ros::Time toRos(const COMMON::Time& time);

COMMON::Time fromRos(const ros::Time& time);