#pragma once

#include "iostream"
#include "vector"
#include "algorithm"
#include "chrono"
#include "thread"

using namespace std;

namespace COMMON {


using Clock = chrono::steady_clock;
using Time  = chrono::steady_clock::time_point;
using Duration = chrono::steady_clock::duration;

double toTimeStamp(const Time time);

double toSecond(const Duration& duration);

int64_t toUniversal(const Time time);

Duration fromSecond(double seconds);

Time fromUniversal(const int64_t ticks);


class Tick {
public:
  Tick(const char* name): name_(name), begin_time_(Clock::now()) {}

  double eclipse() {
    return chrono::duration_cast<chrono::milliseconds>(Clock::now() - begin_time_).count();
  }
  
  void reset() {
    begin_time_ = Clock::now();
  }

  string print() {
    char info[256];
    sprintf(info, "<%s> Time eclipse: %lf s.", name_.c_str(), toSecond(Clock::now() - begin_time_));
    return info;
  }

private:
  string name_;
  Time   begin_time_;
};

};

