#pragma once

#include "iostream"
#include "chrono"

using namespace std;

namespace COMMON {

using Clock = chrono::steady_clock;
using Time  = chrono::steady_clock::time_point;
using Duration = chrono::steady_clock::duration;

class TicToc {
public:
  TicToc(): begin_time_(Clock::now()) {}
  
  void tic()
  {
    begin_time_ = Clock::now();
  }

  double toc() 
  {
    return chrono::duration_cast<chrono::milliseconds>(Clock::now() - begin_time_).count();
  }
private:
  Time   begin_time_;
};

};