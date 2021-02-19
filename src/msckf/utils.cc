#include "msckf/utils.h"

namespace COMMON {

double toSecond(const Duration& duration) {
  const double factor = (double)chrono::milliseconds::period::num/chrono::milliseconds::period::den;
  return chrono::duration_cast<chrono::milliseconds>(duration).count()*factor;
}

Time fromUniversal(const int64_t ticks) { 
  return Time(Duration(ticks)); 
}

Duration fromSecond(double seconds) {
  return std::chrono::duration_cast<Duration>(
      std::chrono::duration<double>(seconds));
}

int64_t toUniversal(const Time time) { 
  return time.time_since_epoch().count();
}

double toTimeStamp(const Time time) {
  return toUniversal(time)*1.e-9;
}

}; // namespace COMMON
