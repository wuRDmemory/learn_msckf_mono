#include "time.h"
#include "iostream"
#include "thread"

#include "msckf/utils.h"

using namespace std;
using namespace COMMON;

int main(int argc, char** argv) {

  Time timestamp = Clock::now();
  Tick tick("A");
  std::this_thread::sleep_for(chrono::seconds(1));
  cout << tick.print();

  return 1;
}