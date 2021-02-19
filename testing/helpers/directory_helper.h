#pragma once

#include "iostream"
#include "vector"
#include "memory"
#include "cstdlib"
#include "string"
#include "istream"
#include "stdio.h"
#include "stdlib.h"

using namespace std;

namespace TEST {

class DirectoryHelper {

public:
  struct ImageInformation {
    double timestamps;
    string image_path;
  };

  struct InertialInformation {
    double timestamps;
    double gx, gy, gz;    
    double ax, ay, az;
    double temperature;
  };
  

  struct DirectoryInformation { 
    string dataset_name;
    string dataset_path;
    vector<ImageInformation>    images_info;
    vector<InertialInformation> inertial_info;
  };  

public:
  DirectoryInformation process(string dataset_name, string dataset_path);

private:
  bool walkInEuroc(string dataset_path, vector<ImageInformation>& images_info, vector<InertialInformation>& inertial_info);
};

} // namespace TEST
