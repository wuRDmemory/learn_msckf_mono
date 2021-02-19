#include "helpers/directory_helper.h"
#include "assert.h"

#include "node/time_convert.h"

namespace TEST {

DirectoryHelper::DirectoryInformation 
DirectoryHelper::process(string dataset_name, string dataset_path) {
  DirectoryHelper::DirectoryInformation result;
  result.dataset_name = dataset_name;
  result.dataset_path = dataset_path;

  if (dataset_name == "euroc") {
    if (!walkInEuroc(dataset_path, result.images_info, result.inertial_info)) {
      cout << "[DH] Some error happened in 'walkInEuroc' function." << endl;
    }
  }
  else {
    cout << "Algo do not support other dataset right now." << endl;
  }

  return result;
}


bool DirectoryHelper::walkInEuroc(string dataset_path, 
                 vector<DirectoryHelper::ImageInformation>& images_info, 
                 vector<DirectoryHelper::InertialInformation>& inertial_info) {
  
  if (dataset_path.back() == '/') {
    dataset_path.pop_back();
  } 

  size_t last_split_char_index = dataset_path.find_last_of('/');
  string base_dir = dataset_path.substr(last_split_char_index);
  if (base_dir != "/mav0") {
    cout << "[Euroc] Can not walk this directory because root is not 'mav0'" << endl;
    assert(false);
    return false;
  }

  { // read images information.

    string image_data_path = dataset_path + "/cam0/data.csv";
    FILE* fp = fopen(image_data_path.c_str(), "r");
    if (!fp) {
      cout << "[Euroc] Can not open 'cam0/data.csv', Please check your dataset's completeness." << endl;
      return false;
    }

    int line_cnt = 0;
    size_t buffer_size = 256;
    char*  buffer      = new char[buffer_size];
    size_t read_len    = getline(&buffer, &buffer_size, fp);

    while (!feof(fp)) {
      if (buffer[0] == '#') {
        read_len = getline(&buffer, &buffer_size, fp);
        continue;
      }

      char image_name[128];
      long long ts_in_ns;
      sscanf(buffer, "%llu,%s\n", &ts_in_ns, image_name);
      images_info.emplace_back(DirectoryHelper::ImageInformation{ts_in_ns*1.e-9,  dataset_path+"/cam0/data/"+image_name});
      ++line_cnt;

      read_len = getline(&buffer, &buffer_size, fp);
    }

    cout << "[Euroc] Read image information done. Total " << images_info.size() << " images." << endl;
  }

  { // read imu information.
    string imu_data_path = dataset_path + "/imu0/data.csv";
    FILE* fp = fopen(imu_data_path.c_str(), "r");
    if (!fp) {
      cout << "[Euroc] Can not open 'imu0/data.csv', Please check your dataset's completeness." << endl;
      return false;
    }

    int line_cnt = 0;
    size_t buffer_size = 384;
    char*  buffer      = new char[buffer_size];
    size_t read_len    = getline(&buffer, &buffer_size, fp);

    while (!feof(fp)) {
      if (buffer[0] == '#') {
        read_len = getline(&buffer, &buffer_size, fp);
        continue;
      }

      long long ts_in_ns;
      double gx, gy, gz, ax, ay, az;
      sscanf(buffer, "%llu,%lf,%lf,%lf,%lf,%lf,%lf\n", &ts_in_ns, &gx, &gy, &gz, &ax, &ay, &az);
      inertial_info.emplace_back(DirectoryHelper::InertialInformation{ts_in_ns*1.e-9, gx, gy, gz, ax, ay, az, 0.0});
      ++line_cnt;

      read_len = getline(&buffer, &buffer_size, fp);
    }

    cout << "[Euroc] Read inertial information done. Total " << inertial_info.size() << " data." << endl;
  }

  return true;
}

};
