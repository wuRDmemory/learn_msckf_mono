#include "iostream"
#include "thread"
#include "opencv2/opencv.hpp"

#include "camera/camera.h"
#include "camera/pinhole_camera.h"
#include "msckf/config.h"
#include "msckf/tracker.h"
#include "node/time_convert.h"

#include "helpers/directory_helper.h"

using namespace std;

using ImageInformation    = TEST::DirectoryHelper::ImageInformation;
using InertialInformation = TEST::DirectoryHelper::InertialInformation;

const cv::String keys =
    "{help h usage ?  || todo help              }"
    "{@config_path    || path to config path    }"
    "{@dataset_name   || dataset name           }"
    "{@dataset_path   || path to dataset        }";

int main(int argc, char** argv) {

  cv::CommandLineParser parser(argc, argv, keys);

  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  if (   !parser.has("@config_path") 
      || !parser.has("@dataset_path")
      || !parser.has("@dataset_name")) {
    parser.printMessage();
    return 0;
  }

  string config_path   = parser.get<string>(0);
  string dataset_name  = parser.get<string>(1);
  string dataset_path  = parser.get<string>(2);
  cout << "Config  path: \t" << config_path << endl;
  cout << "Dataset name: \t" << dataset_name << endl;
  cout << "Dataset path: \t" << dataset_path << endl;

  // config
  Config::getInstance(config_path.c_str());

  // scan directory.
  TEST::DirectoryHelper dir_walker;
  TEST::DirectoryHelper::DirectoryInformation 
  dir_info = dir_walker.process(dataset_name, dataset_path);

  // tracker
  MSCKF::ImageTracker image_tracker(config_path);

  for (size_t i = 0; i < dir_info.images_info.size(); ++i) {
    const ImageInformation& data = dir_info.images_info[i];
    
    double  time  = data.timestamps;
    cv::Mat image = cv::imread(data.image_path, 0);
    
    image_tracker.feedImage(time, image);
  }

  return 1;
}
