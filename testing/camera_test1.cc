#include "msckf/config.h"
#include "camera/camera.h"
#include "camera/pinhole_camera.h"
#include "opencv2/opencv.hpp"

using namespace std;

const cv::String keys =
    "{help h usage ?  || todo help              }"
    "{@config_path    || path to config path    }"
    "{@image_path     || path to image          }";

cv::Point2f projectToImage(const cv::Point3f& pc, const cv::Mat& K) {
  cv::Point2f pi;
  pi.x = K.at<double>(0, 0)*pc.x/pc.z + K.at<double>(0, 2);
  pi.y = K.at<double>(1, 1)*pc.y/pc.z + K.at<double>(1, 2);
  return pi;
}

cv::Point3f projectToSapce(const cv::Point2f& pi, const cv::Mat& K) {
  cv::Point3f pc;
  pc.x = (pi.x - K.at<double>(0, 2))/K.at<double>(0, 0);
  pc.y = (pi.y - K.at<double>(1, 2))/K.at<double>(1, 1);
  pc.z = 1;
  return pc;
}

int main(int argc, char** argv) {
  cv::CommandLineParser parser(argc, argv, keys);

  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  if (!parser.has("@config_path") || !parser.has("@image_path")) {
    parser.printMessage();
    return 0;
  }

  string config_path = parser.get<string>("@config_path");
  string image_path  = parser.get<string>("@image_path");
  cout << "Config file path \t: " << config_path << endl;
  cout << "Image path: \t"        << image_path << endl;

  // construct camera
  cv::FileStorage fs(config_path, cv::FileStorage::READ);
  cv::FileNode    n = fs["camera"];
  CAMERA::Camera* camera = new CAMERA::PinHoleCamera(n);

  // print camera information.
  cout << camera->print();

  // read image
  cv::Mat image = cv::imread(image_path, 0);

  // undistort image
  cv::Mat new_K;
  cv::Mat image_un = camera->undistortImage(image, &new_K);

  // find corner  
  vector<cv::Point2f> undistort_points;
  cv::goodFeaturesToTrack(image_un, undistort_points, 20, 0.01, 50);
  cout << "Detect " << undistort_points.size() << " points." << endl;

  // project and draw
  cv::Mat merge;
  cv::hconcat(image, image_un, merge);
  cv::cvtColor(merge, merge, cv::COLOR_GRAY2BGR);

  cv::Point2f stride(camera->width(), 0);

  vector<cv::Point2f> distort_points(undistort_points.size());
  for (size_t i = 0; i < undistort_points.size(); ++i) {
    cv::Point3f space_point = projectToSapce(undistort_points[i], new_K);
    distort_points[i] = camera->spaceToImage(space_point);

    cv::Scalar color;
    cv::theRNG().fill(color, CV_8U, 0, 255);

    cv::circle(merge, distort_points[i], 2, color, 1);
    cv::circle(merge, undistort_points[i]+stride, 2, color, 1);
    cv::line(merge, distort_points[i], undistort_points[i]+stride, color, 1);
  }

  cv::imshow("[merge]", merge);
  cv::waitKey();

  return 1;
}
