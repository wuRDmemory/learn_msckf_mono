#pragma once

#include "iostream"
#include "vector"
#include "sstream"

#include "opencv2/opencv.hpp"

using namespace std;

namespace CAMERA {

class Camera {
public:
  Camera(string config_path) {
    assert(!config_path.empty());
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    assert(fs.isOpened());

    cv::FileNode n = fs["camera"];
    assert(!n.empty());

    n["type"] >> type_;

    width_  = (int)n["width"];
    height_ = (int)n["height"];

    fx_ = (float)n["fx"];
    fy_ = (float)n["fy"];
    cx_ = (float)n["cx"];
    cy_ = (float)n["cy"];

    k1_ = (float)n["k1"];
    k2_ = (float)n["k2"];
    d1_ = (float)n["d1"];
    d2_ = (float)n["d2"];
  }

  Camera(cv::FileNode& n) {
    assert(!n.empty());

    n["type"] >> type_;

    width_  = (int)n["width"];
    height_ = (int)n["height"];

    fx_ = (float)n["fx"];
    fy_ = (float)n["fy"];
    cx_ = (float)n["cx"];
    cy_ = (float)n["cy"];

    k1_ = (float)n["k1"];
    k2_ = (float)n["k2"];
    d1_ = (float)n["d1"];
    d2_ = (float)n["d2"];
  }

  Camera(int width, int height, string type, 
         float fx, float fy, float cx, float cy, 
         float k1, float k2, float d1, float d2) :
    width_(width), height_(height), type_(type), 
    fx_(fx), fy_(fy), cx_(cx), cy_(cy), 
    k1_(k1), k2_(k2), d1_(d1), d2_(d2) {} 

  Camera(const Camera& camera) = delete;

  Camera& operator=(const Camera& camera) = delete;

  virtual ~Camera() {}

  // image point projects to unit space with undistort.
  // means P.z = 1
  virtual cv::Point3f imageToSpace(const cv::Point2f& pi) const = 0;

  // camera space projects to image with distort.
  virtual cv::Point2f spaceToImage(const cv::Point3f& pc) const = 0;

  // image lifts to unit space without distort correct.
  virtual cv::Point3f liftToSpace(const cv::Point2f& pi) const {
    cv::Point3f pc;
    pc.x = (pi.x - cx_)/fx_;
    pc.y = (pi.y - cy_)/fy_;
    pc.z = 1;
    return pc;
  }

  // unit space projects to image without distort correct
  virtual cv::Point2f projectToImage(const cv::Point3f& pc) const {
    cv::Point2f pi;
    pi.x = (pc.x/pc.z)*fx_ + cx_;
    pi.y = (pc.y/pc.z)*fy_ + cy_;
    return pi;
  }

  // directly undistort point in image coordinate.
  // TODO: check this with VINS-Mono
  virtual cv::Point2f undistortPoint(const cv::Point2f& pi) const {
    cv::Point3f Pc = imageToSpace(pi);
    return projectToImage(Pc);
  }

  virtual cv::Mat undistortImage(const cv::Mat& image, cv::Mat* new_camera_K) const = 0;

  virtual string print() {
    stringstream ss;
    ss << endl;
    ss << "===== " << type() << "'s information =====" << endl;
    ss << "image size: [ " << width() << ", " << height() << " ]." << endl;
    ss << "intrinsic parameter: [ " << fx() << ", " << fy() << ", " << cx() << ", " << cy() << " ]." << endl;
    ss << "undistort parameter: [ " << k1() << ", " << k2() << ", " << k1() << ", " << k2() << " ]." << endl;
    ss << "has mask? " << (mask().empty() ? "No" : "Yes") << endl;
    ss << endl;
    return ss.str();
  }

public:
  string type() const { return type_; }

  int width()  const { return width_;  }
  int height() const { return height_; }

  float fx() const { return fx_; }
  float fy() const { return fy_; }
  float cx() const { return cx_; }
  float cy() const { return cy_; }

  float k1() const { return k1_; }
  float k2() const { return k2_; }
  float d1() const { return d1_; }
  float d2() const { return d2_; }

  const cv::Mat& mask() const { return mask_; }

protected:
  int   width_, height_;
  float fx_, fy_, cx_, cy_;
  float k1_, k2_, d1_, d2_;
  cv::Mat mask_;
  string type_;
};


} // namespace CAMERA
