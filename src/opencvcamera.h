#ifndef OPENCVCAMERA_H
#define OPENCVCAMERA_H

#include <QImage>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "yolofacedetector.h"

#define DEFAULT_WIDTH 640
#define DEFAULT_HEIGHT 480

using std::vector;
using cv::VideoCapture;
using cv::Mat;
using cv::Rect;

class OpenCVCamera {
 public:
  OpenCVCamera();
  ~OpenCVCamera();
  QImage getCurrentFrame();
  QImage getCurrentFace();
  void getCurrentFaceMat(Mat& face);
  bool detectFaceInImage(const Mat& inputImg, Mat& faceROI);

 private:
  VideoCapture capture;
  YOLOFaceDetector yoloDetector;
  Mat frame, main, face;
  QImage mainImage, faceImage;
  vector<Rect> lastFaces;
  const int IMAGE_WIDTH = DEFAULT_WIDTH;
  const int IMAGE_HEIGHT = DEFAULT_HEIGHT;
};

#undef DEFAULT_WIDTH
#undef DEFAULT_HEIGHT

#endif // OPENCVCAMERA_H
