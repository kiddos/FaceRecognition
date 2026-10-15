#ifndef YOLOFACEDETECTOR_H
#define YOLOFACEDETECTOR_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

class YOLOFaceDetector {
 public:
  YOLOFaceDetector();
  explicit YOLOFaceDetector(const std::string& onnxPath);
  YOLOFaceDetector(const std::string& configPath,
                   const std::string& weightsPath);
  ~YOLOFaceDetector();

  bool loadModel(const std::string& configPath,
                 const std::string& weightsPath);
  bool loadONNXModel(const std::string& onnxPath);

  std::vector<cv::Rect> detectFaces(const cv::Mat& frame,
                                    float confThreshold = 0.25f,
                                    float nmsThreshold = 0.45f);

  bool isLoaded() const;

 private:
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "YOLOFaceDetector"};
  Ort::SessionOptions sessionOptions;
  std::unique_ptr<Ort::Session> session;

  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  std::vector<const char*> inputNamePtrs;
  std::vector<const char*> outputNamePtrs;

  bool loaded = false;
  int inputWidth = 640;
  int inputHeight = 640;
};

#endif // YOLOFACEDETECTOR_H
