#include "yolofacedetector.h"
#include <iostream>

YOLOFaceDetector::YOLOFaceDetector() {
  // Try loading default model paths
  if (!loadModel("models/yolov4-tiny.cfg", "models/yolov4-tiny.weights")) {
    if (!loadModel("yolov4-tiny.cfg", "yolov4-tiny.weights")) {
      loadONNXModel("models/yolo-face.onnx");
    }
  }
}

YOLOFaceDetector::YOLOFaceDetector(const std::string& configPath,
                                   const std::string& weightsPath) {
  loadModel(configPath, weightsPath);
}

YOLOFaceDetector::~YOLOFaceDetector() {
}

bool YOLOFaceDetector::loadModel(const std::string& configPath,
                                 const std::string& weightsPath) {
  try {
    net = cv::dnn::readNetFromDarknet(configPath, weightsPath);
    if (!net.empty()) {
      net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
      loaded = true;
      std::cout << "[YOLOFaceDetector] Successfully loaded Darknet YOLO model from "
                << configPath << " and " << weightsPath << std::endl;
      return true;
    }
  } catch (const cv::Exception& e) {
    std::cerr << "[YOLOFaceDetector] Failed to load Darknet model: " << e.what() << std::endl;
  }
  loaded = false;
  return false;
}

bool YOLOFaceDetector::loadONNXModel(const std::string& onnxPath) {
  try {
    net = cv::dnn::readNetFromONNX(onnxPath);
    if (!net.empty()) {
      net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
      loaded = true;
      std::cout << "[YOLOFaceDetector] Successfully loaded ONNX YOLO model from "
                << onnxPath << std::endl;
      return true;
    }
  } catch (const cv::Exception& e) {
    std::cerr << "[YOLOFaceDetector] Failed to load ONNX model: " << e.what() << std::endl;
  }
  loaded = false;
  return false;
}

bool YOLOFaceDetector::isLoaded() const {
  return loaded && !net.empty();
}

std::vector<cv::Rect> YOLOFaceDetector::detectFaces(const cv::Mat& frame,
                                                    float confThreshold,
                                                    float nmsThreshold) {
  std::vector<cv::Rect> faceBoxes;
  if (frame.empty() || !isLoaded()) {
    return faceBoxes;
  }

  cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0,
                                         cv::Size(inputWidth, inputHeight),
                                         cv::Scalar(0, 0, 0), true, false);
  net.setInput(blob);

  std::vector<cv::Mat> outs;
  std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();
  net.forward(outs, outNames);

  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  int frameWidth = frame.cols;
  int frameHeight = frame.rows;

  for (size_t i = 0; i < outs.size(); ++i) {
    float* data = (float*)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
      float objectness = data[4];
      if (objectness > confThreshold) {
        float maxClassScore = 0.0f;
        int classId = -1;
        for (int k = 5; k < outs[i].cols; ++k) {
          if (data[k] > maxClassScore) {
            maxClassScore = data[k];
            classId = k - 5;
          }
        }

        // YOLO detection score: objectness * class probability
        float confidence = (outs[i].cols > 5) ? (objectness * maxClassScore) : objectness;

        // Ensure detection is person/face class (classId == 0 in COCO YOLO) or face single-class YOLO
        if ((classId == 0 || outs[i].cols <= 5) && (confidence > confThreshold || objectness > confThreshold)) {
          int centerX = static_cast<int>(data[0] * frameWidth);
          int centerY = static_cast<int>(data[1] * frameHeight);
          int width = static_cast<int>(data[2] * frameWidth);
          int height = static_cast<int>(data[3] * frameHeight);
          int left = centerX - width / 2;
          int top = centerY - height / 2;

          left = std::max(0, std::min(left, frameWidth - 1));
          top = std::max(0, std::min(top, frameHeight - 1));
          width = std::min(width, frameWidth - left);
          height = std::min(height, frameHeight - top);

          if (width > 10 && height > 10) {
            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence > 0 ? confidence : objectness);
          }
        }
      }
    }
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
  for (int idx : indices) {
    faceBoxes.push_back(boxes[idx]);
  }

  return faceBoxes;
}
