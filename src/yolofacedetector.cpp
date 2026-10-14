#include "yolofacedetector.h"
#include <iostream>
#include <algorithm>

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
      float confidence = 0.0f;
      bool isFaceOrPerson = false;

      if (outs[i].cols > 6) {
        // Multi-class YOLO (e.g., COCO 80 classes where class 0 = person)
        float personScore = data[5];
        float maxClassScore = 0.0f;
        int classId = -1;
        for (int k = 5; k < outs[i].cols; ++k) {
          if (data[k] > maxClassScore) {
            maxClassScore = data[k];
            classId = k - 5;
          }
        }

        float personConfidence = objectness * personScore;
        float maxConfidence = objectness * maxClassScore;

        if ((classId == 0 && maxConfidence >= confThreshold * 0.5f) ||
            (personConfidence >= confThreshold * 0.5f) ||
            (objectness >= confThreshold && personScore >= 0.15f)) {
          isFaceOrPerson = true;
          confidence = std::max(personConfidence, maxConfidence);
        }
      } else {
        // Single-class YOLO (e.g. YOLO Face detector)
        float classScore = (outs[i].cols == 6) ? data[5] : 1.0f;
        confidence = objectness * classScore;
        if (confidence >= confThreshold * 0.5f || objectness >= confThreshold) {
          isFaceOrPerson = true;
        }
      }

      if (isFaceOrPerson) {
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

        // If multi-class model and tall bounding box (person body), crop head/face upper region
        if (outs[i].cols > 6 && height > width * 1.2) {
          int headHeight = static_cast<int>(height * 0.45);
          int headWidth = static_cast<int>(width * 0.75);
          int headLeft = left + (width - headWidth) / 2;
          int headTop = top;

          headLeft = std::max(0, std::min(headLeft, frameWidth - 1));
          headTop = std::max(0, std::min(headTop, frameHeight - 1));
          headWidth = std::min(headWidth, frameWidth - headLeft);
          headHeight = std::min(headHeight, frameHeight - headTop);

          if (headWidth > 10 && headHeight > 10) {
            left = headLeft;
            top = headTop;
            width = headWidth;
            height = headHeight;
          }
        }

        if (width > 10 && height > 10) {
          boxes.push_back(cv::Rect(left, top, width, height));
          confidences.push_back(confidence > 0 ? confidence : objectness);
        }
      }
    }
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, 0.1f, nmsThreshold, indices);
  for (int idx : indices) {
    faceBoxes.push_back(boxes[idx]);
  }

  return faceBoxes;
}
