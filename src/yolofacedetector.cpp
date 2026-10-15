#include "yolofacedetector.h"
#include <iostream>
#include <algorithm>

YOLOFaceDetector::YOLOFaceDetector() {
  // Try loading default ONNX model paths first
  if (!loadONNXModel("models/yolov11n-face.onnx")) {
    if (!loadONNXModel("yolov11n-face.onnx")) {
      if (!loadONNXModel("models/model.onnx")) {
        if (!loadONNXModel("model.onnx")) {
          // Fallback to Darknet YOLO if ONNX is missing
          if (!loadModel("models/yolov4-tiny.cfg", "models/yolov4-tiny.weights")) {
            loadModel("yolov4-tiny.cfg", "yolov4-tiny.weights");
          }
        }
      }
    }
  }
}

YOLOFaceDetector::YOLOFaceDetector(const std::string& onnxPath) {
  loadONNXModel(onnxPath);
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
      inputWidth = 416;
      inputHeight = 416;
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
      inputWidth = 640;
      inputHeight = 640;
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
  float scaleX = static_cast<float>(frameWidth) / inputWidth;
  float scaleY = static_cast<float>(frameHeight) / inputHeight;

  for (size_t i = 0; i < outs.size(); ++i) {
    cv::Mat outMat = outs[i];

    // Handle 3D tensor output from YOLOv8 / YOLOv11 ONNX (e.g. [1, 5, 8400] or [1, 8400, 5])
    if (outMat.dims == 3) {
      int channels = outMat.size[1];
      int numBoxes = outMat.size[2];

      cv::Mat predictions;
      if (channels == 5 || channels == 6) {
        // [1, 5, N] -> reshape to 2D [5, N] then transpose to [N, 5]
        cv::Mat mat2D(channels, numBoxes, CV_32F, outMat.ptr<float>());
        predictions = mat2D.t();
      } else if (outMat.size[2] == 5 || outMat.size[2] == 6) {
        // [1, N, 5] -> reshape to 2D [N, 5]
        predictions = cv::Mat(outMat.size[1], outMat.size[2], CV_32F, outMat.ptr<float>());
      }

      if (!predictions.empty()) {
        for (int j = 0; j < predictions.rows; ++j) {
          float score = predictions.at<float>(j, 4);
          if (score >= confThreshold) {
            float xc = predictions.at<float>(j, 0);
            float yc = predictions.at<float>(j, 1);
            float bw = predictions.at<float>(j, 2);
            float bh = predictions.at<float>(j, 3);

            int left = static_cast<int>((xc - bw / 2.0f) * scaleX);
            int top = static_cast<int>((yc - bh / 2.0f) * scaleY);
            int width = static_cast<int>(bw * scaleX);
            int height = static_cast<int>(bh * scaleY);

            left = std::max(0, std::min(left, frameWidth - 1));
            top = std::max(0, std::min(top, frameHeight - 1));
            width = std::min(width, frameWidth - left);
            height = std::min(height, frameHeight - top);

            if (width > 10 && height > 10) {
              boxes.push_back(cv::Rect(left, top, width, height));
              confidences.push_back(score);
            }
          }
        }
        continue;
      }
    } else if (outMat.dims == 2 && (outMat.rows == 5 || outMat.rows == 6)) {
      // 2D tensor [5, N] -> transpose to [N, 5]
      cv::Mat predictions = outMat.t();
      for (int j = 0; j < predictions.rows; ++j) {
        float score = predictions.at<float>(j, 4);
        if (score >= confThreshold) {
          float xc = predictions.at<float>(j, 0);
          float yc = predictions.at<float>(j, 1);
          float bw = predictions.at<float>(j, 2);
          float bh = predictions.at<float>(j, 3);

          int left = static_cast<int>((xc - bw / 2.0f) * scaleX);
          int top = static_cast<int>((yc - bh / 2.0f) * scaleY);
          int width = static_cast<int>(bw * scaleX);
          int height = static_cast<int>(bh * scaleY);

          left = std::max(0, std::min(left, frameWidth - 1));
          top = std::max(0, std::min(top, frameHeight - 1));
          width = std::min(width, frameWidth - left);
          height = std::min(height, frameHeight - top);

          if (width > 10 && height > 10) {
            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(score);
          }
        }
      }
      continue;
    }

    // Handle 2D tensor Darknet format [N, C] where each row is [x, y, w, h, obj_conf, ...]
    float* data = (float*)outMat.data;
    for (int j = 0; j < outMat.rows; ++j, data += outMat.cols) {
      float objectness = data[4];
      float confidence = 0.0f;
      bool isFaceOrPerson = false;

      if (outMat.cols > 6) {
        float personScore = data[5];
        float maxClassScore = 0.0f;
        int classId = -1;
        for (int k = 5; k < outMat.cols; ++k) {
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
        float classScore = (outMat.cols == 6) ? data[5] : 1.0f;
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

        if (width > 10 && height > 10) {
          boxes.push_back(cv::Rect(left, top, width, height));
          confidences.push_back(confidence > 0 ? confidence : objectness);
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
