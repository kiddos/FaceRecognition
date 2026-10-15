#include "yolofacedetector.h"
#include <iostream>
#include <algorithm>
#include <array>

YOLOFaceDetector::YOLOFaceDetector() {
  if (!loadONNXModel("models/yolov11n-face.onnx")) {
    if (!loadONNXModel("yolov11n-face.onnx")) {
      if (!loadONNXModel("models/model.onnx")) {
        loadONNXModel("model.onnx");
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
  (void)configPath;
  (void)weightsPath;
  return loadONNXModel("models/yolov11n-face.onnx");
}

bool YOLOFaceDetector::loadONNXModel(const std::string& onnxPath) {
  try {
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session = std::make_unique<Ort::Session>(env, onnxPath.c_str(), sessionOptions);
    if (!session) {
      loaded = false;
      return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputs = session->GetInputCount();
    inputNames.clear();
    inputNamePtrs.clear();
    for (size_t i = 0; i < numInputs; ++i) {
      auto nameAlloc = session->GetInputNameAllocated(i, allocator);
      inputNames.push_back(nameAlloc.get());
    }
    for (size_t i = 0; i < inputNames.size(); ++i) {
      inputNamePtrs.push_back(inputNames[i].c_str());
    }

    size_t numOutputs = session->GetOutputCount();
    outputNames.clear();
    outputNamePtrs.clear();
    for (size_t i = 0; i < numOutputs; ++i) {
      auto nameAlloc = session->GetOutputNameAllocated(i, allocator);
      outputNames.push_back(nameAlloc.get());
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
      outputNamePtrs.push_back(outputNames[i].c_str());
    }

    loaded = true;
    inputWidth = 640;
    inputHeight = 640;
    std::cout << "[YOLOFaceDetector] Successfully loaded ONNX YOLO model from "
              << onnxPath << " with ONNX Runtime" << std::endl;
    return true;
  } catch (const Ort::Exception& e) {
    std::cerr << "[YOLOFaceDetector] Ort::Exception loading ONNX model "
              << onnxPath << ": " << e.what() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[YOLOFaceDetector] Exception loading ONNX model: " << e.what() << std::endl;
  }
  loaded = false;
  return false;
}

bool YOLOFaceDetector::isLoaded() const {
  return loaded && session != nullptr;
}

std::vector<cv::Rect> YOLOFaceDetector::detectFaces(const cv::Mat& frame,
                                                    float confThreshold,
                                                    float nmsThreshold) {
  std::vector<cv::Rect> faceBoxes;
  if (frame.empty() || !isLoaded()) {
    return faceBoxes;
  }

  int frameWidth = frame.cols;
  int frameHeight = frame.rows;

  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(inputWidth, inputHeight));

  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  cv::Mat floatImg;
  rgb.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

  int imageArea = inputWidth * inputHeight;
  std::vector<float> inputTensorValues(1 * 3 * imageArea);

  for (int i = 0; i < imageArea; ++i) {
    cv::Vec3f p = floatImg.at<cv::Vec3f>(i / inputWidth, i % inputWidth);
    inputTensorValues[0 * imageArea + i] = p[0];
    inputTensorValues[1 * imageArea + i] = p[1];
    inputTensorValues[2 * imageArea + i] = p[2];
  }

  std::array<int64_t, 4> inputShape = {1, 3, inputHeight, inputWidth};
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
      inputShape.data(), inputShape.size());

  try {
    auto outputTensors = session->Run(
        Ort::RunOptions{nullptr}, inputNamePtrs.data(), &inputTensor, 1,
        outputNamePtrs.data(), outputNamePtrs.size());

    if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
      return faceBoxes;
    }

    float* floatArray = outputTensors[0].GetTensorMutableData<float>();
    auto shapeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    auto shape = shapeInfo.GetShape();

    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    float scaleX = static_cast<float>(frameWidth) / inputWidth;
    float scaleY = static_cast<float>(frameHeight) / inputHeight;

    if (shape.size() == 3) {
      int channels = static_cast<int>(shape[1]);
      int numBoxes = static_cast<int>(shape[2]);

      if (channels == 5 || channels == 6) {
        for (int j = 0; j < numBoxes; ++j) {
          float score = floatArray[4 * numBoxes + j];
          if (score >= confThreshold) {
            float xc = floatArray[0 * numBoxes + j];
            float yc = floatArray[1 * numBoxes + j];
            float bw = floatArray[2 * numBoxes + j];
            float bh = floatArray[3 * numBoxes + j];

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
      } else if (shape[2] == 5 || shape[2] == 6) {
        int N = static_cast<int>(shape[1]);
        int C = static_cast<int>(shape[2]);
        for (int j = 0; j < N; ++j) {
          float score = floatArray[j * C + 4];
          if (score >= confThreshold) {
            float xc = floatArray[j * C + 0];
            float yc = floatArray[j * C + 1];
            float bw = floatArray[j * C + 2];
            float bh = floatArray[j * C + 3];

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
      }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (int idx : indices) {
      faceBoxes.push_back(boxes[idx]);
    }
  } catch (const Ort::Exception& e) {
    std::cerr << "[YOLOFaceDetector] Ort::Exception during inference: " << e.what() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[YOLOFaceDetector] Exception during inference: " << e.what() << std::endl;
  }

  return faceBoxes;
}
