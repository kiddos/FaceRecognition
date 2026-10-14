#include "opencvcamera.h"
#include <iostream>

OpenCVCamera::OpenCVCamera() {
  if (!capture.isOpened()) {
    capture = VideoCapture(0);
  }
}

OpenCVCamera::~OpenCVCamera() {
  if (capture.isOpened()) {
    capture.release();
  }
}

QImage OpenCVCamera::getCurrentFrame() {
  if (capture.isOpened()) {
    capture.read(frame);
  }

  if (frame.data && !frame.empty()) {
    frame.copyTo(main);

    // Detect faces using YOLO
    lastFaces = yoloDetector.detectFaces(frame);

    if (!lastFaces.empty()) {
      // Crop primary detected face
      Rect faceRect = lastFaces[0];
      frame(faceRect).copyTo(face);

      // Draw bounding box on main frame
      cv::rectangle(main, faceRect, cv::Scalar(0, 255, 0), 2);
    }

    cv::cvtColor(main, main, cv::COLOR_BGR2RGB);
    mainImage = QImage((uchar*)main.data, main.cols, main.rows,
                       main.step, QImage::Format_RGB888).copy();
  } else {
    // Generate empty/black frame if camera not available
    main = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    cv::putText(main, "Camera Offline / No Feed", cv::Point(150, 240),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    mainImage = QImage((uchar*)main.data, main.cols, main.rows,
                       main.step, QImage::Format_RGB888).copy();
  }
  return mainImage;
}

QImage OpenCVCamera::getCurrentFace() {
  if (!face.empty()) {
    Mat faceDisplay;
    cv::cvtColor(face, faceDisplay, cv::COLOR_BGR2RGB);
    faceImage = QImage(faceDisplay.data, faceDisplay.cols, faceDisplay.rows,
                       faceDisplay.step, QImage::Format_RGB888).copy();
  } else {
    faceImage = QImage();
  }
  return faceImage;
}

void OpenCVCamera::getCurrentFaceMat(Mat& faceOut) {
  if (!face.empty()) {
    face.copyTo(faceOut);
  } else {
    faceOut = Mat();
  }
}

bool OpenCVCamera::detectFaceInImage(const Mat& inputImg, Mat& faceROI) {
  if (inputImg.empty()) return false;
  auto faces = yoloDetector.detectFaces(inputImg);
  if (!faces.empty()) {
    inputImg(faces[0]).copyTo(faceROI);
    return true;
  }
  // If YOLO doesn't find a face bounding box, use full image as ROI
  inputImg.copyTo(faceROI);
  return true;
}
