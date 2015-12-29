#include "process.h"
#include <stdio.h>

using namespace cv;
using namespace std;

namespace process {
  void changeBrightness(Mat& image, double alpha) {
    Mat newImg = Mat::zeros(image.size(), image.type());

    for (int y = 0 ; y < image.rows; y ++) {
      for (int x = 0 ; x < image.cols; x ++) {
        for (int c = 0 ; c < image.channels() ; c ++) {
          newImg.at<Vec3b>(y, x)[c] =
            saturate_cast<uchar>(alpha*
                image.at<Vec3b>(y, x)[c]);
        }
      }
    }

    for (int y = 0 ; y < image.rows ; y ++) {
      uchar* dst = image.data + y * image.step;
      uchar* src = newImg.data + y * newImg.step;
      for (int x = 0 ; x < image.cols ; x ++) {
        dst[x*image.channels()] = src[x*newImg.channels()];
        dst[x*image.channels()+1] = src[x*newImg.channels()+1];
        dst[x*image.channels()+2] = src[x*newImg.channels()+2];
      }
    }
  }

  void changeBrightness(Mat& image, double alpha, double beta) {
    Mat newImg = Mat::zeros(image.size(), image.type());

    for (int y = 0 ; y < image.rows; y ++) {
      for (int x = 0 ; x < image.cols; x ++) {
        for (int c = 0 ; c < image.channels() ; c ++) {
          newImg.at<Vec3b>(y, x)[c] =
            saturate_cast<uchar>(
                alpha * image.at<Vec3b>(y, x)[c] + beta);
        }
      }
    }

    for (int y = 0 ; y < image.rows ; y ++) {
      uchar* dst = image.data + y * image.step;
      uchar* src = newImg.data + y * newImg.step;
      for (int x = 0 ; x < image.cols ; x ++) {
        dst[x*image.channels()] = src[x*newImg.channels()];
        dst[x*image.channels()+1] = src[x*newImg.channels()+1];
        dst[x*image.channels()+2] = src[x*newImg.channels()+2];
      }
    }
  }

  void rotateImage(Mat& image, const double deg) {
    Mat newImg;
    const double scale = 1.0;
    const int len = max(image.cols, image.rows);
    const Point center(len / 2.0, len / 2.0);

    Mat rotation = getRotationMatrix2D(center, deg, scale);

    warpAffine(image, newImg, rotation, Size(len, len));

    for (int y = 0 ; y < image.rows ; y ++) {
      uchar* dst = image.data + y * image.step;
      uchar* src = newImg.data + y * newImg.step;
      for (int x = 0 ; x < image.cols ; x ++) {
        dst[x*image.channels()] = src[x*newImg.channels()];
        dst[x*image.channels()+1] = src[x*newImg.channels()+1];
        dst[x*image.channels()+2] = src[x*newImg.channels()+2];
      }
    }
  }

  void computeLBP(Mat& image, Mat& lbp) {
    if (lbp.type() != CV_32SC1 || lbp.cols != 256)
      return;

    for (int i = 0 ; i < lbp.cols ; i ++) lbp.ptr<int>()[i] = 0;

    Mat gray;
    if (image.channels() == 3)
      cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
      cvtColor(image, gray, CV_BGRA2GRAY);
    for (int i = 1 ; i < gray.rows - 1 ; i ++) {
      uchar *lastRow = gray.data + (i-1) * gray.step;
      uchar *thisRow = gray.data + (i) * gray.step;
      uchar *nextRow = gray.data + (i+1) * gray.step;
      for (int j = 1 ; j < gray.cols - 1 ; j ++) {
        uint32_t value = 0;
        if (thisRow[j] > lastRow[j-1])
          value += 128;
        else if (thisRow[j] > lastRow[j])
          value += 64;
        else if (thisRow[j] > lastRow[j+1])
          value += 32;
        else if (thisRow[j] > thisRow[j+1])
          value += 16;
        else if (thisRow[j] > nextRow[j+1])
          value += 8;
        else if (thisRow[j] > nextRow[j])
          value += 4;
        else if (thisRow[j] > nextRow[j-1])
          value += 2;
        else if (thisRow[j] > thisRow[j-1])
          value += 1;

        lbp.ptr<int>()[value] ++;
      }
    }
  }
}