#include "process.h"
#include <stdio.h>

using cv::Vec3b;
using cv::Point;
using cv::Size;
using cv::Rect;
using cv::getRotationMatrix2D;
using cv::saturate_cast;
using cv::max;
using cv::cvtColor;

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
    lbp = Mat::zeros(1, LBP_FEATURE_LENGTH, CV_32FC1);
    Mat gray;

    if (image.channels() == 3) {
      cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
      cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else if (image.channels() == 1) {
      image.copyTo(gray);
    } else {
#ifdef DEBUG
      cout << "ERROR: image null" << endl;
#endif
      return;
    }

    for (int i = 1 ; i < gray.rows - 1 ; i ++) {
      uchar *lastRow = gray.ptr<uchar>(i-1);
      uchar *thisRow = gray.ptr<uchar>(i);
      uchar *nextRow = gray.ptr<uchar>(i+1);
      for (int j = 1 ; j < gray.cols - 1 ; j ++) {
        uint32_t value = 0;
        if (thisRow[j] > lastRow[j-1])
          value += 128;
        if (thisRow[j] > lastRow[j])
          value += 64;
        if (thisRow[j] > lastRow[j+1])
          value += 32;
        if (thisRow[j] > thisRow[j+1])
          value += 16;
        if (thisRow[j] > nextRow[j+1])
          value += 8;
        if (thisRow[j] > nextRow[j])
          value += 4;
        if (thisRow[j] > nextRow[j-1])
          value += 2;
        if (thisRow[j] > thisRow[j-1])
          value += 1;

        lbp.ptr<float>(0)[value] += 1.0f;
      }
    }
  }

  void computeLTP(Mat& image, Mat& ltp, int threshold) {
    Mat gray;
    ltp = Mat::zeros(1, LTP_FEATURE_LENGTH, CV_32FC1);

    // convert to gray image
    if (image.channels() == 3) {
      cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
      cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else if (image.channels() == 1) {
      image.copyTo(gray);
    } else {
#ifdef DEBUG
      cout << "ERROR: image null" << endl;
#endif
      return;
    }

    for (int i = 1 ; i < gray.rows - 1 ; i ++) {
      uchar *lastRow = gray.ptr<uchar>(i-1);
      uchar *thisRow = gray.ptr<uchar>(i);
      uchar *nextRow = gray.ptr<uchar>(i+1);
      for (int j = 1 ; j < gray.cols - 1 ; j ++) {
        uint32_t value = 0;
        if (lastRow[j-1] > thisRow[j] + threshold) {
          value += 4374;    // 2 * 3^7
        }
        if (lastRow[j-1] <= thisRow[j] + threshold &&
                    lastRow[j-1] >= thisRow[j] - threshold) {
          value += 2187;    // 1 * 3^7
        }

        if (lastRow[j] > thisRow[j] + threshold) {
          value += 1458;    // 2 * 3^6
        }
        if (lastRow[j] <= thisRow[j] + threshold &&
                   lastRow[j] >= thisRow[j] - threshold) {
          value += 729;     // 1 * 3^6
        }

        if (lastRow[j+1] > thisRow[j] + threshold) {
          value += 486;     // 2 * 3^5
        }
        if (lastRow[j+1] <= thisRow[j] + threshold &&
                   lastRow[j+1] >= thisRow[j] - threshold) {
          value += 243;     // 1 * 3^5
        }

        if (thisRow[j+1] > thisRow[j] + threshold) {
          value += 162;     // 2 * 3^4
        }
        if (thisRow[j+1] <= thisRow[j] + threshold &&
                   thisRow[j+1] >= thisRow[j] - threshold) {
          value += 81;      // 1 * 3^4
        }

        if (nextRow[j+1] > thisRow[j] + threshold) {
          value += 54;      // 2 * 3^3
        }
        if (nextRow[j+1] <= thisRow[j] + threshold &&
                   nextRow[j+1] >= thisRow[j] - threshold) {
          value += 27;      // 1 * 3^3
        }

        if (nextRow[j] > thisRow[j] + threshold) {
          value += 18;      // 2 * 3^2
        }
        if (nextRow[j] <= thisRow[j] + threshold &&
                   nextRow[j] >= thisRow[j] - threshold) {
          value += 9;       // 1 * 3^2
        }

        if (nextRow[j-1] > thisRow[j] + threshold) {
          value += 6;       // 2 * 3^1
        }
        if (nextRow[j-1] <= thisRow[j] + threshold &&
                   nextRow[j-1] >= thisRow[j] - threshold) {
          value += 3;       // 1 * 3^1
        }

        if (thisRow[j-1] > thisRow[j] + threshold) {
          value += 2;       // 2 * 3^0
        }
        if (thisRow[j-1] <= thisRow[j] + threshold &&
                   thisRow[j-1] >= thisRow[j] - threshold) {
          value += 1;       // 1 * 3^0
        }

        ltp.ptr<float>()[value] ++;
      }
    }
  }

  void computeCSLTP(Mat& image, Mat& csltp, int threshold) {
    csltp = Mat::zeros(1, CSLTP_FEATURE_LENGTH, CV_32FC1);
    Mat gray;

    if (image.channels() == 3) {
      cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
      cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else if (image.channels() == 1) {
      image.copyTo(gray);
    } else {
#ifdef DEBUG
      cout << "ERROR: image null" << endl;
#endif
      return;
    }

    for (int i = 1 ; i < gray.rows - 1 ; i ++) {
      uchar *lastRow = gray.ptr<uchar>(i-1);
      uchar *thisRow = gray.ptr<uchar>(i);
      uchar *nextRow = gray.ptr<uchar>(i+1);
      for (int j = 1 ; j < gray.cols - 1 ; j ++) {
        uint32_t value = 0;
        if (lastRow[j-1] - nextRow[j+1] > threshold) {
          value += 54;
        }
        if (lastRow[j-1] - nextRow[j+1] <= threshold &&
            lastRow[j-1] - nextRow[j+1] >= -threshold) {
          value += 27;
        }

        if (lastRow[j] - nextRow[j] > threshold) {
          value += 18;
        }
        if (lastRow[j] - nextRow[j] <= threshold &&
            lastRow[j] - nextRow[j] >= -threshold) {
          value += 9;
        }

        if (lastRow[j+1] - nextRow[j-1] > threshold) {
          value += 6;
        }
        if (lastRow[j+1] - nextRow[j-1] <= threshold &&
            lastRow[j+1] - nextRow[j-1] >= -threshold) {
          value += 3;
        }

        if (thisRow[j+1] - thisRow[j-1] > threshold) {
          value += 2;
        }
        if (thisRow[j+1] - thisRow[j-1] <= threshold &&
            thisRow[j+1] - thisRow[j-1] >= -threshold) {
          value += 1;
        }

        csltp.ptr<float>()[value] ++;
      }
    }
  }

  static unsigned int sum(Mat& mat) {
    if (mat.type() != CV_8UC1) {
      return 0;
    }

    unsigned int value = 0;
    for (int y = 0 ; y < mat.rows ; y ++) {
      for (int x = 0 ; x < mat.cols ; x ++) {
          const int index = y*mat.step + x;
          value += mat.ptr<uchar>()[index];
      }
    }
    return value;
  }

  void computeHaar(Mat& image, Mat& haar,
                  unsigned int& featureLength) {
    Mat gray;

    if (image.channels() == 3) {
      cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
      cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else if (image.channels() == 1) {
      image.copyTo(gray);
    } else {
#ifdef DEBUG
      cout << "ERROR: image null" << endl;
#endif
      return;
    }

    const unsigned int boxSize = 4;
    const int xBound = gray.cols - boxSize;
    const int yBound = gray.rows - boxSize;
    if (xBound <= 0 || yBound <= 0) {
#ifdef DEBUG
      cout << "ERROR: image too small" << endl;
#endif
      return;
    }

    featureLength = xBound * yBound;
    haar = Mat::zeros(1, featureLength, CV_32FC1);

    for (int y = 0 ; y < yBound ; y ++) {
      for (int x = 0 ; x < xBound ; x ++) {
        const int index = y * xBound + x;
        Rect roi(x, y, boxSize, boxSize);
        Mat computeRegion = gray(roi).clone();
        Mat region1, region2, region3, region4;

        // edge feature 1
        region1 = computeRegion(Rect(0, 0, 2, 4));
        region2 = computeRegion(Rect(2, 0, 2, 4));
        if (sum(region1) > sum(region2)) {
          haar.ptr<float>()[index] += 1;
        }

        // edge feature 2
        region1 = computeRegion(Rect(0, 0, 4, 2));
        region2 = computeRegion(Rect(0, 2, 4, 2));
        if (sum(region1) > sum(region2)) {
          haar.ptr<float>()[index] += 2;
        }

        // line feature 1
        region1 = computeRegion(Rect(0, 0, 1, 4));
        region2 = computeRegion(Rect(1, 0, 2, 4));
        region3 = computeRegion(Rect(3, 0, 1, 4));
        if (sum(region1) + sum(region3) > sum(region2)) {
          haar.ptr<float>()[index] += 4;
        }

        // line feature 2
        region1 = computeRegion(Rect(0, 0, 4, 1));
        region2 = computeRegion(Rect(0, 1, 4, 2));
        region3 = computeRegion(Rect(0, 3, 4, 1));
        if (sum(region1) + sum(region3) > sum(region2)) {
          haar.ptr<float>()[index] += 8;
        }

        // rect feature
        region1 = computeRegion(Rect(0, 0, 2, 2));
        region2 = computeRegion(Rect(2, 0, 2, 2));
        region3 = computeRegion(Rect(0, 2, 2, 2));
        region4 = computeRegion(Rect(2, 2, 2, 2));
        if (sum(region1) + sum(region4) >
            sum(region2) + sum(region3)) {
          haar.ptr<float>()[index] += 16;
        }
      }
    }
  }

}
