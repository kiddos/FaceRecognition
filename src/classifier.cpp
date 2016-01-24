#include "classifier.h"

#define DEFAULT_CLASSIIFIER_TYPE C_SVC
#define DEFAULT_CLASSIIFIER_KERNEL_TYPE RBF
#define DEFAULT_GAMMA 0.1
#define DEFAULT_C 1
#define DEFAULT_NU 0.1
#define DEFAULT_DEGREE 2
#define DEFAULT_COEF0 0.1
#define DEFAULT_P 0

#define LTP_THRESHOLD 25

// macro
#undef MIN
#define MIN(n1, n2) (n1 < n2 ? n1 : n2)

#ifdef QT_DEBUG
using std::cout;
using std::endl;
#endif

using std::vector;
using std::pair;
using cv::ml::TrainData;
using cv::ml::ROW_SAMPLE;
using cv::imread;
using cv::resize;
using cv::ml::StatModel;

namespace classifier {
// constants
const char* const DEFAULT_BG_DIR = "bg";
const char* const DEFAULT_POS_DIR = "/pos";
const char* const DEFAULT_NEG_DIR = "/pos";
const char* const DEFAULT_MODEL_OUTPUT = "facemodel.xml";
const double DEFAULT_TEST_PERCENT = 0.1;
const double DEFAULT_TRAINING_STEP = 0.05;
const double DEFAULT_IMAGE_SIZE = 64;
const double TEST_ACCURACY_REQUIREMENT = 0.966;
const uint32_t MAX_ITERATION = 360;
// local constants

/***** TrainingDataLoader ******/
TrainingDataLoader::TrainingDataLoader(const LoadingParams params) {
  this->directory = params.directory;
  this->percent = params.percentForTraining;
  this->featureType = params.featureType;
  this->bgDir = params.bgDir;
  this->posDir = params.posDir;
  this->negDir = params.negDir;
  this->imageSize = params.imageSize;
}

void TrainingDataLoader::load(Mat& trainingData,
                              Mat& trainingLabel,
                              map<int,string>& names) {
  // directory constants
  size_t trainingSize = 0, testingSize = 0;
  vector<string> userFiles, exclusion;
  exclusion.push_back(".");
  exclusion.push_back("..");

  scanDir(directory, userFiles, exclusion);
  for (uint32_t i = 0 ; i < userFiles.size() ; i ++) {
    string path;
    vector<string> imagePaths;
    if (strcmp(userFiles[i].c_str(), bgDir.c_str()) == 0) {
      // background images
      path = directory + string(SEPARATOR) + userFiles[i];
    } else {
      // users images
      path = directory + string(SEPARATOR) + userFiles[i] + posDir;
    }
    scanDir(path, imagePaths, exclusion);
    trainingSize += static_cast<size_t>(imagePaths.size() * percent);
    testingSize += static_cast<size_t>(imagePaths.size() * (1-percent));

    // mappings
    names.insert(pair<int,string>((i-userFiles.size()/2),
                                  userFiles[i]));
  }

#ifdef DEBUG
  cout << "trainingSize: " << trainingSize << endl;
  cout << "testingSize: " << testingSize << endl;
#endif

#ifdef QT_DEBUG
  sendMessage(QString("training size: ") +
              QString::number(trainingSize));
#endif

  // prepare the matrix for training
  uint32_t featureLength = 0;
  switch (featureType) {
    case LBP:
      featureLength = process::LBP_FEATURE_LENGTH;
      break;
    case LTP:
      featureLength = process::LTP_FEATURE_LENGTH;
      break;
    case CSLTP:
      featureLength = process::CSLTP_FEATURE_LENGTH;
      break;
  }

#ifdef DEBUG
  cout << "feature length: " << featureLength << endl;
#endif

#ifdef QT_DEBUG
  sendMessage(QString("feature length: ") +
              QString::number(featureLength));
#endif

  // data
  Mat tnd = Mat::zeros(trainingSize, featureLength, CV_32FC1);
  Mat ttd = Mat::zeros(testingSize, featureLength, CV_32FC1);
  // labels
  Mat tnl = Mat::zeros(trainingSize, 1, CV_32SC1);
  Mat ttl = Mat::zeros(testingSize, 1, CV_32SC1);
  Mat image, X;

  // extracting lbp/ctlp data for individual samples
  size_t trainingPos = 0, testingPos = 0;
  for (uint32_t i = 0 ; i < userFiles.size() ; i ++) {
    string path;
    vector<string> imagePaths;
    if (strcmp(userFiles[i].c_str(), bgDir.c_str()) == 0) {
      // background images
      path = directory + string(SEPARATOR) + userFiles[i];
    } else {
      // users images
      path = directory + string(SEPARATOR) + userFiles[i] + posDir;
    }
    scanDir(path, imagePaths, exclusion);

#ifdef DEBUG
    cout << "current training size: " << trainingPos << endl;
#endif

#ifdef QT_DEBUG
    sendMessage(QString("current training size: ") +
                QString::number(trainingPos));
#endif

    // read individual images for training
    size_t trainingImageCount = static_cast<size_t>(imagePaths.size() * percent);
    for (uint32_t j = 0 ; j < trainingImageCount ; j ++) {
      QString processingType;
      string briefMat;
      string imagePath = path + string(SEPARATOR) + imagePaths[j];
      image = imread(imagePath);

      if (image.data) {
        Mat resized = Mat::zeros(imageSize, image.type());
        resize(image, resized, imageSize);
        // compute feature
        switch (featureType) {
          case LBP:
            process::computeLBP(resized, X);
            processingType = "LBP";
            break;
          case LTP:
            process::computeLTP(resized, X, LTP_THRESHOLD);
            processingType = "LTP";
            break;
          case CSLTP:
            process::computeCSLTP(resized, X, LTP_THRESHOLD);
            processingType = "CSLTP";
            break;
        }

#ifdef DEBUG
      cout << "tnd.rows = " << tnd.rows << " j = " << j << endl;
#endif

        TrainingDataLoader::brief(X, briefMat);
        sendMessage(QString("Training: loading image from ") +
                    QString(imagePath.c_str()) +
                    QString(" | processing image with ") +
                    processingType);
        sendMessage(QString("Training sample: ") +
                    QString(briefMat.c_str()));

        // copy the x sample to tnd
        for (uint32_t k = 0 ; k < featureLength ; k ++) {
          tnd.ptr<float>(j + trainingPos)[k] =
              X.ptr<float>(0)[k];
        }

        // set label for this sample
        tnl.ptr<int>(j + trainingPos)[0] = i - userFiles.size() / 2;
      }
    }
    trainingPos += trainingImageCount;

    // read individual images for testing
    size_t testingImageCount = static_cast<size_t>(imagePaths.size() * (1-percent));
    for (uint32_t j = 0 ; j < testingImageCount ; j ++) {
      QString processingType;
      string imagePath = path + string(SEPARATOR) +
          imagePaths[j + (size_t)(imagePaths.size() * percent)];
      image = imread(imagePath);

      if (image.data) {
        Mat resized = Mat::zeros(imageSize, image.type());
        resize(image, resized, imageSize);
        // compute feature
        switch (featureType) {
          case LBP:
            process::computeLBP(resized, X);
            processingType = "LBP";
            break;
          case LTP:
            process::computeLTP(resized, X, LTP_THRESHOLD);
            processingType = "LTP";
            break;
          case CSLTP:
            process::computeCSLTP(resized, X, LTP_THRESHOLD);
            processingType = "CSLTP";
            break;
        }

        sendMessage(QString("Testing: loading image from ") +
                    QString(imagePath.c_str()) +
                    QString(" | processing image with ") +
                    processingType);

        // copy the x sample to tnd
        for (uint32_t k = 0 ; k < featureLength ; k ++) {
          ttd.ptr<float>(j + testingPos)[k] =
              X.ptr<float>(0)[k];
        }
        // set label for this sample
        ttl.ptr<int>(j + testingPos)[0] = i - userFiles.size() / 2;
      }
    }
    testingPos += testingImageCount;
  }

  // debug info
#ifdef DEBUG
  cout << tnd << endl;
  cout << ttd << endl;
  cout << tnl << endl;
  cout << ttl << endl;
#endif

  // put all data/lables into trainingData/Labels
  trainingData = Mat::zeros(trainingSize + testingSize, featureLength, CV_32FC1);
  trainingLabel = Mat::zeros(trainingSize + testingSize, 1, CV_32SC1);

  const uint32_t cols = trainingData.cols;
  for (uint32_t i = 0 ; i < trainingSize ; i ++) {
    // training data
    for (uint32_t j = 0 ; j < cols ; j ++) {
      trainingData.ptr<float>(i)[j] = tnd.ptr<float>(i)[j];
    }
    // training label
    trainingLabel.ptr<int>(i)[0] = tnl.ptr<int>(i)[0];
  }
  for (uint32_t i = 0 ; i < testingSize ; i ++) {
    // testing data
    for (uint32_t j = 0 ; j < cols ; j ++) {
      trainingData.ptr<float>(i + trainingSize)[j] = ttd.ptr<float>(i)[j];
    }
    // testing label
    trainingLabel.ptr<int>(i + trainingSize)[0] = ttl.ptr<int>(i)[0];
  }

#ifdef DEBUG
  cout << trainingData << endl;
  cout << trainingLabel << endl;
#endif
}

void TrainingDataLoader::brief(const Mat& mat, string& str) {
  const int maxRows = 10;
  const int maxCols = 20;
  const int rowLimit = MIN(maxRows, mat.rows);
  const int colLimit = MIN(maxCols, mat.cols);

  str = "[";
  if (rowLimit > 0) {
    for (int i = 0 ; i < rowLimit ; i ++) {
      for (int j = 0 ; j < colLimit ; j ++) {
        switch (mat.type()) {
          case CV_8UC3:
            str += std::to_string(
                  mat.ptr<uchar>(i)[j * mat.channels()]);
            break;
          case CV_8UC1:
            str += std::to_string(
                  mat.ptr<uchar>(i)[j]);
            break;
          case CV_32SC1:
            str += std::to_string(
                  mat.ptr<int>(i)[j]);
            break;
          case CV_32FC1:
            str += std::to_string(
                  static_cast<int>(mat.ptr<float>(i)[j]));
            break;
        }
        if (j != colLimit - 1) str += ",";
      }
    }
  }
  str += ", ...]";
}
/*----- end of TrainingDataLoader -----*/

/****** old training data loading function ******/
void loadTrainingData(LoadingParams params,
                      Mat& trainingData,
                      Mat& trainingLabel,
                      map<int,string>& names) {
  // prepare the params
  const string directory = params.directory;
  const string bgDir = params.bgDir;
  const string posDir = params.posDir;
  const double percent = params.percentForTraining;
  const FeatureType featureType = params.featureType;
  const Size size = params.imageSize;

  size_t trainingSize = 0, testingSize = 0;
  vector<string> userFiles, exclusion;
  exclusion.push_back(".");
  exclusion.push_back("..");

  scanDir(directory, userFiles, exclusion);
  for (uint32_t i = 0 ; i < userFiles.size() ; i ++) {
    string path;
    vector<string> imagePaths;
    if (strcmp(userFiles[i].c_str(), bgDir.c_str()) == 0) {
      // background images
      path = directory + string(SEPARATOR) + userFiles[i];
    } else {
      // users images
      path = directory + string(SEPARATOR) + userFiles[i] + posDir;
    }
    scanDir(path, imagePaths, exclusion);
    trainingSize += static_cast<size_t>(imagePaths.size() * percent);
    testingSize += static_cast<size_t>(imagePaths.size() * (1-percent));

    // mappings
    names.insert(pair<int,string>((i-userFiles.size()/2),
                                  userFiles[i]));
  }

#ifdef DEBUG
  cout << "trainingSize: " << trainingSize << endl;
  cout << "testingSize: " << testingSize << endl;
#endif

  // prepare the matrix for training
  uint32_t featureLength = 0;
  switch (featureType) {
    case LBP:
      featureLength = process::LBP_FEATURE_LENGTH;
      break;
    case LTP:
      featureLength = process::LTP_FEATURE_LENGTH;
      break;
    case CSLTP:
      featureLength = process::CSLTP_FEATURE_LENGTH;
      break;
  }

#ifdef DEBUG
  cout << "feature length: " << featureLength << endl;
#endif

  // data
  Mat tnd = Mat::zeros(trainingSize, featureLength, CV_32FC1);
  Mat ttd = Mat::zeros(testingSize, featureLength, CV_32FC1);
  // labels
  Mat tnl = Mat::zeros(trainingSize, 1, CV_32SC1);
  Mat ttl = Mat::zeros(testingSize, 1, CV_32SC1);
  Mat image, X;

  // extracting feature data for individual samples
  size_t trainingPos = 0, testingPos = 0;
  for (uint32_t i = 0 ; i < userFiles.size() ; i ++) {
    string path;
    vector<string> imagePaths;
    if (strcmp(userFiles[i].c_str(), bgDir.c_str()) == 0) {
      // background images
      path = directory + string(SEPARATOR) + userFiles[i];
    } else {
      // users images
      path = directory + string(SEPARATOR) + userFiles[i] + posDir;
    }
    scanDir(path, imagePaths, exclusion);

#ifdef DEBUG
    cout << "current training size: " << trainingPos << endl;
#endif

    // read individual images for training
    size_t trainingImageCount =
        static_cast<size_t>(imagePaths.size() * percent);

    for (uint32_t j = 0 ; j < trainingImageCount ; j ++) {
#ifdef DEBUG
      cout << "tnd.rows = " << tnd.rows << " j = " << j << endl;
#endif

      string imagePath = path + string(SEPARATOR) + imagePaths[j];
      image = imread(imagePath);

      if (image.data) {
        Mat resized = Mat::zeros(size, image.type());
        resize(image, resized, size);
        // compute feature
        switch (featureType) {
          case LBP:
            process::computeLBP(resized, X);
            break;
          case LTP:
            process::computeLTP(resized, X, LTP_THRESHOLD);
            break;
          case CSLTP:
            process::computeCSLTP(resized, X, LTP_THRESHOLD);
            break;
        }
        // copy the x sample to tnd
        for (uint32_t k = 0 ; k < featureLength ; k ++) {
          tnd.ptr<float>(j + trainingPos)[k] =
              X.ptr<float>(0)[k];
        }

        // set label for this sample
        tnl.ptr<int>(j)[0] = i - userFiles.size() / 2;
      }
    }
    trainingPos += trainingImageCount;

    // read individual images for testing
    size_t testingImageCount =
        static_cast<size_t>(imagePaths.size() * (1-percent));

    for (uint32_t j = 0 ; j < testingImageCount ; j ++) {
      string imagePath = path + string(SEPARATOR) +
          imagePaths[j + (size_t)(imagePaths.size() * percent)];
      image = imread(imagePath);

      if (image.data) {
        Mat resized = Mat::zeros(size, image.type());
        resize(image, resized, size);
        // compute feature
        switch (featureType) {
          case LBP:
            process::computeLBP(resized, X);
            break;
          case LTP:
            process::computeLTP(resized, X, LTP_THRESHOLD);
            break;
          case CSLTP:
            process::computeCSLTP(resized, X, LTP_THRESHOLD);
            break;
        }

        // copy the x sample to tnd
        for (uint32_t k = 0 ; k < featureLength ; k ++) {
          ttd.ptr<float>(j)[k] =
              X.ptr<float>(0)[k];
        }
        // set label for this sample
        ttl.ptr<int>(j + testingPos)[0] = i - userFiles.size() / 2;
      }
    }
    testingPos += testingImageCount;

  }

  // debug info
#ifdef DEBUG
  cout << tnd << endl;
  cout << ttd << endl;
  cout << tnl << endl;
  cout << ttl << endl;
#endif

  // put all data/lables into trainingData/Labels
  trainingData = Mat::zeros(trainingSize + testingSize, featureLength, CV_32FC1);
  trainingLabel = Mat::zeros(trainingSize + testingSize, 1, CV_32SC1);

  const uint32_t cols = trainingData.cols;
  for (uint32_t i = 0 ; i < trainingSize ; i ++) {
    // training data
    for (uint32_t j = 0 ; j < cols ; j ++) {
      trainingData.ptr<float>(i)[j] = tnd.ptr<float>(i)[j];
    }
    // training label
    trainingLabel.ptr<int>(i)[0] = tnl.ptr<int>(i)[0];
  }
  for (uint32_t i = 0 ; i < testingSize ; i ++) {
    // testing data
    for (uint32_t j = 0 ; j < cols ; j ++) {
      trainingData.ptr<float>(i + trainingSize)[j] = ttd.ptr<float>(i)[j];
    }
    // testing label
    trainingLabel.ptr<int>()[i + trainingSize] = ttl.ptr<int>()[i];
  }

#ifdef DEBUG
  cout << trainingData << endl;
  cout << trainingLabel << endl;
#endif
}
/*----- end of old training data loading function -----*/

/****** FaceClassifier ******/
FaceClassifier::FaceClassifier() {
  this->type = DEFAULT_CLASSIIFIER_TYPE;
  this->kernelType = DEFAULT_CLASSIIFIER_KERNEL_TYPE;

  this->gamma = DEFAULT_GAMMA;
  this->c = DEFAULT_C;
  this->nu = DEFAULT_NU;
  this->degree = DEFAULT_DEGREE;
  this->coef0 = DEFAULT_COEF0;
  this->p = DEFAULT_P;

  this->imageSize = Size(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE);

  this->setupSVM();
}

FaceClassifier::FaceClassifier(FaceClassifierParams param) {
  // svm training paramter
  this->type = param.type;
  this->kernelType = param.kernelType;

  this->gamma = param.gamma;
  this->c = param.c;
  this->nu = param.nu;
  this->degree = param.degree;
  this->coef0 = param.coef0;
  this->p = param.p;

  // other parameter
  this->imageSize = param.imageSize;
  this->trainingStep = param.trainingStep;
  this->testPercent = param.testingPercent;

  this->setupSVM();
}

FaceClassifier::FaceClassifier(FaceClassifierParams param,
                               Mat &data, Mat &label) {
  // svm training paramter
  this->type = param.type;
  this->kernelType = param.kernelType;

  this->gamma = param.gamma;
  this->c = param.c;
  this->nu = param.nu;
  this->degree = param.degree;
  this->coef0 = param.coef0;
  this->p = param.p;

  // other parameter
  this->imageSize = param.imageSize;
  this->trainingStep = param.trainingStep;
  this->testPercent = param.testingPercent;

  this->setupSVM();
  this->setupTrainingData(data, label);
}

void FaceClassifier::setupSVM() {
  this->svm = SVM::create();

  switch (this->type) {
    case C_SVC:
      this->svm->setType(SVM::C_SVC);
      break;
    case NU_SVC:
      this->svm->setType(SVM::NU_SVC);
      break;
    case ONE_CLASS:
      this->svm->setType(SVM::ONE_CLASS);
      break;
    case EPS_SVR:
      this->svm->setType(SVM::EPS_SVR);
      break;
    case NU_SVR:
      this->svm->setType(SVM::NU_SVR);
      break;
    default:
#ifdef DEBUG
      fprintf(stderr, "No such type of SVM implemented\n");
      fprintf(stderr, "default to C_SVC\n");
#endif

#ifdef QT_DEBUG
      sendMessage("Warning!! No such type of SVM implmented");
      sendMessage("Warning!! default to C_SVC");
#endif

      this->svm->setType(cv::ml::SVM::C_SVC);
      break;
  }

  switch (this->kernelType) {
    case LINEAR:
      this->svm->setKernel(SVM::LINEAR);
      break;
    case POLY:
      this->svm->setKernel(SVM::POLY);
      break;
    case RBF:
      this->svm->setKernel(SVM::RBF);
      break;
    case SIGMOID:
      this->svm->setKernel(SVM::SIGMOID);
      break;
    default:
#ifdef DEBUG
      fprintf(stderr, "No such kernel type of SVM implemented\n");
      fprintf(stderr, "default to RBF\n");
#endif

#ifdef QT_DEBUG
      sendMessage("Warning!! No such kernel type of SVM implmented");
      sendMessage("Warning!! default to RBF");
#endif

      this->svm->setKernel(SVM::RBF);
      break;
  }

  this->svm->setC(this->c);
  this->svm->setNu(this->nu);
  this->svm->setGamma(this->gamma);
  this->svm->setCoef0(this->coef0);
  this->svm->setP(this->p);
}

void FaceClassifier::setupTrainingData(Mat &data, Mat &label) {
  size_t testingSize = data.rows * testPercent > 1 ?
        static_cast<size_t>(data.rows * testPercent) : 1;
  size_t trainingSize = data.rows - testingSize;

  if (data.type() == CV_32FC1 && label.type() == CV_32SC1) {
    trainingData = Mat::zeros(trainingSize, data.cols, data.type());
    testingData = Mat::zeros(testingSize, data.cols, data.type());
    trainingLabel = Mat::zeros(trainingSize,
                               label.cols, label.type());
    testingLabel = Mat::zeros(testingSize,
                              label.cols, label.type());

    for (int i = 0 ; i < trainingData.rows ; i ++) {
      for (int j = 0 ; j < trainingData.cols ; j ++) {
        trainingData.ptr<float>(i)[j] =
          data.ptr<float>(i)[j];
      }
      trainingLabel.ptr<int>(i)[0] = label.ptr<int>(i)[0];
    }
    for (int i = 0 ; i < testingData.rows ; i ++) {
      for (int j = 0 ; j < testingData.cols ; j ++) {
        testingData.ptr<float>(i)[j] =
          data.ptr<float>(trainingData.rows + i)[j];
      }
      testingLabel.ptr<int>(i)[0] = label.ptr<int>(trainingLabel.rows + i)[0];
    }
  }
}

void FaceClassifier::setImageSize(Size newSize) {
  this->imageSize = newSize;
}

void FaceClassifier::saveModel() {
  this->svm->save(DEFAULT_MODEL_OUTPUT);
}

void FaceClassifier::saveModel(string modelPath) {
  this->svm->save(modelPath);
}

void FaceClassifier::train() {
  if (this->trainingData.data && this->trainingLabel.data &&
      this->testingData.data && this->testingLabel.data) {
    // prepare training data
    Ptr<TrainData> td = TrainData::create(trainingData,
                                          ROW_SAMPLE,
                                          trainingLabel);

#ifdef QT_DEBUG
    sendMessage(QString("decreasing training parameter"));
#endif

    double maxAccuracy = 0;
    double maxGamma = 0;
    double accuracy = 0;

    // cache up original parameters
    this->gammaCache = this->gamma;

    for (unsigned int i = 0 ; i < MAX_ITERATION ; i ++) {
      this->svm->train(td);
      accuracy = this->testAccuracy();
      if (accuracy > maxAccuracy) {
        maxAccuracy = accuracy;
        maxGamma = this->gamma;
      }

#ifdef DEBUG
      fprintf(stdout, "test accuracy: %lf\n", accuracy);
#endif

      sendMessage(QString("test accuracy: ") +
                  QString::number(accuracy) +
                  QString(" | gamma = ") +
                  QString::number(this->gamma) +
                  QString(" | continue to update..."));

      if (accuracy >= TEST_ACCURACY_REQUIREMENT) {
        determineFeatureType();
        sendMessage(QString("test accuracy: ") +
                    QString::number(accuracy) +
                    QString(" | requirement reach | stop training"));
        return;
      }

      double l = 0;
      switch (this->kernelType) {
        case LINEAR:
          break;
        case POLY:
          l = log10(this->gamma);
          l -= trainingStep;
          this->gamma = pow(10, l);
          break;
        case RBF:
          l = log10(this->gamma);
          l -= trainingStep;
          this->gamma = pow(10, l);
          break;
        case SIGMOID:
          l = log10(this->gamma);
          l -= trainingStep;
          this->gamma = pow(10, l);
          break;
      }
      this->setupSVM();
    }

#ifdef QT_DEBUG
    sendMessage(QString("increasing training parameter"));
#endif

    // restore gamma to original value
    this->gamma = this->gammaCache;

    for (unsigned int i = 0 ; i < MAX_ITERATION ; i ++) {
      this->svm->train(td);
      accuracy = this->testAccuracy();
      if (accuracy > maxAccuracy) {
        maxAccuracy = accuracy;
        maxGamma = this->gamma;
      }

#ifdef DEBUG
      fprintf(stdout, "test accuracy: %lf\n", accuracy);
#endif

      sendMessage(QString("test accuracy: ") +
                  QString::number(accuracy) +
                  QString(" | gamma = ") +
                  QString::number(this->gamma) +
                  QString(" | continue to update..."));

      if (accuracy >= TEST_ACCURACY_REQUIREMENT) {
        determineFeatureType();
        sendMessage(QString("test accuracy: ") +
                    QString::number(accuracy) +
                    QString(" | requirement reach | stop training"));
        return;
      }

      double l = 0;
      switch (this->kernelType) {
        case LINEAR:
          break;
        case POLY:
          l = log10(this->gamma);
          l += trainingStep;
          this->gamma = pow(10, l);
          break;
        case RBF:
          l = log10(this->gamma);
          l += trainingStep;
          this->gamma = pow(10, l);
          break;
        case SIGMOID:
          l = log10(this->gamma);
          l += trainingStep;
          this->gamma = pow(10, l);
          break;
      }
      this->setupSVM();
    }

    // if desire accuracy cannot reach
    // use the gamma with highest testing accuracy
    this->gamma = maxGamma;
    this->setupSVM();
    this->svm->train(td);
    accuracy = this->testAccuracy();
    sendMessage(QString("cannot reach desire test accuracy | ") +
                QString("using max test accuracy gamma | ") +
                QString("test accuracy: ") +
                QString::number(accuracy) +
                QString(" | gamma = ") +
                QString::number(this->gamma));

    determineFeatureType();
  } else {
#ifdef DEBUG
    fprintf(stderr, "No training data and label prepared\n");
#endif

#ifdef QT_DEBUG
    sendMessage("no training data and label prepared");
#endif
  }
}

void FaceClassifier::train(Mat& data, Mat& label) {
  this->setupTrainingData(data, label);
  this->train();
}

int FaceClassifier::predict(Mat& sample) {
  if (this->svm->isTrained()) {
    if (sample.rows == 1 && sample.cols == trainingData.cols &&
        sample.type() == trainingData.type()) {
      return this->svm->predict(sample);
    } else {
#ifdef DEBUG
      fprintf(stderr, "SVM not trained\n");
#endif

#ifdef QT_DEBUG
      sendMessage("SVM not trained");
#endif
      return INT_MAX;
    }
  } else {
#ifdef DEBUG
    fprintf(stderr, "SVM not trained\n");
#endif

#ifdef QT_DEBUG
    sendMessage("SVM not trained");
#endif
    return INT_MAX;
  }
}

int FaceClassifier::predictImageSample(Mat& imageSample) {
  Mat sample, resized;
  string briefMat;

  resized = Mat::zeros(imageSize, imageSample.type());
  resize(imageSample, resized, imageSize);

  switch (this->featureType) {
    case LBP:
      process::computeLBP(resized, sample);
      break;
    case LTP:
      process::computeLTP(resized, sample, LTP_THRESHOLD);
      break;
    case CSLTP:
      process::computeCSLTP(resized, sample, LTP_THRESHOLD);
      break;
  }

  // debug sample matrix
#ifdef DEBUG
  cout << sample << endl;
#endif

#ifdef QT_DEBUG
  TrainingDataLoader::brief(sample, briefMat);
  sendMessage(QString("sample mat: ") + QString(briefMat.c_str()));
#endif

  // if the svm is train then predict the sample
  if (this->svm->isTrained()) {
#ifdef DEBUG
      cout << "feature length: " << this->svm->getVarCount() << endl;
      cout << "sample length: " << sample.cols << endl;
#endif

      return this->svm->predict(sample);
  } else {
#ifdef DEBUG
    fprintf(stderr, "SVM not trained\n");
#endif

#ifdef QT_DEBUG
    sendMessage("SVM not trained");
#endif
    return INT_MAX;
  }
}

bool FaceClassifier::load(string modelPath) {
  try {
    svm = StatModel::load<SVM>(modelPath);
    determineFeatureType();
    return true;
  } catch (cv::Exception e) {
    sendMessage("Error: Not a valid svm:");
    sendMessage(e.msg.c_str());
    return false;
  }
  return false;
}

double FaceClassifier::testAccuracy() {
  if (this->svm->isTrained()) {
    size_t correct = 0;
    Mat testResult;
    this->svm->predict(testingData, testResult);

    for (int i = 0 ; i < testResult.rows ; i ++) {
      if (static_cast<int>(testResult.ptr<float>(i)[0]) ==
          testingLabel.ptr<int>(i)[0])
        correct ++;
    }
    return static_cast<double>(correct) / testingLabel.rows;
  } else {
#ifdef DEBUG
    fprintf(stderr, "SVM not trained\n");
#endif

#ifdef QT_DEBUG
    sendMessage("SVM not trained");
#endif
    return 0;
  }
}

bool FaceClassifier::isLoaded() {
  return svm->isTrained();
}

void FaceClassifier::determineFeatureType() {
  switch (this->svm->getVarCount()) {
    case process::LBP_FEATURE_LENGTH:
      this->featureType = LBP;
      break;
    case process::LTP_FEATURE_LENGTH:
      this->featureType = LTP;
      break;
    case process::CSLTP_FEATURE_LENGTH:
      this->featureType = CSLTP;
      break;
  }
}

FeatureType FaceClassifier::getFeatureType() {
  return this->featureType;
}

} // classifier namespace

// undefine constants
#undef DEFAULT_CLASSIIFIER_TYPE
#undef DEFAULT_CLASSIIFIER_KERNEL_TYPE
#undef DEFAULT_GAMMA
#undef DEFAULT_C
#undef DEFAULT_NU
#undef DEFAULT_DEGREE
#undef DEFAULT_COEF0
#undef DEFAULT_P

#undef LTP_THRESHOLD

#undef MIN
