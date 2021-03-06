#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QPixmap>
#include <QImage>
#include <QString>
#include <QMap>
#include <QFile>
#include <QXmlStreamWriter>
#include <QXmlStreamReader>
#include <QMessageBox>
#include <QInputDialog>
#include <QFileDialog>
#include <QLineEdit>

#include "opencvcamera.h"
#include "imageviewer.h"
#include "trainingtask.h"
#include "classifier.h"

#define FACE_IMAGE_ROOT_DIR "faces"
#define FACE_MODEL_BASE_NAME "facemodel"
#define FACE_MODEL_EXTENSION ".xml"
#define FACE_MODEL_BASE_DIR "svmmodel"
#define FACE_EXTRA_INFO_BASENAME "extra"
#define IMAGE_OUTPUT_EXTENSION ".jpg"
#define NAME_MAP "names.xml"

#define LIST_NAME "list"
#define ENTRY_NAME "entry"
#define KEY_NAME "key"
#define VALUE_NAME "value"
#define SELECT_TEXT "Select one..."

#define INTERVAL 66
#define PERCENT 0.76
#define TRAINING_STEP_LEFT -1
#define TRAINING_STEP_DELTA 0.01

using classifier::FaceClassifier;
using classifier::FeatureType;
using cv::Mat;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
  Q_OBJECT
 public:
  explicit MainWindow(QWidget *parent = 0);
  ~MainWindow();
  void writeMap();
  void readMap();
  void loadClassifier(const QString modelPath,
                      const QString extraPath);
  void loadNameList();
  void loadNameMap();

 public slots:
  void setImage();
  void train();
  void trainingComplete(QString modelPath,
                        QString extraPath,
                        QMap<int, QString> names);
  void takePicture();
  void resume();
  void setLog(QString log);
  void addTrainingData();
  void addNewPerson();
  void adjustTrainingStep(int value);
  void addNewPersonWithPrompt(bool);
  void importModel(bool);
  void exportModel(bool);
  void exit(bool);

 private:
  Ui::MainWindow *ui = nullptr;
  OpenCVCamera camera;
  QTimer* timer = nullptr;
  ImageViewer* mainDisplay = nullptr;
  ImageViewer* faceDisplay = nullptr;
  TrainingTask* trainingTask = nullptr;
  FaceClassifier* faceClassifier = nullptr;
  QMap<int, QString> names;
  Mat face;
  bool pictureTaken = false;
  const char* FACE_IMAGE_DIR = FACE_IMAGE_ROOT_DIR;
  const char* MODEL_BASE_NAME = FACE_MODEL_BASE_NAME;
  const char* MODEL_EXTENSION = FACE_MODEL_EXTENSION;
  const char* MODEL_BASE_DIR = FACE_MODEL_BASE_DIR;
  const char* EXTRA_INFO_BASENAME = FACE_EXTRA_INFO_BASENAME;
  const char* BG_IMAGE_DIR = DEFAULT_BG_DIR;
  const char* POS_DIR = DEFAULT_POS_DIR;
  const char* NEG_DIR = DEFAULT_NEG_DIR;
  const char* IMAGE_OUTPUT = IMAGE_OUTPUT_EXTENSION;
  const char* MAPPING_FILE = NAME_MAP;
  const char* LIST = LIST_NAME;
  const char* ENTRY = ENTRY_NAME;
  const char* KEY = KEY_NAME;
  const char* VALUE = VALUE_NAME;
  const char* SELECT = SELECT_TEXT;
  const int CAMEAR_INTERVAL = INTERVAL;
  const double LOADING_PERCENT = PERCENT;
  const double IMAGE_SIZE = classifier::DEFAULT_IMAGE_SIZE;
  const double TRAINING_STEP = classifier::DEFAULT_TRAINING_STEP;
  const double LEFT = TRAINING_STEP_LEFT;
  const double DELTA = TRAINING_STEP_DELTA;
};

#undef FACE_IMAGE_ROOT_DIR
#undef FACE_MODEL_BASE_NAME
#undef FACE_MODEL_EXTENSION
#undef IMAGE_OUTPUT_EXTENSION
#undef NAME_MAP

#undef LIST_NAME
#undef ENTRY_NAME
#undef KEY_NAME
#undef VALUE_NAME
#undef SELECT_TEXT

#undef INTERVAL
#undef PERCENT
#undef TRAINING_STEP_LEFT
#undef TRAINING_STEP_DELTA

#endif // MAINWINDOW_H
