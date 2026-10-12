#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QPixmap>
#include <QImage>
#include <QString>
#include <QMessageBox>
#include <QInputDialog>
#include <QFileDialog>
#include <QLineEdit>
#include <QListWidget>

#include "opencvcamera.h"
#include "imageviewer.h"
#include "facedatabase.h"

#define CAMERA_INTERVAL 66

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget *parent = 0);
  ~MainWindow();

  void setLog(const QString& log);
  void updateDatabaseList();

 public slots:
  void setImage();
  void addCurrentFaceToDB();
  void importFaceImageToDB();
  void deleteSelectedPerson();
  void clearDatabase();
  void exitApp();

 private:
  Ui::MainWindow *ui = nullptr;
  OpenCVCamera camera;
  QTimer* timer = nullptr;
  ImageViewer* mainDisplay = nullptr;
  ImageViewer* faceDisplay = nullptr;
  FaceDatabase faceDB;
  cv::Mat currentFaceMat;
};

#endif // MAINWINDOW_H
