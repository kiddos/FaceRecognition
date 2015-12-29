#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
            QMainWindow(parent),
            ui(new Ui::MainWindow),
            timer(new QTimer()),
            mainDisplay(new ImageViewer()),
            faceDisplay(new ImageViewer()) {
  ui->setupUi(this);

  ui->mainWidget->layout()->addWidget(mainDisplay);
  ui->faceLayout->addWidget(faceDisplay);

  timer->start(INTERVAL);
  connect(timer, SIGNAL(timeout()), this, SLOT(setImage()));
}

MainWindow::~MainWindow() {
  if (timer != nullptr)
    delete timer;
  if (ui != nullptr)
    delete ui;
  if (mainDisplay != nullptr)
      delete mainDisplay;
  if (faceDisplay != nullptr)
      delete faceDisplay;
}

void MainWindow::setImage() {
#ifdef QT_DEBUG
  cout << "getting image" << endl;
#endif

  QImage image = camera.getCurrentFrame();
  QImage face = camera.getCurrentFace();
  mainDisplay->setImage(image);
  if (!face.isNull()) {
    faceDisplay->setImage(face);
  }
}