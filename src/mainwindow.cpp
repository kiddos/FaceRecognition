#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDateTime>
#include <iostream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow),
      timer(new QTimer(this)),
      mainDisplay(new ImageViewer()),
      faceDisplay(new ImageViewer()) {
  ui->setupUi(this);

  // Add display widgets to layouts
  ui->mainWidget->layout()->addWidget(mainDisplay);
  ui->faceLayout->addWidget(faceDisplay);

  // Connect button signals
  connect(ui->btnAddCurrentFace, SIGNAL(pressed()), this, SLOT(addCurrentFaceToDB()));
  connect(ui->btnImportFaceImage, SIGNAL(pressed()), this, SLOT(importFaceImageToDB()));
  connect(ui->btnDeletePerson, SIGNAL(pressed()), this, SLOT(deleteSelectedPerson()));
  connect(ui->btnClearDatabase, SIGNAL(pressed()), this, SLOT(clearDatabase()));

  // Connect menu actions
  connect(ui->actionAddPerson, SIGNAL(triggered()), this, SLOT(addCurrentFaceToDB()));
  connect(ui->actionImportImage, SIGNAL(triggered()), this, SLOT(importFaceImageToDB()));
  connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(exitApp()));

  // Setup timer for camera updates
  connect(timer, SIGNAL(timeout()), this, SLOT(setImage()));
  timer->start(CAMERA_INTERVAL);

  // Populate database list
  updateDatabaseList();
  setLog("Application started. YOLO face recognition pipeline active.");
  setLog("Loaded " + QString::number(faceDB.recordCount()) + " records from face database.");
}

MainWindow::~MainWindow() {
  if (timer) {
    timer->stop();
  }
  delete ui;
}

void MainWindow::setLog(const QString& log) {
  QString timeStr = QDateTime::currentDateTime().toString("hh:mm:ss");
  ui->logText->append("[" + timeStr + "] " + log);
}

void MainWindow::updateDatabaseList() {
  ui->listEnrolledPersons->clear();
  auto records = faceDB.getRecords();
  for (const auto& rec : records) {
    QString itemText = QString("ID: %1 | Name: %2").arg(rec.id).arg(rec.name);
    ui->listEnrolledPersons->addItem(itemText);
  }
}

void MainWindow::setImage() {
  // 1. Capture main frame from camera (YOLO detects face and draws bounding box)
  QImage frameImg = camera.getCurrentFrame();
  mainDisplay->setImage(frameImg);

  // 2. Get cropped face image and Mat
  QImage faceImg = camera.getCurrentFace();
  camera.getCurrentFaceMat(currentFaceMat);

  if (!faceImg.isNull() && !currentFaceMat.empty()) {
    faceDisplay->setImage(faceImg);

    // 3. Extract face features & Compare against database
    QString matchedName;
    double similarity = 0.0;
    bool found = faceDB.compareFace(currentFaceMat, matchedName, similarity, 50.0);

    if (found) {
      ui->whoLabel->setText(QString("Recognized: %1\n(Similarity: %2%)")
                               .arg(matchedName)
                               .arg(similarity, 0, 'f', 1));
      ui->statusBar->showMessage("Matched: " + matchedName + " (" + QString::number(similarity, 'f', 1) + "%)");
    } else {
      if (faceDB.recordCount() == 0) {
        ui->whoLabel->setText("YOLO Face Detected\n(Database empty)");
      } else {
        ui->whoLabel->setText(QString("YOLO Face Detected\nUnknown Person (%1%)")
                                 .arg(similarity, 0, 'f', 1));
      }
      ui->statusBar->showMessage("Unknown Person Detected");
    }
  } else {
    faceDisplay->setImage(QImage());
    ui->whoLabel->setText("No Face Detected");
    ui->statusBar->showMessage("No face detected by YOLO");
  }
}

void MainWindow::addCurrentFaceToDB() {
  if (currentFaceMat.empty()) {
    QMessageBox::warning(this, "No Face Captured",
                         "No face is currently detected by YOLO. Please ensure a face is visible in the frame.");
    return;
  }

  QString personName = ui->lePersonName->text().trimmed();
  if (personName.isEmpty()) {
    bool ok = false;
    personName = QInputDialog::getText(this, "Enter Person Name",
                                       "Name of the person:", QLineEdit::Normal,
                                       "", &ok).trimmed();
    if (!ok || personName.isEmpty()) {
      setLog("Cancelled adding face to database.");
      return;
    }
  }

  bool success = faceDB.addFace(personName, currentFaceMat);
  if (success) {
    setLog("Added face for '" + personName + "' to database successfully.");
    ui->lePersonName->clear();
    updateDatabaseList();
    QMessageBox::information(this, "Face Added", "Successfully added face for '" + personName + "' to database!");
  } else {
    setLog("Failed to add face to database.");
    QMessageBox::critical(this, "Error", "Failed to add face to database.");
  }
}

void MainWindow::importFaceImageToDB() {
  QString filePath = QFileDialog::getOpenFileName(this, "Import Face Image",
                                                  QDir::homePath(),
                                                  "Image Files (*.jpg *.png *.jpeg *.bmp)");
  if (filePath.isEmpty()) return;

  cv::Mat importedImg = cv::imread(filePath.toStdString());
  if (importedImg.empty()) {
    QMessageBox::warning(this, "Invalid Image", "Failed to load the selected image file.");
    return;
  }

  // Detect face using YOLO
  cv::Mat faceROI;
  camera.detectFaceInImage(importedImg, faceROI);

  QString personName = ui->lePersonName->text().trimmed();
  if (personName.isEmpty()) {
    bool ok = false;
    personName = QInputDialog::getText(this, "Enter Person Name",
                                       "Name of the person in the image:", QLineEdit::Normal,
                                       "", &ok).trimmed();
    if (!ok || personName.isEmpty()) {
      setLog("Cancelled import image to database.");
      return;
    }
  }

  bool success = faceDB.addFace(personName, faceROI);
  if (success) {
    setLog("Imported face image for '" + personName + "' from " + filePath);
    ui->lePersonName->clear();
    updateDatabaseList();
    QMessageBox::information(this, "Image Imported", "Successfully imported face for '" + personName + "' to database!");
  } else {
    setLog("Failed to import face image to database.");
    QMessageBox::critical(this, "Error", "Failed to extract face features and add to database.");
  }
}

void MainWindow::deleteSelectedPerson() {
  QListWidgetItem* item = ui->listEnrolledPersons->currentItem();
  if (!item) {
    QMessageBox::information(this, "Select Person", "Please select a record from the database list to delete.");
    return;
  }

  QString text = item->text();
  // Format: "ID: <id> | Name: <name>"
  int id = 0;
  if (text.startsWith("ID: ")) {
    int endIdx = text.indexOf(" |");
    if (endIdx > 4) {
      id = text.mid(4, endIdx - 4).toInt();
    }
  }

  if (id > 0) {
    if (QMessageBox::question(this, "Confirm Delete", "Are you sure you want to delete this record?",
                              QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
      faceDB.removeFaceRecord(id);
      setLog("Deleted face record ID: " + QString::number(id));
      updateDatabaseList();
    }
  }
}

void MainWindow::clearDatabase() {
  if (faceDB.recordCount() == 0) return;

  if (QMessageBox::question(this, "Confirm Clear DB",
                            "Are you sure you want to clear all enrolled faces from the database?",
                            QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
    faceDB.clear();
    setLog("Cleared face database.");
    updateDatabaseList();
  }
}

void MainWindow::exitApp() {
  close();
}
