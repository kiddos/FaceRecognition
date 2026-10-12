QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FaceRecognition
TEMPLATE = app

CONFIG += c++11 link_pkgconfig
DEFINES += DEBUG

PKGCONFIG += opencv4

SOURCES += src/main.cpp\
           src/mainwindow.cpp \
           src/classifier.cpp \
           src/common.cpp \
           src/process.cpp \
           src/opencvcamera.cpp \
           src/imageviewer.cpp \
           src/trainingtask.cpp

HEADERS  += src/mainwindow.h \
            src/classifier.h \
            src/common.h \
            src/process.h \
            src/opencvcamera.h \
            src/imageviewer.h \
            src/trainingtask.h

FORMS    += ui/mainwindow.ui
