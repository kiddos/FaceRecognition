cmake_minimum_required(VERSION 3.10)
project(FaceRecognition VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ui)

# Auto-download YOLO model weights if missing
set(YOLO_WEIGHTS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/models/yolov4-tiny.weights")
set(YOLO_WEIGHTS_URL "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights")

if(NOT EXISTS "${YOLO_WEIGHTS_FILE}")
    message(STATUS "Downloading YOLOv4-tiny weights from ${YOLO_WEIGHTS_URL}...")
    file(DOWNLOAD "${YOLO_WEIGHTS_URL}" "${YOLO_WEIGHTS_FILE}"
         SHOW_PROGRESS
         STATUS DOWNLOAD_STATUS)
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(WARNING "Failed to download YOLO weights. Status: ${DOWNLOAD_STATUS}")
    else()
        message(STATUS "YOLOv4-tiny weights downloaded successfully.")
    endif()
endif()

find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED)

include_directories(
    src
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

set(SOURCES
    src/main.cpp
    src/mainwindow.cpp
    src/classifier.cpp
    src/common.cpp
    src/process.cpp
    src/opencvcamera.cpp
    src/imageviewer.cpp
    src/trainingtask.cpp
    src/yolofacedetector.cpp
    src/facedatabase.cpp
)

set(HEADERS
    src/mainwindow.h
    src/classifier.h
    src/common.h
    src/process.h
    src/opencvcamera.h
    src/imageviewer.h
    src/trainingtask.h
    src/yolofacedetector.h
    src/facedatabase.h
)

set(FORMS
    ui/mainwindow.ui
)

add_executable(FaceRecognition
    ${SOURCES}
    ${HEADERS}
    ${FORMS}
)

target_link_libraries(FaceRecognition PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    ${OpenCV_LIBS}
)
