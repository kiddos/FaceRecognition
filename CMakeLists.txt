cmake_minimum_required(VERSION 3.10)
project(FaceRecognition VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ui)

# Auto-download YOLO model weights if missing
set(YOLO_ONNX_FILE "${CMAKE_CURRENT_SOURCE_DIR}/models/yolov11n-face.onnx")

if(NOT EXISTS "${YOLO_ONNX_FILE}")
    message(STATUS "Preparing YOLOv11n face detection model...")
    execute_process(
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_models.sh"
        RESULT_VARIABLE DOWNLOAD_RESULT
    )
    if(NOT DOWNLOAD_RESULT EQUAL 0)
        message(WARNING "Failed to prepare YOLOv11n face detection model.")
    else()
        message(STATUS "YOLOv11n face detection model ready.")
    endif()
endif()

find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(
  onnx
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnx)

if(NOT TARGET onnxruntime)
  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnx_SOURCE_DIR}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${onnx_SOURCE_DIR}/include"
  )
endif()

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
    onnxruntime
)
