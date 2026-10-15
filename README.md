# Face Recognition

A C++ GUI application for real-time face detection, feature extraction, and face recognition built with Qt 5, OpenCV, and ONNX Runtime.

## Features

- **Real-Time Face Detection**: Uses a YOLOv11n-face model (`models/yolov11n-face.onnx`) running via ONNX Runtime C++ API.
- **Face Recognition & Matching**: Compares detected face features against an enrolled face database.
- **Database Management**: Enroll individuals directly from the live camera stream or import image files, delete records, or clear the database.
- **Live Video & Log Feed**: Graphical UI built with Qt 5 displaying real-time video feed, face crop, recognition status with similarity confidence, and event logs.

## Prerequisites

Before building, ensure the following dependencies and tools are installed on your system:

- **C++ Compiler**: Supporting C++17 or later (e.g., `g++` 8+)
- **CMake**: Version 3.10 or higher
- **Qt 5**: `Qt5::Core`, `Qt5::Gui`, `Qt5::Widgets` (`qtbase5-dev` on Debian/Ubuntu)
- **OpenCV 4**: `libopencv-dev`
- **Curl**: Required during CMake configuration to download the YOLO ONNX model if not present.

On Ubuntu / Debian systems, install dependencies with:

```bash
sudo apt update
sudo apt install -y build-essential cmake qtbase5-dev libopencv-dev curl
```

*(Note: ONNX Runtime C++ library is automatically downloaded and built/linked by CMake via `FetchContent` during build configuration.)*

## Building

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kiddos/FaceRecognition.git
   cd FaceRecognition
   ```

2. **Create a build directory:**

   ```bash
   mkdir build
   cd build
   ```

3. **Configure the project with CMake:**

   ```bash
   cmake ..
   ```

   *Note: During CMake configuration, the setup script will automatically download the YOLOv11n face detection model (`models/yolov11n-face.onnx`) if it is not already present.*

4. **Compile the executable:**

   ```bash
   make -j$(nproc)
   ```

   The binary `FaceRecognition` will be built inside the `build/` directory.

## Running

Run the executable from the project root or build directory:

```bash
./build/FaceRecognition
```

or directly from within the `build/` directory:

```bash
./FaceRecognition
```

### Usage Instructions

- **Live Video Feed**: Upon launching, the application automatically starts capturing video from the webcam (or generates a test pattern if no camera device is available).
- **Adding Enrolled Persons**:
  - Enter a name in the **Person Name** input box.
  - Click **Add Current Face** to capture the face detected in the live camera feed into the database.
  - Alternatively, click **Import Face Image** to select an image file (`.jpg`, `.png`, `.jpeg`, `.bmp`) from disk.
- **Managing the Database**:
  - Select an enrolled person from the database list and click **Delete Selected** to remove them.
  - Click **Clear Database** to remove all enrolled face records.
- **Recognition Display**: Detected faces are highlighted with bounding boxes in the main display, and the matched person's name and similarity percentage are shown in real time.
