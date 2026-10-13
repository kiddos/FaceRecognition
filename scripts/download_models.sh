#!/bin/bash
set -e

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

WEIGHTS_PATH="$MODELS_DIR/yolov4-tiny.weights"
CFG_PATH="$MODELS_DIR/yolov4-tiny.cfg"

if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Downloading yolov4-tiny.weights..."
    curl -L "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights" -o "$WEIGHTS_PATH"
fi

if [ ! -f "$CFG_PATH" ]; then
    echo "Downloading yolov4-tiny.cfg..."
    curl -L "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg" -o "$CFG_PATH"
fi

echo "YOLO model files ready in $MODELS_DIR"
