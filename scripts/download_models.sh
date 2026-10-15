#!/bin/bash
set -e

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

ONNX_PATH="$MODELS_DIR/yolov11n-face.onnx"

if [ ! -f "$ONNX_PATH" ]; then
    echo "Downloading and preparing YOLOv11n face detection model..."
    python3 -c "
import os, sys, onnx
import numpy as np
from onnx import numpy_helper, helper
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

onnx_target = '$ONNX_PATH'
pt_path = hf_hub_download(repo_id='AdamCodd/YOLOv11n-face-detection', filename='model.pt')
model = YOLO(pt_path)
exported_path = model.export(format='onnx', imgsz=640, dynamic=False, simplify=True, opset=12)

# Fix ONNX graph for OpenCV DNN compatibility
model_onnx = onnx.load(exported_path)
init_dict = {init.name: numpy_helper.to_array(init) for init in model_onnx.graph.initializer}

new_nodes = []
added_initializers = []

for node in model_onnx.graph.node:
    if node.op_type == 'Split' and len(node.input) > 1 and node.input[1] in init_dict:
        split_sizes = init_dict[node.input[1]].tolist()
        if len(set(split_sizes)) > 1:
            axis = 1
            for attr in node.attribute:
                if attr.name == 'axis':
                    axis = attr.i
            start_idx = 0
            in_tensor = node.input[0]
            for idx, (out_name, size) in enumerate(zip(node.output, split_sizes)):
                end_idx = start_idx + size
                start_name = f'{node.name}_slice_{idx}_starts'
                end_name = f'{node.name}_slice_{idx}_ends'
                axes_name = f'{node.name}_slice_{idx}_axes'
                added_initializers.append(helper.make_tensor(start_name, onnx.TensorProto.INT64, [1], [start_idx]))
                added_initializers.append(helper.make_tensor(end_name, onnx.TensorProto.INT64, [1], [end_idx]))
                added_initializers.append(helper.make_tensor(axes_name, onnx.TensorProto.INT64, [1], [axis]))
                slice_node = helper.make_node(
                    'Slice',
                    inputs=[in_tensor, start_name, end_name, axes_name],
                    outputs=[out_name],
                    name=f'{node.name}_slice_{idx}'
                )
                new_nodes.append(slice_node)
                start_idx = end_idx
            continue
    new_nodes.append(node)

model_onnx.graph.ClearField('node')
model_onnx.graph.node.extend(new_nodes)
model_onnx.graph.initializer.extend(added_initializers)

for init in model_onnx.graph.initializer:
    arr = numpy_helper.to_array(init)
    if arr.shape in [(1, 8400), (1, 1, 8400)]:
        if arr.ndim == 2:
            arr = arr.reshape(1, 1, 8400)
        tiled_arr = np.tile(arr, (1, 4, 1))
        new_init = numpy_helper.from_array(tiled_arr, name=init.name)
        init.CopyFrom(new_init)

onnx.save(model_onnx, onnx_target)
print('YOLOv11n face detection ONNX model created successfully at:', onnx_target)
" || {
    echo "Python export failed, downloading direct ONNX model..."
    curl -L "https://huggingface.co/AdamCodd/YOLOv11n-face-detection/resolve/main/model.onnx" -o "$ONNX_PATH.tmp"
    python3 -c "import onnxslim; onnxslim.slim('$ONNX_PATH.tmp', '$ONNX_PATH')" 2>/dev/null || mv "$ONNX_PATH.tmp" "$ONNX_PATH"
    rm -f "$ONNX_PATH.tmp"
}
fi

echo "YOLO model files ready in $MODELS_DIR"
