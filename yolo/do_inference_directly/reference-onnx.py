import onnx
import onnxruntime as ort
import numpy as np

# Load the ONNX model
model_path = "yolov5n.onnx"  # Replace with the path to your ONNX model
onnx_model = onnx.load(model_path)

# Check the model for errors
onnx.checker.check_model(onnx_model)

# Create an InferenceSession
sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])  # Or 'GPUExecutionProvider' if available