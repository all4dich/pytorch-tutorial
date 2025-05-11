import torch
import yaml
import cv2
import numpy as np

model_file_path = "yolov5s.yaml"
model_weights_path = "yolov5s.pt"
# 1. Load YAML configuration
with open(model_file_path, "r") as f:
    model_config = yaml.safe_load(f)

# 2. Load .pt model weights
model = torch.load(model_weights_path, weights_only=False, map_location=torch.device('cpu'))  # Assume model.pt is saved in a format that can be loaded directly

# 3. Prepare input data
image = cv2.imread("bus.jpg")
# Preprocess the image (e.g., resize, normalize) according to model_config
# (you'll need to implement the specific preprocessing based on your model)

# 4. Perform inference
with torch.no_grad():
    input_tensor = torch.from_numpy(image).float().unsqueeze(0).permute(0, 3, 1,
                                                                        2)  # Convert to tensor, add batch dimension, permute for channels_first
    predictions = model(input_tensor)

# 5. Post-process predictions (you'll need to implement this based on your model's output format)
# For example, non-maximum suppression (NMS)

# (Example NMS)
#  -  This is a conceptual NMS. Your actual NMS needs to be adapted to your model's output
#
#  -  Let's assume the model outputs a tensor like: [batch, num_classes, num_boxes, box_coords]
#  -  where box_coords = [x1, y1, x2, y2]
#
#  -  Iterate over the predictions for each class
#  -  For each class, apply NMS to select the best bounding boxes
#  -  Sort boxes by confidence score
#  -  Remove overlapping boxes
#  -  Output the remaining boxes with their confidence scores and class labels
#
print(predictions)
