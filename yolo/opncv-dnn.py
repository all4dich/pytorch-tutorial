import cv2
import numpy as np
import time # Optional: To calculate FPS

# --- Constants ---
MODEL_PATH = 'yolov5s.onnx' # Path to your YOLOv5s ONNX model
INPUT_WIDTH = 640          # Input width for YOLOv5
INPUT_HEIGHT = 640         # Input height for YOLOv5
SCORE_THRESHOLD = 0.45     # Confidence threshold for filtering detections
NMS_THRESHOLD = 0.45       # Non-Maximum Suppression threshold
CONFIDENCE_THRESHOLD = 0.5 # Object confidence threshold (different from class score threshold)

# COCO class names (standard 80 classes for YOLOv5)
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# --- Helper Functions ---

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resizes and pads image while meeting stride-multiple constraints.
    Maintains aspect ratio. Adds padding ('letterboxing').

    Args:
        im (np.ndarray): Input image BGR.
        new_shape (tuple): Desired output shape (height, width).
        color (tuple): Padding color.
        auto (bool): Minimum rectangle calculation.
        scaleFill (bool): Stretch image to fill new_shape.
        scaleup (bool): Allow scaling up if image is smaller than new_shape.
        stride (int): Stride constraint.

    Returns:
        tuple: (Padded image, (ratio_w, ratio_h), (pad_w, pad_h))
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, original_w, original_h, ratio, pad):
    """
    Draws the bounding box and label on the image.

    Args:
        img (np.ndarray): Original image frame.
        class_id (int): Class index.
        confidence (float): Confidence score.
        x, y, x_plus_w, y_plus_h (int): Bounding box coordinates relative to padded/resized image.
        original_w, original_h (int): Dimensions of the original frame.
        ratio (tuple): Width and height scaling ratios (ratio_w, ratio_h).
        pad (tuple): Width and height padding (pad_w, pad_h).
    """
    # Adjust coordinates back to the original image size
    # 1. Remove padding
    x -= pad[0]
    y -= pad[1]
    x_plus_w -= pad[0]
    y_plus_h -= pad[1]
    # 2. Adjust for scaling ratio
    x = int(x / ratio[0])
    y = int(y / ratio[1])
    x_plus_w = int(x_plus_w / ratio[0])
    y_plus_h = int(y_plus_h / ratio[1])

    # Ensure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    x_plus_w = min(original_w -1, x_plus_w)
    y_plus_h = min(original_h -1, y_plus_h)

    # Get label and color
    label = f'{CLASSES[class_id]}: {confidence:.2f}'
    # Generate a consistent color for each class
    color = ( (class_id * 30) % 255, (class_id * 70) % 255, (class_id * 110) % 255 )

    # Draw rectangle and label
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# --- Main Execution ---

# Load the network
try:
    print("Loading YOLOv5 model...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    print("Model loaded successfully.")
    # Optional: Set preferable backend and target (e.g., for GPU acceleration)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # or for OpenVINO:
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # or DNN_TARGET_OPENCL, DNN_TARGET_MYRIAD
except Exception as e:
    print(f"Error loading the model: {e}")
    print(f"Please ensure '{MODEL_PATH}' exists and is a valid ONNX file.")
    exit()

# Initialize video capture from default camera (index 0)
print("Starting video capture...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

# Set camera resolution (optional, might not work on all cameras)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_frame_time = 0 # For FPS calculation
new_frame_time = 0  # For FPS calculation

print("Processing video stream... Press 'q' to quit.")

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    original_height, original_width = frame.shape[:2]

    # --- Preprocessing ---
    # 1. Letterbox the image to fit the model input size
    img_letterboxed, ratio, pad = letterbox(frame, (INPUT_WIDTH, INPUT_HEIGHT), auto=False) # Use auto=False for consistent padding

    # 2. Create blob from image
    #    - Scale values to 0-1 range (divide by 255.0)
    #    - Resize to INPUT_WIDTH x INPUT_HEIGHT (already done by letterbox)
    #    - Swap BGR to RGB (YOLO expects RGB)
    #    - Do not crop
    blob = cv2.dnn.blobFromImage(img_letterboxed, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)

    # --- Inference ---
    net.setInput(blob)
    try:
        outputs = net.forward() # YOLOv5 has a single output layer
    except Exception as e:
        print(f"Error during model inference: {e}")
        continue # Skip this frame if inference fails

    # --- Post-processing ---
    # The output shape for YOLOv5 ONNX is typically (1, 25200, 85)
    # where 25200 is the number of proposals (anchors * grid cells)
    # and 85 = 4 (bbox) + 1 (object confidence) + 80 (class scores)

    outputs = outputs[0].transpose() # Transpose to (85, 25200) for easier processing

    boxes = []
    confidences = []
    class_ids = []

    # Extract information for each detection proposal
    for det in outputs:
        object_confidence = det[4]

        # Filter based on object confidence
        if object_confidence >= CONFIDENCE_THRESHOLD:
            class_scores = det[5:] # Scores for all classes
            class_id = np.argmax(class_scores)
            max_class_score = class_scores[class_id]

            # Filter based on class score threshold
            if max_class_score >= SCORE_THRESHOLD:
                confidences.append(float(max_class_score * object_confidence)) # Combine scores
                class_ids.append(class_id)

                # Extract bounding box coordinates (center_x, center_y, width, height)
                cx, cy, w, h = det[0:4]

                # Convert from center coordinates to top-left corner (x, y)
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                width = int(w)
                height = int(h)

                boxes.append([x, y, width, height])

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

    # --- Draw results ---
    if len(indices) > 0:
        # Flatten indices array if necessary
        if isinstance(indices, tuple): # Older OpenCV versions might return a tuple
             indices = indices[0]
        indices = indices.flatten()

        for i in indices:
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # Draw the bounding box on the *original* frame
            draw_bounding_box(frame, class_ids[i], confidences[i], x, y, x + w, y + h,
                              original_width, original_height, ratio, pad)

    # Calculate and display FPS (optional)
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("YOLOv5 Object Detection", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Resources released.")