import cv2
import numpy as np
from onnxruntime import SessionOptions
import onnxruntime
import argparse
import yaml  # For loading class names
import time  # For benchmarking
from collections import Counter

# Removed: from torch.cpu import stream (unused import)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resizes and pads an image to a new shape while maintaining the aspect ratio.
    This is a common preprocessing step for object detection models to ensure
    input images have consistent dimensions.
    Source: Adapted from Ultralytics YOLOv5 utils/augmentations.py

    Args:
        img (np.array): Input image (HWC format).
        new_shape (tuple): Target shape (height, width) for the output image. Can be an int for square shape.
        color (tuple): Padding color (BGR).
        auto (bool): If True, pad to the minimum rectangle. If False, pad to the exact new_shape.
        scaleFill (bool): If True, stretch the image to fill new_shape.
        scaleup (bool): If True, allow scaling up the image if new_shape is larger.
        stride (int): Stride of the model, used for auto padding to a multiple of stride.

    Returns:
        tuple: (letterboxed_img, ratio, (dw, dh))
            - letterboxed_img (np.array): The resized and padded image.
            - ratio (tuple): Scaling ratio (width_ratio, height_ratio).
            - (dw, dh) (tuple): Padding added to the left/right and top/bottom sides respectively.
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    # Calculate the scaling ratio needed to fit the image into new_shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        # If scaleup is False, don't enlarge the image if it's smaller than new_shape
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios (assuming aspect ratio is preserved)
    # Calculate the new dimensions of the image after resizing with ratio 'r'
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # Calculate the total padding needed to reach new_shape
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        # For auto padding, ensure padding is a multiple of the stride
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        # If scaleFill is True, no padding is added, image is stretched
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios (different if stretched)

    # Divide padding into 2 sides (left/right and top/bottom)
    dw /= 2
    dh /= 2

    # Resize the image
    if shape[::-1] != new_unpad:  # resize if the new size is different
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add border (padding)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # Return letterboxed image, scaling ratio, and padding on one side
    return img, ratio, (dw, dh)  # dw, dh are padding on one side (left/top)


def xywh2xyxy(x):
    """
    Converts bounding boxes from [center_x, center_y, width, height] format
    to [top_left_x, top_left_y, bottom_right_x, bottom_right_y] format.

    Args:
        x (np.array): Nx4 array of bounding boxes in [x, y, w, h] format.

    Returns:
        np.array: Nx4 array of bounding boxes in [x1, y1, x2, y2] format.
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x = center_x - width / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y = center_y - height / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x = center_x + width / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y = center_y + height / 2
    return y


def non_max_suppression(boxes, scores, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) on bounding boxes.
    NMS removes redundant overlapping boxes, keeping only the most confident ones.

    Args:
        boxes (np.array): Px4 array of bounding boxes in (x1, y1, x2, y2) format.
        scores (np.array): P array of confidence scores for each box.
        iou_threshold (float): IoU threshold for NMS. Boxes with IoU > iou_threshold
                               with a higher scoring box are suppressed.

    Returns:
        list: Indices of the boxes to keep after NMS.
    """
    # If no boxes, return empty list
    if boxes.shape[0] == 0:
        return []

    # Sort indices by scores in descending order
    idxs = scores.argsort()[::-1]

    # Extract coordinates and calculate areas
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # List to store indices of boxes to keep
    keep = []

    # Process boxes while there are still indices to consider
    while idxs.size > 0:
        # Take the index of the box with the highest score among remaining boxes
        i = idxs[0]
        # Add this index to the keep list
        keep.append(i)

        # Calculate intersection coordinates with all other remaining boxes
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        # Calculate intersection area
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # Calculate IoU (Intersection over Union)
        iou = inter / (areas[i] + areas[idxs[1:]] - inter)

        # Keep indices where IoU is less than the threshold
        # These are the boxes that do not overlap significantly with the current highest scoring box
        idxs = idxs[1:][iou < iou_threshold]

    # Return the indices of the boxes that were kept
    return keep


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescales bounding box coordinates (in xyxy format) from the dimensions
    of the letterboxed image (img1_shape) back to the dimensions of the
    original image (img0_shape).

    Args:
        img1_shape (tuple): Shape (height, width) of the image the coords are currently for (e.g., network input size).
        coords (np.array): Nx4 array of bounding boxes in (x1, y1, x2, y2) format, relative to img1_shape.
        img0_shape (tuple): Shape (height, width) of the original image.
        ratio_pad (tuple, optional): (ratio, (pad_w, pad_h)) from letterbox function.
                                     If None, it's calculated from img0_shape.

    Returns:
        np.array: Nx4 array of bounding boxes in (x1, y1, x2, y2) format, relative to img0_shape.
    """
    # Calculate gain and padding if not provided (should match letterbox calculation)
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_w = (img1_shape[1] - img0_shape[1] * gain) / 2
        pad_h = (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        # Use the gain and padding returned by letterbox
        gain = ratio_pad[0][0]  # Assuming width and height ratio are the same
        pad_w, pad_h = ratio_pad[1]

    # Adjust coordinates by subtracting padding
    coords[:, [0, 2]] -= pad_w  # x1, x2 -= left_pad
    coords[:, [1, 3]] -= pad_h  # y1, y2 -= top_pad

    # Rescale coordinates by dividing by the gain
    coords[:, :4] /= gain

    # Clip bounding box coordinates to be within the original image dimensions
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2 clipped to [0, original_width]
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2 clipped to [0, original_height]

    return coords


def process_mask(mask_coeffs, mask_prototypes, original_image_shape, letterboxed_shape, ratio_pad, mask_threshold=0.5):
    """
    Processes mask coefficients and prototypes to generate a binary segmentation mask
    for a single instance, scaled and aligned with the original image dimensions.

    Args:
        mask_coeffs (np.array): Mask coefficients for a single instance (num_coeffs,).
        mask_prototypes (np.array): Mask prototypes (num_coeffs, proto_h, proto_w).
        original_image_shape (tuple): Shape of the original image (orig_h, orig_w).
        letterboxed_shape (tuple): Shape of the letterboxed image (lb_h, lb_w), e.g., network input size.
        ratio_pad (tuple): (ratio, (pad_w, pad_h)) from letterbox. 'ratio' is (r_w, r_h), pad_w is left_pad, pad_h is top_pad.
        mask_threshold (float): Threshold (0-1) to binarize the mask probabilities.

    Returns:
        np.array: Binary mask (0 or 1) as a NumPy array with the same shape as original_image_shape.
                  Returns a zero mask if processing fails or mask is empty.
    """
    # Get dimensions from mask prototypes
    num_coeffs, proto_h, proto_w = mask_prototypes.shape

    # 1. Generate low-resolution instance mask by matrix multiplication
    # This combines the global prototypes using instance-specific coefficients
    # Reshape prototypes to (num_coeffs, proto_h * proto_w) for matrix multiplication
    # Result is (proto_h * proto_w,)
    instance_mask_low_res = mask_coeffs @ mask_prototypes.reshape(num_coeffs, -1)
    # Reshape back to the prototype dimensions (proto_h, proto_w)
    instance_mask_low_res = instance_mask_low_res.reshape(proto_h, proto_w)

    # 2. Apply sigmoid activation
    # Convert raw mask scores to probabilities (0 to 1)
    instance_mask_sigmoid = 1 / (1 + np.exp(-instance_mask_low_res))

    # 3. Upsample the low-resolution mask to the letterboxed image size (network input size)
    # This increases the mask resolution to match the input size of the network
    instance_mask_upsampled_to_lb = cv2.resize(
        instance_mask_sigmoid,
        (letterboxed_shape[1], letterboxed_shape[0]),  # (width, height) for cv2.resize
        interpolation=cv2.INTER_LINEAR  # Use linear interpolation for smooth resizing
    )

    # 4. Crop the mask to the region corresponding to the original image within the letterboxed image
    # The letterboxed image contains the original image plus padding. We need to crop out the padding.
    # ratio_pad = ( (r_w, r_h), (left_pad, top_pad) )
    # gain_w, gain_h = ratio_pad[0] # Should be same if aspect ratio preserved
    gain = ratio_pad[0][0]  # Assuming r_w = r_h = r, the scaling factor applied to the original image
    left_pad, top_pad = ratio_pad[1] # Padding added to the left and top sides

    orig_h, orig_w = original_image_shape # Original image dimensions

    # Calculate the coordinates of the original image content within the letterboxed image
    # These are the pixel coordinates in the letterboxed image where the original image starts and ends
    img_top_in_lb = int(round(top_pad))
    img_left_in_lb = int(round(left_pad))
    # The bottom/right coordinates are calculated based on the original size scaled by the gain, plus padding
    img_bottom_in_lb = int(round(top_pad + orig_h * gain))
    img_right_in_lb = int(round(left_pad + orig_w * gain))

    # Ensure cropping indices are within the bounds of the upsampled mask
    img_bottom_in_lb = min(img_bottom_in_lb, letterboxed_shape[0])
    img_right_in_lb = min(img_right_in_lb, letterboxed_shape[1])

    # Crop the upsampled mask using the calculated coordinates
    mask_cropped_to_orig_region = instance_mask_upsampled_to_lb[img_top_in_lb:img_bottom_in_lb,
                                  img_left_in_lb:img_right_in_lb]

    # Check if the cropped region is empty (e.g., due to calculation errors or object entirely in padding)
    if mask_cropped_to_orig_region.size == 0:
        # Return a zero mask with the original image shape if cropping resulted in an empty array
        return np.zeros(original_image_shape, dtype=np.uint8)

    # 5. Resize the cropped mask to the original image dimensions
    # This is the final step to get the mask at the same resolution as the input frame
    final_mask_original_res = cv2.resize(
        mask_cropped_to_orig_region,
        (orig_w, orig_h),  # (width, height) for cv2.resize
        interpolation=cv2.INTER_LINEAR
    )

    # 6. Threshold the mask probabilities to get a binary mask (0 or 1)
    # Pixels with probability > mask_threshold are considered part of the object
    return (final_mask_original_res > mask_threshold).astype(np.uint8)


def main(onnx_model_path, image_path, class_names_path, conf_thres=0.25, iou_thres=0.45, source=0):
    """
    Main function to run YOLO ONNX segmentation inference on a video stream or image.

    Args:
        onnx_model_path (str): Path to the ONNX segmentation model file.
        image_path (str, optional): Path to a single input image. Used if video source fails.
        class_names_path (str): Path to the YAML file containing class names.
        conf_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for Non-Maximum Suppression (NMS).
        source (str or int): Video file path, webcam ID (e.g., '0', '1'), or RTSP stream URL.
    """
    # 1. Load class names from YAML file
    if class_names_path:
        try:
            with open(class_names_path, 'r') as f:
                class_names = yaml.safe_load(f)['names']
            print(f"Loaded {len(class_names)} class names from {class_names_path}")
        except Exception as e:
            # Fallback to generic class names if loading fails
            print(f"Warning: Could not load class names from {class_names_path}: {e}")
            class_names = [f'class_{i}' for i in range(80)] # Default to 80 classes (common for COCO)
            print(f"Using generic class names: class_0 ... class_{len(class_names) - 1}")
    else:
        # Fallback if class_names_path is not provided
        class_names = [f'class_{i}' for i in range(80)]  # Default COCO classes
        print(f"Warning: class_names_path not provided, using default {len(class_names)} generic class names.")

    # Generate random colors for each class for mask visualization
    # Using a fixed seed (3) ensures the colors are the same every time the script runs
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(len(class_names), 3)).astype(np.uint8)


    # 2. Initialize ONNX runtime session
    try:
        # Create SessionOptions to configure the ONNX Runtime session
        sess_options = SessionOptions()
        # Set logging levels for verbose output
        # LogSeverity: 0:VERBOSE, 1:INFO, 2:WARNING, 3:ERROR, 4:FATAL
        sess_options.log_severity_level = 0 # Set minimum severity to VERBOSE
        sess_options.log_verbosity_level = 0 # Set verbosity level to VERBOSE (most detailed)

        # Create the InferenceSession
        # providers specifies the execution backend (CPU, CUDA, etc.)
        session = onnxruntime.InferenceSession(onnx_model_path,
                                               sess_options=sess_options, # Pass the configured options
                                               providers=['CPUExecutionProvider']) # Use CPU, or 'CUDAExecutionProvider' if available

        print(f"ONNX model loaded from {onnx_model_path}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return # Exit if model loading fails

    # Get model input details (name and shape)
    input_details = session.get_inputs()[0]
    input_name = input_details.name
    input_shape = input_details.shape # e.g., [1, 3, 640, 640]

    # Determine the network input size (height, width)
    # Handle dynamic input shapes (where H or W might be represented as strings)
    if isinstance(input_shape[2], str) or isinstance(input_shape[3], str):
        print(f"Model has dynamic input shape: {input_shape}. Using default 640x640 for processing.")
        network_input_size = (640, 640)  # (height, width) - common default for YOLO models
    else:
        network_input_size = (input_shape[2], input_shape[3])  # (height, width) from model input shape
    print(f"Network input size set to: {network_input_size}")

    # 3. Initialize Video Capture
    try:
        # Attempt to convert source to integer (for webcam ID)
        stream_source = int(source)
    except ValueError:
        # If conversion fails, treat source as a file path or URL
        stream_source = source

    # Open the video capture source
    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        print(f"Error: Could not open video stream from source: {stream_source}.")
        # Basic fallback logic: if webcam failed and an image_path was provided
        if image_path and isinstance(stream_source, int):
             print(f"Attempting to load image from: {image_path}")
             original_image_single = cv2.imread(image_path)
             if original_image_single is None:
                 print(f"Error: Could not read image from {image_path} either.")
                 return # Exit if image loading also fails
             # TODO: Implement single image processing logic here if needed
             # Currently, the script is structured for a video loop.
             # To process a single image, you would call the processing logic once here
             # and then display/save the result before exiting.
             print("Single image processing not fully implemented in this loop structure. Exiting.")
             return # Exit if webcam failed and single image processing isn't handled

        return # Exit if video source cannot be opened

    print(f"Starting video stream from {stream_source}. Press 'q' to quit.")

    # 4. Main Video Processing Loop
    while True:
        # Read a frame from the video source
        ret, original_image = cap.read()

        # Check if frame was read successfully
        if not ret:
            print("End of video stream or error reading frame. Exiting loop.")
            break # Exit the loop if no frame is read

        # Get the shape of the original image (height, width)
        original_image_shape = original_image.shape[:2]  # H, W

        # 5. Preprocess Frame
        # Resize and pad the original image to the network input size
        image_letterboxed, ratio, (pad_w, pad_h) = letterbox(original_image, new_shape=network_input_size,
                                                             auto=False, scaleup=True)

        # Prepare the image blob for the model:
        # - Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        # - Reverse channel order from BGR (OpenCV default) to RGB (common model input) [::-1]
        image_blob = image_letterboxed.transpose((2, 0, 1))[::-1]
        # - Convert to float32 and normalize pixel values to [0, 1]
        image_blob = np.ascontiguousarray(image_blob, dtype=np.float32) / 255.0
        # - Add a batch dimension (Batch=1)
        image_blob = image_blob[np.newaxis, ...] # Shape becomes (1, 3, H, W)

        # 6. Run Inference
        start_time = time.time()
        try:
            # Run the ONNX model inference
            # session.run returns a list of output tensors
            # For segmentation, expect two outputs:
            # model_outputs[0]: detections (e.g., shape (1, num_attributes, num_proposals))
            # model_outputs[1]: mask_prototypes (e.g., shape (1, num_mask_coeffs, proto_h, proto_w))
            model_outputs = session.run(None, {input_name: image_blob})
        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            break # Exit loop on inference error
        end_time = time.time()
        # print(f"Inference time: {end_time - start_time:.4f} seconds") # Optional: print inference time

        # 7. Postprocess Outputs for Segmentation (Anchor-Free Model)

        # Check if the expected number of outputs is received
        if len(model_outputs) < 2:
            print("Error: Segmentation model should output at least 2 tensors (detections and masks).")
            continue # Skip to the next frame if output format is unexpected

        # Extract the raw detection output and mask prototypes
        detection_raw_output = model_outputs[0]
        mask_prototypes_output = model_outputs[1]

        # Remove the batch dimension from prototypes
        # proto shape is typically (num_mask_coeffs, proto_h, proto_w)
        proto = mask_prototypes_output[0]
        # Determine the number of mask coefficients from the prototypes shape
        num_mask_coeffs = proto.shape[0]

        # Process the raw detection output
        # Assuming detection_raw_output is (batch, num_attributes, num_proposals)
        # num_attributes = 4 (box) + num_classes + num_mask_coeffs
        predictions_with_batch = detection_raw_output # Shape e.g., (1, 112, 8400) for 80 classes, 32 coeffs
        predictions_attrs_first = predictions_with_batch[0] # Remove batch dim: (112, 8400)

        # Transpose to get shape (num_proposals, num_attributes)
        # This makes it easier to slice by proposal
        predictions = predictions_attrs_first.transpose(1, 0) # Shape e.g., (8400, 112)

        # Validate the number of attributes per prediction
        expected_attributes = 4 + len(class_names) + num_mask_coeffs
        if predictions.shape[1] != expected_attributes:
            print(f"Warning: Output shape mismatch. Expected {expected_attributes} attributes "
                  f"(4 bbox + {len(class_names)} classes + {num_mask_coeffs} mask_coeffs), but got {predictions.shape[1]}. "
                  "Post-processing might be incorrect. Please verify yolo11n-seg.onnx output format.")
            # Depending on the severity or if this is flexible, you might choose to continue or exit.
            # For now, we'll try to proceed but this is a critical check.

        # Extract bounding boxes (cx, cy, w, h), class scores, and mask coefficients
        # Slicing based on the assumed attribute order
        boxes_xywh = predictions[:, :4] # First 4 columns are box coords [cx, cy, w, h]
        class_scores_all = predictions[:, 4 : 4 + len(class_names)] # Next num_classes columns are class scores
        mask_coeffs_all = predictions[:, 4 + len(class_names) : 4 + len(class_names) + num_mask_coeffs] # Last num_mask_coeffs columns are mask coefficients

        # Find the class ID and the score for the class with the highest confidence for each prediction
        class_ids = np.argmax(class_scores_all, axis=1) # Index of the highest score for each proposal
        max_scores = np.max(class_scores_all, axis=1)  # The highest score itself (P(class_i|obj)P(obj))

        # Apply confidence threshold using the max scores
        conf_mask = (max_scores >= conf_thres) # Boolean mask: True for proposals above threshold

        # If no detections are above the confidence threshold, display the original image and continue
        if not np.any(conf_mask):
            cv2.imshow("Segmentations", original_image) # Show original image if no detections
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
            continue # Skip the rest of the processing for this frame

        # Filter detections based on the confidence mask
        boxes_xywh_f = boxes_xywh[conf_mask] # Filter boxes
        class_ids_f = class_ids[conf_mask] # Filter class IDs
        scores_f = max_scores[conf_mask] # Filter scores (used for NMS)
        mask_coeffs_f = mask_coeffs_all[conf_mask] # Filter mask coefficients

        # Convert filtered boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format
        # This format is required for the NMS function
        boxes_xyxy_f = xywh2xyxy(boxes_xywh_f)

        # List to store final detections after NMS
        final_detections = []
        # Get unique class IDs present in the filtered detections
        unique_class_ids_f = np.unique(class_ids_f)

        # Perform NMS separately for each class
        for class_id_val in unique_class_ids_f:
            # Create a mask to select detections belonging to the current class ID
            current_class_mask_indices = (class_ids_f == class_id_val)

            # Select boxes, scores, and mask coefficients for the current class
            current_class_boxes = boxes_xyxy_f[current_class_mask_indices]
            current_class_scores = scores_f[current_class_mask_indices]
            current_class_mask_coeffs = mask_coeffs_f[current_class_mask_indices]

            # If no boxes for this class, continue to the next class
            if len(current_class_boxes) == 0:
                continue

            # Apply Non-Maximum Suppression to the boxes of the current class
            keep_indices = non_max_suppression(current_class_boxes, current_class_scores, iou_thres)

            # Append the kept detections for this class to the final list
            for idx in keep_indices:
                final_detections.append({
                    "box": current_class_boxes[idx],  # Box in xyxy format (relative to letterboxed image)
                    "score": current_class_scores[idx], # Confidence score
                    "class_id": class_id_val, # Class ID
                    "mask_coeffs": current_class_mask_coeffs[idx] # Mask coefficients for this instance
                })

        # 8. Visualize Results (Masks, Boxes, Labels)

        # Create a copy of the original image to draw on
        output_image = original_image.copy()
        # Create a separate overlay image for drawing masks.
        # This allows blending all masks at once for better visual results.
        mask_color_overlay = np.zeros_like(output_image, dtype=np.uint8)

        # Process and draw masks first
        for det in final_detections:
            class_id = det["class_id"]

            # Process mask coefficients and prototypes to get the binary mask
            # The mask is generated at the original image's resolution
            final_instance_mask = process_mask(
                det['mask_coeffs'], # Coefficients for this specific instance
                proto, # Global mask prototypes
                original_image_shape, # Shape of the original frame
                network_input_size,  # Shape of the letterboxed input to the network
                (ratio, (pad_w, pad_h)),  # Scaling ratio and padding from letterbox
                mask_threshold=0.5 # Threshold to binarize the mask
            )

            # If the processed mask is not empty (i.e., contains some object pixels)
            if np.any(final_instance_mask):
                # Get the color for the class (using modulo to handle cases where class_id > len(colors))
                color = colors[class_id % len(colors)].tolist()
                # Apply the color to the mask_color_overlay where the mask is 1
                mask_color_overlay[final_instance_mask == 1] = color

        # Blend the mask overlay with the original image
        # cv2.addWeighted(src1, alpha, src2, beta, gamma)
        # alpha=1.0 means full opacity for original_image
        # beta=0.4 means 40% opacity for mask_color_overlay (adjust for desired transparency)
        # gamma=0 is a scalar added to the result (usually 0)
        output_image = cv2.addWeighted(output_image, 1.0, mask_color_overlay, 0.4, 0)

        # Draw bounding boxes and labels on top of the blended image
        all_class_names = []
        all_class_ids = []
        for det in final_detections:
            # Get the box coordinates (which are currently relative to the letterboxed image)
            box_lb = np.array([det["box"]]).astype(np.float32)

            # Scale the box coordinates back to the original image dimensions
            scaled_box = \
            scale_coords(network_input_size, box_lb, original_image_shape, ratio_pad=(ratio, (pad_w, pad_h)))[0]
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, scaled_box)

            # Get score and class ID
            score = det["score"]
            class_id = det["class_id"]

            # Create the label text
            if class_id < len(class_names):
                label_text = f"{class_names[class_id]}: {score:.2f}"
                all_class_names.append(class_names[class_id])
            else:
                label_text = f"class_{class_id}: {score:.2f}" # Fallback if class ID is out of bounds
                all_class_ids.append(class_id)

            # Draw bounding box rectangle
            # Use the same color as the mask for consistency
            cv2.rectangle(output_image, (x1, y1), (x2, y2), colors[class_id % len(colors)].tolist(), 2)

            # Determine label position (slightly above the box, or below if too close to top edge)
            label_y_pos = y1 - 10 if y1 - 10 > 10 else y1 + 20
            # Draw label text
            cv2.putText(output_image, label_text, (x1, label_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_id % len(colors)].tolist(), 2) # Use same color

        # Display the final output image with segmentations, boxes, and labels
        cv2.imshow("Segmentations", output_image) # Changed window title
        # Count occurrences of detected classes
        counter_class_name = Counter(all_class_names)
        counter_class_id = Counter(all_class_ids)
        class_name_status = ""
        class_id_status = ""

        # Format class names with their counts
        for class_name in sorted(counter_class_name.keys()):
            class_name_status += f"{class_name}: {counter_class_name[class_name]}, "

        # Format class IDs with their counts (for classes not in the class_names list)
        for class_id in sorted(counter_class_id.keys()):
            class_id_status += f"class_{class_id}: {counter_class_id[class_id]}, "

        # Remove trailing comma and space if present
        class_name_status = class_name_status.rstrip(", ")
        class_id_status = class_id_status.rstrip(", ")

        # Print detection summary
        if class_name_status:
            print(f"Detected classes: {class_name_status}")
        if class_id_status:
            print(f"Detected unknown classes: {class_id_status}")

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # 9. Cleanup
    cap.release() # Release the video capture object
    cv2.destroyAllWindows() # Close all OpenCV windows


if __name__ == '__main__':
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="YOLO ONNX Segmentation Script for Webcam/Video")

    # Add arguments
    parser.add_argument("--onnx_model", type=str, default="yolo11n-seg.onnx",  # Updated default model name
                        help="Path to the ONNX segmentation model file.")
    parser.add_argument("--image", type=str, default=None,
                        help="Optional: Path to a single input image (if webcam/video source fails or for single image mode).")
    parser.add_argument("--class_names", type=str, default="coco.yaml",
                        help="Path to YAML file with class names (e.g., coco.yaml from YOLOv8 repo).")
    parser.add_argument("--conf_thres", type=float, default=0.25,
                        help="Object confidence threshold.")
    parser.add_argument("--iou_thres", type=float, default=0.45,
                        help="IOU threshold for NMS.")
    parser.add_argument("--source", type=str, default="0",  # Default to webcam "0"
                        help="Video file path or webcam ID (e.g., 0, 1) or RTSP stream URL.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.onnx_model, args.image, args.class_names, args.conf_thres, args.iou_thres, args.source)