import cv2
import numpy as np
from onnxruntime import SessionOptions
import onnxruntime
import argparse
import yaml  # For loading class names
import time  # For benchmarking


# Removed: from torch.cpu import stream (unused)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints.
    Source: Ultralytics YOLOv5 utils/augmentations.py
    """
    shape = img.shape[:2]  # current shape [height, width]
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
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)  # dw, dh are padding on one side (left/top)


def xywh2xyxy(x):
    """
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.
    """
    if boxes.shape[0] == 0:
        return []
    idxs = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[idxs[1:]] - inter)
        idxs = idxs[1:][iou < iou_threshold]
    return keep


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape.
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_w = (img1_shape[1] - img0_shape[1] * gain) / 2
        pad_h = (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad_w, pad_h = ratio_pad[1]

    coords[:, [0, 2]] -= pad_w  # x padding
    coords[:, [1, 3]] -= pad_h  # y padding
    coords[:, :4] /= gain

    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
    return coords


def process_mask(mask_coeffs, mask_prototypes, original_image_shape, letterboxed_shape, ratio_pad, mask_threshold=0.5):
    """
    Process mask coefficients and prototypes to generate a binary mask aligned with the original image.

    Args:
        mask_coeffs (np.array): Mask coefficients for a single instance (num_coeffs,).
        mask_prototypes (np.array): Mask prototypes (num_coeffs, proto_h, proto_w).
        original_image_shape (tuple): Shape of the original image (orig_h, orig_w).
        letterboxed_shape (tuple): Shape of the letterboxed image (lb_h, lb_w), e.g., network input size.
        ratio_pad (tuple): (ratio, (pad_w, pad_h)) from letterbox. 'ratio' is (r_w, r_h), pad_w is left_pad, pad_h is top_pad.
        mask_threshold (float): Threshold to binarize the mask.

    Returns:
        np.array: Binary mask (0 or 1) aligned with the original image dimensions.
    """
    num_coeffs, proto_h, proto_w = mask_prototypes.shape

    # 1. Generate low-resolution instance mask by matrix multiplication
    instance_mask_low_res = mask_coeffs @ mask_prototypes.reshape(num_coeffs, -1)
    instance_mask_low_res = instance_mask_low_res.reshape(proto_h, proto_w)

    # 2. Apply sigmoid
    instance_mask_sigmoid = 1 / (1 + np.exp(-instance_mask_low_res))

    # 3. Upsample to letterboxed image size (network input size)
    instance_mask_upsampled_to_lb = cv2.resize(
        instance_mask_sigmoid,
        (letterboxed_shape[1], letterboxed_shape[0]),  # (width, height) for cv2.resize
        interpolation=cv2.INTER_LINEAR
    )

    # 4. Crop the mask to the region of the original image within the letterboxed image
    # ratio_pad = ( (r_w, r_h), (left_pad, top_pad) )
    # gain_w, gain_h = ratio_pad[0] # Should be same if aspect ratio preserved
    gain = ratio_pad[0][0]  # Assuming r_w = r_h = r
    left_pad, top_pad = ratio_pad[1]

    orig_h, orig_w = original_image_shape

    # Calculate coordinates of the original image within the letterboxed image
    img_top_in_lb = int(round(top_pad))
    img_left_in_lb = int(round(left_pad))
    img_bottom_in_lb = int(round(top_pad + orig_h * gain))
    img_right_in_lb = int(round(left_pad + orig_w * gain))

    # Ensure cropping indices are within bounds of instance_mask_upsampled_to_lb
    img_bottom_in_lb = min(img_bottom_in_lb, letterboxed_shape[0])
    img_right_in_lb = min(img_right_in_lb, letterboxed_shape[1])

    mask_cropped_to_orig_region = instance_mask_upsampled_to_lb[img_top_in_lb:img_bottom_in_lb,
                                  img_left_in_lb:img_right_in_lb]

    if mask_cropped_to_orig_region.size == 0:
        # This can happen if padding calculations or cropping are off, or if the object is entirely in padding
        return np.zeros(original_image_shape, dtype=np.uint8)

    # 5. Resize cropped mask to original image dimensions
    final_mask_original_res = cv2.resize(
        mask_cropped_to_orig_region,
        (orig_w, orig_h),  # (width, height) for cv2.resize
        interpolation=cv2.INTER_LINEAR
    )

    # 6. Threshold to get binary mask
    return (final_mask_original_res > mask_threshold).astype(np.uint8)


def main(onnx_model_path, image_path, class_names_path, conf_thres=0.25, iou_thres=0.45, source=0):
    # 1. Load class names
    if class_names_path:
        try:
            with open(class_names_path, 'r') as f:
                class_names = yaml.safe_load(f)['names']
            print(f"Loaded {len(class_names)} class names from {class_names_path}")
        except Exception as e:
            print(f"Warning: Could not load class names from {class_names_path}: {e}")
            class_names = [f'class_{i}' for i in range(80)]
            print(f"Using generic class names: class_0 ... class_{len(class_names) - 1}")
    else:
        class_names = [f'class_{i}' for i in range(80)]
        print(f"Warning: class_names_path not provided, using default {len(class_names)} generic class names.")

    # Generate random colors for each class for mask visualization
    rng = np.random.default_rng(3)  # Seed for consistent colors
    colors = rng.uniform(0, 255, size=(len(class_names), 3)).astype(np.uint8)

    # 2. Initialize ONNX runtime session
    try:
        sess_options = SessionOptions()
        sess_options.log_severity_level = 0
        sess_options.log_verbosity_level = 0
        session = onnxruntime.InferenceSession(onnx_model_path,
                                               sess_options=sess_options,
                                               providers=['CPUExecutionProvider'])
        print(f"ONNX model loaded from {onnx_model_path}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    input_details = session.get_inputs()[0]
    input_name = input_details.name
    input_shape = input_details.shape

    if isinstance(input_shape[2], str) or isinstance(input_shape[3], str):
        print(f"Model has dynamic input shape: {input_shape}. Using default 640x640 for processing.")
        network_input_size = (640, 640)  # (height, width)
    else:
        network_input_size = (input_shape[2], input_shape[3])  # (height, width)
    print(f"Network input size set to: {network_input_size}")

    try:
        stream_source = int(source)
    except ValueError:
        stream_source = source  # If it's a file path

    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        print(f"Error: Could not open video stream from source: {stream_source}.")
        # Fallback or single image processing logic can be enhanced here
        if image_path and isinstance(stream_source, int):  # if webcam failed and image_path is given
            print(f"Attempting to load image from: {image_path}")
            original_image_single = cv2.imread(image_path)
            if original_image_single is None:
                print(f"Error: Could not read image from {image_path} either.")
                return
            # TODO: Implement single image processing logic here
            print("Single image processing not fully implemented in this loop structure. Exiting.")
            return
        return

    print(f"Starting video stream from {stream_source}. Press 'q' to quit.")
    while True:
        ret, original_image = cap.read()
        if not ret:
            print("End of video stream or error reading frame. Exiting loop.")
            break

        original_image_shape = original_image.shape[:2]  # H, W

        image_letterboxed, ratio, (pad_w, pad_h) = letterbox(original_image, new_shape=network_input_size,
                                                             auto=False, scaleup=True)

        image_blob = image_letterboxed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image_blob = np.ascontiguousarray(image_blob, dtype=np.float32) / 255.0
        image_blob = image_blob[np.newaxis, ...]

        start_time = time.time()
        try:
            # For segmentation, expect two outputs: detections and mask prototypes
            # model_outputs[0]: detections (e.g., 1, 4+num_classes+num_mask_coeffs, 8400)
            # model_outputs[1]: mask_prototypes (e.g., 1, num_mask_coeffs, proto_h, proto_w)
            model_outputs = session.run(None, {input_name: image_blob})
        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            break
        end_time = time.time()
        # print(f"Inference time: {end_time - start_time:.4f} seconds")

        if len(model_outputs) < 2:
            print("Error: Segmentation model should output at least 2 tensors (detections and masks).")
            continue

        detection_raw_output = model_outputs[0]
        mask_prototypes_output = model_outputs[1]

        # Postprocess for SEGMENTATION
        predictions_with_batch = detection_raw_output
        proto = mask_prototypes_output[0]  # Remove batch dim, e.g. (32, 160, 160)
        num_mask_coeffs = proto.shape[0]

        # Assuming detection_raw_output is (batch, num_attributes, num_proposals)
        # num_attributes = 4 (box) + num_classes + num_mask_coeffs
        predictions_attrs_first = predictions_with_batch[0]
        predictions = predictions_attrs_first.transpose(1, 0)  # (num_proposals, num_attributes)

        expected_attributes = 4 + len(class_names) + num_mask_coeffs
        if predictions.shape[1] != expected_attributes:
            print(f"Warning: Output shape mismatch. Expected {expected_attributes} attributes "
                  f"(4 bbox + {len(class_names)} classes + {num_mask_coeffs} mask_coeffs), but got {predictions.shape[1]}. "
                  "Post-processing might be incorrect.")
            # Continue cautiously or add stricter error handling

        boxes_xywh = predictions[:, :4]
        class_scores_all = predictions[:, 4: 4 + len(class_names)]
        mask_coeffs_all = predictions[:, 4 + len(class_names): 4 + len(class_names) + num_mask_coeffs]

        class_ids = np.argmax(class_scores_all, axis=1)
        max_scores = np.max(class_scores_all, axis=1)  # These are class confidences

        conf_mask = (max_scores >= conf_thres)

        if not np.any(conf_mask):
            cv2.imshow("Segmentations", original_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
            continue

        boxes_xywh_f = boxes_xywh[conf_mask]
        class_ids_f = class_ids[conf_mask]
        scores_f = max_scores[conf_mask]
        mask_coeffs_f = mask_coeffs_all[conf_mask]

        boxes_xyxy_f = xywh2xyxy(boxes_xywh_f)

        final_detections = []
        unique_class_ids_f = np.unique(class_ids_f)

        for class_id_val in unique_class_ids_f:
            class_mask_indices = (class_ids_f == class_id_val)

            current_class_boxes = boxes_xyxy_f[class_mask_indices]
            current_class_scores = scores_f[class_mask_indices]
            current_class_mask_coeffs = mask_coeffs_f[class_mask_indices]

            if len(current_class_boxes) == 0:
                continue

            keep_indices = non_max_suppression(current_class_boxes, current_class_scores, iou_thres)

            for idx in keep_indices:
                final_detections.append({
                    "box": current_class_boxes[idx],
                    "score": current_class_scores[idx],
                    "class_id": class_id_val,
                    "mask_coeffs": current_class_mask_coeffs[idx]
                })

        output_image = original_image.copy()
        # Create a combined overlay for all masks to draw once for better blending
        mask_color_overlay = np.zeros_like(output_image, dtype=np.uint8)

        for det in final_detections:
            class_id = det["class_id"]

            # Process and get the binary mask aligned with original image
            final_instance_mask = process_mask(
                det['mask_coeffs'],
                proto,
                original_image_shape,
                network_input_size,  # letterboxed_shape
                (ratio, (pad_w, pad_h)),  # ratio_pad
                mask_threshold=0.5
            )

            if np.any(final_instance_mask):
                color = colors[class_id % len(colors)].tolist()  # Get color for class
                mask_color_overlay[final_instance_mask == 1] = color

        # Blend the combined mask overlay with the output image
        # Adjust alpha (e.g., 0.4 for masks) for blending strength
        output_image = cv2.addWeighted(output_image, 1.0, mask_color_overlay, 0.4, 0)

        # Draw bounding boxes and labels on top of masks
        for det in final_detections:
            box_lb = np.array([det["box"]]).astype(np.float32)  # Coords are for letterboxed image
            scaled_box = \
            scale_coords(network_input_size, box_lb, original_image_shape, ratio_pad=(ratio, (pad_w, pad_h)))[0]
            x1, y1, x2, y2 = map(int, scaled_box)

            score = det["score"]
            class_id = det["class_id"]

            if class_id < len(class_names):
                label_text = f"{class_names[class_id]}: {score:.2f}"
            else:
                label_text = f"class_{class_id}: {score:.2f}"

            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), colors[class_id % len(colors)].tolist(), 2)
            # Draw label
            label_y_pos = y1 - 10 if y1 - 10 > 10 else y1 + 20  # Adjust for visibility
            cv2.putText(output_image, label_text, (x1, label_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_id % len(colors)].tolist(), 2)

        cv2.imshow("Segmentations", output_image)  # Changed window title
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO ONNX Segmentation Script for Webcam/Video")
    parser.add_argument("--onnx_model", type=str, default="yolo11n-seg.onnx",  # Updated default model
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

    args = parser.parse_args()
    main(args.onnx_model, args.image, args.class_names, args.conf_thres, args.iou_thres, args.source)