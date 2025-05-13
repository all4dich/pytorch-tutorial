import cv2
import numpy as np
from onnxruntime import SessionOptions
import onnxruntime
import argparse
import yaml  # For loading class names
import time  # For benchmarking


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
    return img, ratio, (dw, dh)


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

    Parameters:
    boxes (np.array): Px4 array of bounding boxes in (x1, y1, x2, y2) format.
    scores (np.array): P array of confidence scores for each box.
    iou_threshold (float): IoU threshold for NMS.

    Returns:
    list: Indices of boxes to keep.
    """
    if boxes.shape[0] == 0:
        return []

    # Sort by scores
    idxs = scores.argsort()[::-1]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)

        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[idxs[1:]] - inter)

        # Keep boxes with IoU less than threshold
        idxs = idxs[1:][iou < iou_threshold]

    return keep


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape.
    img1_shape: (height, width) of the image the coords are currently for (e.g., network input size)
    coords: (x1, y1, x2, y2)
    img0_shape: (height, width) of the original image
    ratio_pad: (ratio, (pad_w, pad_h)) from letterbox
    """
    if ratio_pad is None:  # calculate from img0_shape if the caller doesn't provide `ratio_pad`
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]  # Assuming width and height ratio are the same
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain  # Rescale to original image size

    # Clip bounding xyxy bounding boxes to image shape (height, width)
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
    return coords


def main(onnx_model_path, image_path, class_names_path, conf_thres=0.25, iou_thres=0.45):
    # 1. Load class names
    if class_names_path:
        try:
            with open(class_names_path, 'r') as f:
                class_names = yaml.safe_load(f)['names']
            print(f"Loaded {len(class_names)} class names from {class_names_path}")
        except Exception as e:
            print(f"Warning: Could not load class names from {class_names_path}: {e}")
            class_names = [f'class_{i}' for i in range(80)]  # Default to 80 classes
            print(f"Using generic class names: class_0 ... class_{len(class_names) - 1}")
    else:
        class_names = [f'class_{i}' for i in range(80)]  # Default COCO classes
        print(f"Warning: class_names_path not provided, using default {len(class_names)} generic class names.")

    # 2. Initialize ONNX runtime session
    try:
        sess_options = SessionOptions()
        sess_options.log_severity_level = 0
        sess_options.log_verbosity_level = 0  # Increase verbosity
        # sess_options.enable_profiling = True
        session = onnxruntime.InferenceSession(onnx_model_path,
                                                sess_options=sess_options,
                                               providers=['CPUExecutionProvider'])  # Or ['CUDAExecutionProvider']
        print(f"ONNX model loaded from {onnx_model_path}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    input_details = session.get_inputs()[0]
    input_name = input_details.name
    input_shape = input_details.shape

    if isinstance(input_shape[2], str) or isinstance(input_shape[3], str):
        print(f"Model has dynamic input shape: {input_shape}. Using default 640x640 for processing.")
        network_input_size = (640, 640)
    else:
        network_input_size = (input_shape[2], input_shape[3])
    print(f"Network input size set to: {network_input_size}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        if image_path:
            print(f"Attempting to load image from: {image_path}")
            # This part would need to be structured to process a single image
            # For now, we'll just exit if webcam fails and an image is the fallback.
            # To process a single image, the main loop logic would be called once.
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Error: Could not read image from {image_path} either.")
                return
            print("Webcam failed. Single image processing needs to be fully implemented outside the loop.")
            # Process single image here (call a dedicated function or adapt the loop)
            # For this example, we'll return.
            return
        else:
            return

    print("Starting webcam feed. Press 'q' to quit.")
    while True:
        ret, original_image = cap.read()
        if not ret:
            print("Error: Could not read frame from video stream. Exiting loop.")
            break  # Changed from return to break to allow cleanup

        original_image_shape = original_image.shape[:2]  # H, W

        # Letterbox
        image_letterboxed, ratio, (dw, dh) = letterbox(original_image, new_shape=network_input_size,
                                                       auto=False,
                                                       scaleup=True)  # Ensure scaleup is True, auto=False for exact shape

        # Convert HWC to CHW, BGR to RGB, normalize
        image_blob = image_letterboxed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # MODIFIED: Changed dtype to np.float32, more common for ONNX models
        image_blob = np.ascontiguousarray(image_blob, dtype=np.float32) / 255.0
        image_blob = image_blob[np.newaxis, ...]

        # 4. Run inference
        start_time = time.time()
        try:
            # model_outputs is a list of output tensors. We assume yolo11n has one primary output.
            model_outputs = session.run(None, {input_name: image_blob})
        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            break  # Exit loop on inference error
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f} seconds")

        # 5. Postprocess outputs for an ANCHOR-FREE model (e.g., yolo11n)
        #    Assumption: Output format is (batch_size, num_predictions, 4_bbox_coords + num_class_scores)
        #    where bbox_coords are [cx, cy, w, h] and class_scores already include objectness.

        # Some models might output (batch_size, 4+num_classes, num_predictions).
        # If so, a transpose would be needed: e.g., model_outputs[0].transpose(0, 2, 1)
        # For yolo11n, verify its exact output dimensions. We assume (batch, num_preds, attrs) here.
        predictions_with_batch = model_outputs[0] # 1, 84, 8400

        # Remove batch dimension
        predictions = predictions_with_batch[0]  # Shape: (84, 8400) => (4 + num_classes, num_predictions)

        predictions = predictions.transpose(1,0) # (84,8400) => (8400,84)

        # Validate the number of attributes per prediction
        expected_attributes = 4 + len(class_names) #expected_attributes: 84
        if predictions.shape[1] != expected_attributes:
            print(f"Warning: Output shape mismatch. Expected {expected_attributes} attributes "
                  f"(4 bbox + {len(class_names)} classes), but got {predictions.shape[1]}. "
                  "Post-processing might be incorrect. Please verify yolo11n.onnx output format.")
            # Depending on the severity or if this is flexible, you might choose to continue or exit.
            # For now, we'll try to proceed but this is a critical check.

        # Extract bounding boxes (cx, cy, w, h) and combined class scores
        boxes_xywh = predictions[:, :4] #predictions.shape : (8400,84), boxes_xywh.shape : (8400,4)
        class_scores_combined = predictions[:, 4:]  # Shape: (num_predictions, num_classes), Maybe [8400, 80]

        # Find the class ID and the score for the class with the highest confidence for each prediction
        class_ids = np.argmax(class_scores_combined, axis=1) # (8400, )
        max_scores = np.max(class_scores_combined, axis=1)  # This is P(class_i|obj)P(obj), (8400, )

        # Apply confidence threshold using the max scores
        conf_mask = (max_scores >= conf_thres) # (8400,): `True` or `False`

        if not np.any(conf_mask):
            cv2.imshow("Detections", original_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
            continue

        # Filter detections based on confidence
        boxes_xywh_filtered = boxes_xywh[conf_mask] # boxes_xywh.shape: (8400,4), boxes_xywh_filtered: (n=18,4)
        class_ids_filtered = class_ids[conf_mask] # class_ids.shape: (8400,), class_ids_filtered: (18,)
        # These are the scores to be used for NMS, already filtered by conf_thres
        scores_for_nms_input = max_scores[conf_mask] # (18, 0)

        # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
        # If yolo11n directly outputs x1,y1,x2,y2, this conversion is not needed
        # and boxes_xyxy_filtered = boxes_xywh_filtered would be used (assuming first 4 are x1y1x2y2).
        boxes_xyxy_filtered = xywh2xyxy(boxes_xywh_filtered)

        final_detections = []
        # Perform NMS per class
        unique_class_ids_after_conf = np.unique(class_ids_filtered)

        for class_id_val in unique_class_ids_after_conf:
            # Create a mask for the current class ID among the confidence-filtered detections
            current_class_mask = (class_ids_filtered == class_id_val)

            # Select boxes and scores for NMS for the current class
            class_boxes_for_nms = boxes_xyxy_filtered[current_class_mask]
            class_scores_for_nms_current_class = scores_for_nms_input[current_class_mask]

            if len(class_boxes_for_nms) == 0:
                continue

            keep_indices = non_max_suppression(class_boxes_for_nms, class_scores_for_nms_current_class, iou_thres)

            for idx in keep_indices:
                final_detections.append({
                    "box": class_boxes_for_nms[idx],  # Already x1,y1,x2,y2
                    "score": class_scores_for_nms_current_class[idx],
                    "class_id": class_id_val
                })

        output_image = original_image.copy()
        for det in final_detections:
            # Box coordinates are for the letterboxed image
            box_lb = np.array([det["box"]]).astype(np.float32)

            scaled_box = scale_coords(network_input_size, box_lb, original_image_shape, ratio_pad=(ratio, (dw, dh)))[0]
            x1, y1, x2, y2 = map(int, scaled_box)

            score = det["score"]
            class_id = det["class_id"]

            if class_id < len(class_names):
                label = f"{class_names[class_id]}: {score:.2f}"
            else:
                label = f"class_{class_id}: {score:.2f}"  # Fallback

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Detections", output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO ONNX Inference Script for Webcam (Anchor-Free Model)")
    parser.add_argument("--onnx_model", type=str, default="yolo11n.onnx",  # Default to yolo11n
                        help="Path to the ONNX model file.")
    parser.add_argument("--image", type=str, default=None,
                        help="Optional: Path to a single input image.")
    parser.add_argument("--class_names", type=str, default="coco.yaml",
                        help="Path to YAML file with class names (e.g., coco.yaml from YOLOv5/v8 repo).")
    parser.add_argument("--conf_thres", type=float, default=0.25,
                        help="Object confidence threshold.")
    parser.add_argument("--iou_thres", type=float, default=0.45,
                        help="IOU threshold for NMS.")

    args = parser.parse_args()
    main(args.onnx_model, args.image, args.class_names, args.conf_thres, args.iou_thres)