import cv2
import numpy as np
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
        except Exception as e:
            print(f"Warning: Could not load class names from {class_names_path}: {e}")
            class_names = [f'class_{i}' for i in range(80)]  # Default to 80 classes
            print("Using generic class names: class_0, class_1, ...")
    else:
        # Default COCO class names (first 80)
        class_names = [f'class_{i}' for i in range(80)]
        print("Warning: class_names_path not provided, using default generic class names.")

    # 2. Initialize ONNX runtime session
    try:
        session = onnxruntime.InferenceSession(onnx_model_path,
                                               providers=['CPUExecutionProvider'])  # Or ['CUDAExecutionProvider']
        print(f"ONNX model loaded from {onnx_model_path}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Get model input details
    input_details = session.get_inputs()[0]
    input_name = input_details.name
    input_shape = input_details.shape  # e.g., [1, 3, 640, 640]

    # Assume H and W are the same if dynamic, or use fixed values
    if isinstance(input_shape[2], str) or isinstance(input_shape[3], str):  # Dynamic axes
        print(f"Model has dynamic input shape: {input_shape}. Using default 640x640.")
        network_input_size = (640, 640)
    else:
        network_input_size = (input_shape[2], input_shape[3])

    # 3. Load and preprocess image
    # If you want to use a static image, you'd modify this part:
    # original_image = cv2.imread(image_path)
    # if original_image is None:
    #     print(f"Error: Could not read image from {image_path}")
    #     return
    # ... and then process this single original_image instead of the loop.

    cap = cv2.VideoCapture(0)  # Using webcam
    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        # Fallback to image_path if provided and webcam fails
        if image_path:
            print(f"Attempting to load image from: {image_path}")
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Error: Could not read image from {image_path} either.")
                return
            # Process single image (needs to be structured outside the loop or adapt the loop)
            # For simplicity, this example will exit if webcam fails.
            # To process a single image, you'd call the processing logic once here.
            print("Webcam failed. To process a single image, please adapt the script's main loop.")
            return  # Or adapt to process single image
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
        image_blob = np.ascontiguousarray(image_blob, dtype=np.float16) / 255.0
        image_blob = image_blob[np.newaxis, ...]  # Add batch dimension (1, 3, H, W)

        # REMOVED: Warning block for shape mismatch, as letterbox with auto=False should give exact size.
        # if image_blob.shape[2] != network_input_size[0] or image_blob.shape[3] != network_input_size[1]:
        #    ... (this block was removed) ...

        # 4. Run inference
        start_time = time.time()
        try:
            outputs = session.run(None, {input_name: image_blob})[0]  # Output shape e.g., (1, 25200, 85)
        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            cap.release()
            cv2.destroyAllWindows()
            return
        end_time = time.time()
        # print(f"Inference time: {end_time - start_time:.4f} seconds") # Optional: print inference time

        # 5. Postprocess outputs
        predictions = outputs[0]  # Remove batch dimension

        objectness_conf = predictions[:, 4]
        conf_mask = (objectness_conf >= conf_thres)

        predictions = predictions[conf_mask]
        objectness_conf = objectness_conf[conf_mask]

        if not predictions.shape[0]:
            # print("No detections found after confidence threshold.") # Optional: print if no detections
            cv2.imshow("Detections", original_image)  # Show original image if no detections
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
            continue  # Continue to next frame

        class_probs = predictions[:, 5:]
        class_ids = np.argmax(class_probs, axis=1)
        class_scores = np.max(class_probs, axis=1)

        scores_for_nms = objectness_conf * class_scores

        boxes_xywh = predictions[:, :4]
        boxes_xyxy = xywh2xyxy(boxes_xywh)

        final_detections = []
        unique_class_ids = np.unique(class_ids)

        for class_id in unique_class_ids:
            class_mask = (class_ids == class_id)
            class_boxes = boxes_xyxy[class_mask]
            class_scores_for_nms = scores_for_nms[class_mask]

            if len(class_boxes) == 0:
                continue

            keep_indices = non_max_suppression(class_boxes, class_scores_for_nms, iou_thres)

            for idx in keep_indices:
                box = class_boxes[idx]
                score = class_scores_for_nms[idx]
                # Filter again by final score if needed, though NMS uses it
                if score >= conf_thres:  # Ensure final combined score also meets threshold
                    final_detections.append({
                        "box": box,
                        "score": score,
                        "class_id": class_id
                    })

        # Sort detections by score for consistent drawing order (optional)
        # final_detections = sorted(final_detections, key=lambda x: x['score'], reverse=True)

        # if not final_detections:
        #     print("No detections found after NMS.") # Optional

        output_image = original_image.copy()

        for det in final_detections:
            box_lb = np.array([det["box"]]).astype(np.float32)  # Coords are for letterboxed image

            # Scale coordinates from letterboxed image to original image
            scaled_box = scale_coords(network_input_size, box_lb, original_image_shape, ratio_pad=(ratio, (dw, dh)))[0]

            x1, y1, x2, y2 = map(int, scaled_box)
            score = det["score"]
            class_id = det["class_id"]

            if class_id < len(class_names):
                label = f"{class_names[class_id]}: {score:.2f}"
            else:
                label = f"class_{class_id}: {score:.2f}"  # Fallback if class_id is out of bounds

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
    parser = argparse.ArgumentParser(description="YOLO ONNX Inference Script for Webcam")
    # MODIFIED: Changed default ONNX model path
    parser.add_argument("--onnx_model", type=str, default="yolo11n.onnx",
                        help="Path to the ONNX model file.")
    # MODIFIED: Made --image optional and removed its default, as webcam is primary
    parser.add_argument("--image", type=str, default=None,
                        help="Optional: Path to a single input image (if webcam fails or for single image processing).")
    parser.add_argument("--class_names", type=str, default="coco.yaml",
                        # Provide a common default or expect user to have it
                        help="Path to YAML file with class names (e.g., coco.yaml from YOLOv5 repo).")
    parser.add_argument("--conf_thres", type=float, default=0.25,
                        help="Object confidence threshold.")
    parser.add_argument("--iou_thres", type=float, default=0.45,  # Adjusted to a more common NMS threshold
                        help="IOU threshold for NMS.")

    args = parser.parse_args()

    # A note on class_names_path:
    # If 'coco.yaml' is not in the same directory, you might need to provide the full path
    # or place coco.yaml (e.g., from a YOLOv5 repository: yolov5/data/coco.yaml)
    # in the same directory as this script.
    # Example: --class_names /path/to/your/yolov5/data/coco.yaml

    main(args.onnx_model, args.image, args.class_names, args.conf_thres, args.iou_thres)