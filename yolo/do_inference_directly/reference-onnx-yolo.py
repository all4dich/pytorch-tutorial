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
    if ratio_pad is None:  # calculate from img0_shape
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
        with open(class_names_path, 'r') as f:
            class_names = yaml.safe_load(f)['names']
    else:
        # Default COCO class names (first 80)
        class_names = [f'class_{i}' for i in range(80)]
        print("Warning: class_names_path not provided, using default class names.")

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
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    original_image_shape = original_image.shape[:2]  # H, W

    # Letterbox
    image_letterboxed, ratio, (dw, dh) = letterbox(original_image, new_shape=network_input_size,
                                                   auto=False)  # auto=False for exact shape

    # Convert HWC to CHW, BGR to RGB, normalize
    image_blob = image_letterboxed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image_blob = np.ascontiguousarray(image_blob, dtype=np.float16) / 255.0
    image_blob = image_blob[np.newaxis, ...]  # Add batch dimension (1, 3, H, W)

    if image_blob.shape[2] != network_input_size[0] or image_blob.shape[3] != network_input_size[1]:
        print(
            f"Warning: Preprocessed image shape {image_blob.shape} does not match network input {network_input_size}. Resizing.")
        # This case might happen if auto=True in letterbox and stride wasn't 1. Forcing size:
        temp_image_blob = np.zeros((1, 3, network_input_size[0], network_input_size[1]), dtype=np.float32)
        # A more robust way would be to resize image_blob if needed or ensure letterbox gives exact size
        # For now, this assumes letterbox output matches network_input_size if auto=False
        # If not, one might need to resize image_blob before feeding.
        # This is a simplified handling.

    # 4. Run inference
    start_time = time.time()
    try:
        outputs = session.run(None, {input_name: image_blob})[0]  # Output shape e.g., (1, 25200, 85)
    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        return
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")

    # 5. Postprocess outputs
    # outputs shape: (batch_size, num_predictions, 5 + num_classes)
    # num_predictions = (e.g., 25200 for 640x640 input)
    # For each prediction: [cx, cy, w, h, obj_conf, class1_conf, ..., classN_conf]

    predictions = outputs[0]  # Remove batch dimension

    # Filter out detections with low objectness confidence
    objectness_conf = predictions[:, 4]
    conf_mask = (objectness_conf >= conf_thres)

    predictions = predictions[conf_mask]
    objectness_conf = objectness_conf[conf_mask]

    if not predictions.shape[0]:
        print("No detections found after confidence threshold.")
        # Display original image if no detections
        cv2.imshow("Detections", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Get class scores and class indices
    class_probs = predictions[:, 5:]
    class_ids = np.argmax(class_probs, axis=1)
    class_scores = np.max(class_probs, axis=1)  # Score of the most likely class

    # Overall confidence for filtering (objectness * class_score)
    # You might choose to use just objectness_conf or this combined score for NMS
    # For this example, we use class_scores directly for NMS if they represent P(class|object)
    # and objectness_conf has already filtered P(object)
    # A common approach is to use (objectness_conf * class_scores) as the final score for NMS.
    scores_for_nms = objectness_conf * class_scores

    # Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
    # Coordinates are relative to the letterboxed image (network_input_size)
    boxes_xywh = predictions[:, :4]
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # Perform NMS per class
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
            final_detections.append({
                "box": box,  # (x1, y1, x2, y2) for letterboxed image
                "score": score,
                "class_id": class_id
            })

    if not final_detections:
        print("No detections found after NMS.")
    else:
        print(f"Found {len(final_detections)} detections after NMS.")

    # 6. Scale coordinates and draw boxes on the original image
    # Create a copy of the original image to draw on
    output_image = original_image.copy()

    for det in final_detections:
        box = np.array([det["box"]]).astype(np.float32)  # Has to be 2D for scale_coords
        scaled_box = scale_coords(network_input_size, box, original_image_shape, ratio_pad=(ratio, (dw, dh)))[0]

        x1, y1, x2, y2 = map(int, scaled_box)
        score = det["score"]
        class_id = det["class_id"]
        label = f"{class_names[class_id]}: {score:.2f}"

        # Draw rectangle
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 7. Display/Save image
    cv2.imshow("Detections", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Example: Save the output image
    # output_image_path = image_path.replace('.', '_detected.')
    # cv2.imwrite(output_image_path, output_image)
    # print(f"Processed image saved to {output_image_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 ONNX Inference Script")
    parser.add_argument("--onnx_model", type=str, required=True, help="Path to the ONNX model file.", default="yolov5n.onnx")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.", default="bus.jpg")
    parser.add_argument("--class_names", type=str, default="/Users/sunjoo/workspace/yolo/yolov5/yolov5-7.0/data/coco.yaml",
                        help="Path to YAML file with class names (e.g., coco.yaml).", )
    parser.add_argument("--conf_thres", type=float, default=0.25, help="Object confidence threshold.")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="IOU threshold for NMS.")

    args = parser.parse_args()

    main(args.onnx_model, args.image, args.class_names, args.conf_thres, args.iou_thres)