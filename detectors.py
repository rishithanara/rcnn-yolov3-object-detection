"""Detector wrappers and confidence-based fusion."""
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# COCO labels shared by standard pretrained Faster R-CNN and YOLOv3 models.
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

_rcnn = None

def detect_faster_rcnn(frame, threshold=0.35):
    """Torchvision pretrained Faster R-CNN COCO detector (R-CNN-family model)."""
    global _rcnn
    try:
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    except Exception as e:
        raise RuntimeError("Install compatible torch and torchvision packages. See README.md.") from e
    if _rcnn is None:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.eval()
        _rcnn = (model, weights.transforms())
    model, transforms = _rcnn
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    with torch.no_grad(): result = model([transforms(tensor)])[0]
    labels = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
    detections = []
    for box, label_id, score in zip(result["boxes"].cpu().numpy(), result["labels"].cpu().numpy(), result["scores"].cpu().numpy()):
        if float(score) >= threshold and int(label_id) < len(labels):
            label = labels[int(label_id)]
            if label == "N/A": continue
            detections.append({"box": tuple(map(int, box)), "label": label, "score": float(score), "model": "Faster R-CNN"})
    return detections

def detect_yolov3(frame, threshold=0.35):
    """OpenCV DNN YOLOv3 detector. Requires models/yolov3.weights and yolov3.cfg."""
    weights = MODELS_DIR / "yolov3.weights"
    cfg = MODELS_DIR / "yolov3.cfg"
    names = MODELS_DIR / "coco.names"
    missing = [p.name for p in (weights, cfg, names) if not p.exists()]
    if missing:
        raise RuntimeError("Missing YOLOv3 model files: " + ", ".join(missing) + ". Run python download_yolov3.py; if download fails, follow README.md.")
    with open(names, "r", encoding="utf-8") as f: classes = [line.strip() for line in f if line.strip()]
    net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
    # CPU mode maximizes compatibility on Windows; OpenCV DNN uses optimized CPU routines.
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outputs = net.forward(output_names)
    h, w = frame.shape[:2]; boxes = []; scores = []; class_ids = []
    for output in outputs:
        for row in output:
            objectness = float(row[4]); class_scores = row[5:]
            class_id = int(np.argmax(class_scores)); score = objectness * float(class_scores[class_id])
            if score < threshold or class_id >= len(classes): continue
            cx, cy, bw, bh = row[:4] * np.array([w, h, w, h])
            x = int(cx - bw / 2); y = int(cy - bh / 2)
            boxes.append([x, y, int(bw), int(bh)]); scores.append(score); class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.45)
    detections = []
    if len(indices):
        for i in np.array(indices).flatten():
            x, y, bw, bh = boxes[int(i)]
            detections.append({"box": (max(0,x), max(0,y), min(w,x+bw), min(h,y+bh)), "label": classes[class_ids[int(i)]], "score": float(scores[int(i)]), "model": "YOLOv3"})
    return detections

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1); ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    area_a = max(0,ax2-ax1)*max(0,ay2-ay1); area_b=max(0,bx2-bx1)*max(0,by2-by1)
    return inter / max(area_a + area_b - inter, 1e-9)

def fuse_detections(detections, iou_threshold=0.5):
    """Class-aware confidence-based NMS over outputs from both detectors."""
    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in ordered:
        if not any(d["label"] == k["label"] and _iou(d["box"], k["box"]) >= iou_threshold for k in kept):
            kept.append(d)
    return kept

def draw_detections(frame, detections):
    out = frame.copy()
    for d in detections:
        x1,y1,x2,y2 = d["box"]
        color = (50, 210, 70) if d["model"] == "YOLOv3" else (255, 150, 40)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        label = f'{d["label"]} {d["score"]:.2f} [{d["model"]}]'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytop = max(0, y1-th-8)
        cv2.rectangle(out, (x1,ytop), (min(out.shape[1]-1,x1+tw+6),y1), color, -1)
        cv2.putText(out, label, (x1+3,max(th, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return out
