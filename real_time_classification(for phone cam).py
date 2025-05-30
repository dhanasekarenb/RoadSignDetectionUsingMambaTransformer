import os
import sys
# Set YOLOv5 path
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.insert(0, YOLOV5_PATH)

import cv2
import torch
import numpy as np
import csv
import time
from datetime import datetime
from PIL import Image
from torchvision import transforms
from mamba_model import MambaTransformerClassifier
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.models.common import DetectMultiBackend


# ======================== Setup ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half_precision = device.type == "cuda"
# Load YOLOv5 model
# YOLOv5 Model Load
# yolo = DetectMultiBackend("yolov5s.pt", device=device, dnn=False) changed(5/7/2025)
yolo = DetectMultiBackend(r"C:\Users\dhana\OneDrive\Desktop\Final year project\yolov5-master\runs\train\traffic_signs\weights\best.pt", device=device, dnn=False)
stride, names = yolo.stride, yolo.names
img_size = 640
if half_precision:
    yolo.model.half()

# Mamba Classifier Load
num_classes = 43
mamba = MambaTransformerClassifier(num_classes).to(device)
mamba.load_state_dict(torch.load("mamba_transformer_best.pth", map_location=device), strict=False)
mamba.eval()
if half_precision:
    mamba.half()

# Image transform (optimized to use torchvision transforms efficiently)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Class labels
class_labels = {
    0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50", 3: "Speed Limit 60",
    4: "Speed Limit 70", 5: "Speed Limit 80", 6: "End Speed Limit 80", 7: "Speed Limit 100",
    8: "Speed Limit 120", 9: "No Overtaking", 10: "No Overtaking by Trucks", 11: "Priority at Next Intersection",
    12: "Priority Road", 13: "Yield", 14: "Stop", 15: "No Vehicles", 16: "No Trucks", 17: "No Entry",
    18: "General Danger", 19: "Curve Left", 20: "Curve Right", 21: "Double Curve", 22: "Uneven Road",
    23: "Slippery Road", 24: "Road Narrows Right", 25: "Road Work", 26: "Traffic Signals", 27: "Pedestrians",
    28: "Children", 29: "Bicycles", 30: "Ice/Snow", 31: "Wild Animals", 32: "End Restriction", 33: "Turn Right",
    34: "Turn Left", 35: "Go Straight", 36: "Go Straight or Right", 37: "Go Straight or Left", 38: "Keep Right",
    39: "Keep Left", 40: "Roundabout", 41: "End of No Overtaking", 42: "End No Overtake Trucks"
}

# CSV writer setup
csv_file = open("detections.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)

def log_detection(label):
    csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label])
    csv_file.flush()

from collections import deque, Counter

# Store detections with timestamps (deque for performance)
detections_history = deque()

def update_detections(label):
    current_time = time.time()
    detections_history.append((current_time, label))

    # Remove old entries (older than 30 seconds)
    while detections_history and current_time - detections_history[0][0] > 30:
        detections_history.popleft()


# ==================== Helper Functions ====================

def detect_shape(c):
    approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
    sides = len(approx)
    if sides == 3:
        return "triangle"
    elif sides == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect = float(w) / h
        return "rhombus" if 0.85 < aspect < 1.15 else "rectangle"
    elif sides == 8:
        return "octagon"
    else:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri == 0:
            return None
        circularity = 4 * np.pi * area / (peri ** 2)
        return "circle" if circularity > 0.75 else None

def save_to_csv(label):
    csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label])
    csv_file.flush()

def process_frame(frame):
    orig = frame.copy()
    img_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    if half_precision:
        img_tensor = img_tensor.half()

    with torch.no_grad():
        pred = yolo(img_tensor)
        det = non_max_suppression(pred, 0.25, 0.45, agnostic=False)[0]

    if det is not None and len(det):
        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], orig.shape).round()
        for *xyxy, conf, cls in reversed(det):
            if conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            roi = orig[y1:y2, x1:x2]
            if roi.shape[0] < 20 or roi.shape[1] < 20:
                continue

            try:
                crop_resized = cv2.resize(roi, (32, 32))
                gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

                if any(detect_shape(c) in ["circle", "triangle", "octagon", "rhombus"] for c in contours):
                    pil_img = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
                    tensor = transform(pil_img).unsqueeze(0).to(device)
                    if half_precision:
                        tensor = tensor.half()

                    with torch.no_grad():
                        out = mamba(tensor)
                        prob = torch.nn.functional.softmax(out, dim=1)
                        conf_val, label_idx = torch.max(prob, dim=1)
                        if conf_val.item() > 0.2:
                            label = class_labels.get(label_idx.item(), "Unknown")
                            log_detection(label)
                            update_detections(label)
                            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(orig, f"{label} ({conf_val.item()*100:.1f}%)", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"[!] Mamba Classification Error: {e}")
    return orig


# ==================== Main Loop ====================

def main():
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        return

    fps = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.flip(frame, -1)  # fix mirror (horizontal flip)
            #rotate
            (h, w) = frame.shape[:2]
            center = (w // 2, h // 2)
            angle = 180  # degrees

            # Get the rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Rotate the frame
            frame = cv2.warpAffine(frame, M, (w, h))


            processed = process_frame(frame)
            draw_top_detections(processed)

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time))
            prev_time = curr_time

            cv2.putText(processed, f"FPS: {fps:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("YOLO + Mamba Real-Time Traffic Sign Detection", processed)

            if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("YOLO + Mamba Real-Time Traffic Sign Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()

def draw_top_detections(frame):
    # Count frequency
    counter = Counter([label for _, label in detections_history])
    top = counter.most_common(5)  # Show top 5 signs

    x, y = 10, 60  # Starting position
    for idx, (label, count) in enumerate(top):
        text = f"{idx+1}. {label} ({count})"
        cv2.putText(frame, text, (x, y + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)

if __name__ == "__main__":
    main()