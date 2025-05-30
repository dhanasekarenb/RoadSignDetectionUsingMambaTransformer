import os
import sys
import time
import cv2
import torch
import numpy as np
import csv
from datetime import datetime
from PIL import Image
from torchvision import transforms
from collections import deque, Counter
import pygetwindow as gw

# Set YOLOv5 path
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.insert(0, YOLOV5_PATH)

from mamba_model import MambaTransformerClassifier
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.models.common import DetectMultiBackend

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half_precision = device.type == "cuda"

# Load YOLOv5 model
yolo = DetectMultiBackend(
    r"C:\Users\dhana\OneDrive\Desktop\Final year project\yolov5-master\runs\train\traffic_signs\weights\best.pt",
    device=device,
    dnn=False
)
stride, names = yolo.stride, yolo.names
img_size = 640
if half_precision:
    yolo.model.half()

# Load Mamba classifier
num_classes = 43
mamba = MambaTransformerClassifier(num_classes).to(device)
mamba.load_state_dict(torch.load("mamba_transformer_best.pth", map_location=device), strict=False)
mamba.eval()
if half_precision:
    mamba.half()

# Transform for classifier (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Class labels dictionary
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

# Create a directory for saving ROI images that are passed to Mamba
output_dir = "yolo_crops"
os.makedirs(output_dir, exist_ok=True)

# CSV logging
csv_file = open("detections.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)

def log_detection(label):
    csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label])
    csv_file.flush()

# Maintain a short-term history of detections
detections_history = deque()

def update_detections(label):
    current_time = time.time()
    detections_history.append((current_time, label))
    while detections_history and current_time - detections_history[0][0] > 30:
        detections_history.popleft()

def detect_shape(c):
    # Approximate contour and check the number of vertices
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
        # Use circularity for other cases
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri == 0:
            return None
        circularity = 4 * np.pi * area / (peri ** 2)
        return "circle" if circularity > 0.75 else None

def draw_top_predictions(img, top_preds):
    """
    Draws only the predictions with high confidence (> 0.9).
    Also, highlights only the top-1 prediction.
    """
    x, y = 10, 60
    for idx, (label, prob) in enumerate(top_preds):
        if prob < 0.9:
            continue  # Skip low-confidence predictions
        text = f"{idx + 1}. {label} ({int(prob * 100)}%)"
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)
        y += 30

    # Optionally, highlight only the top-1 prediction
    if top_preds and top_preds[0][1] > 0.9:
        label, prob = top_preds[0]
        cv2.putText(img, f"{label} ({int(prob * 100)}%)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def process_frame(frame):
    # Brighten/normalize the frame for improved contrast/visibility.
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    orig = frame.copy()
    
    # Prepare frame for YOLO
    img_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    if half_precision:
        img_tensor = img_tensor.half()

    with torch.no_grad():
        pred = yolo(img_tensor)
        det = non_max_suppression(pred, 0.25, 0.45, agnostic=False)[0]

    top_preds = []  # List to hold classifier predictions

    if det is not None and len(det):
        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], orig.shape).round()
        for *xyxy, conf, cls in reversed(det):
            # Apply YOLO confidence threshold (changed to 0.8)
            if conf < 0.8:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            roi = orig[y1:y2, x1:x2]
            if roi.shape[0] < 20 or roi.shape[1] < 20:
                continue

            try:
                crop_resized = cv2.resize(roi, (32, 32))
                # Save the cropped ROI image to the output folder
                save_filename = f"roi_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                save_path = os.path.join(output_dir, save_filename)
                cv2.imwrite(save_path, crop_resized)
                
                gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

                # Only consider the ROI if it has an expected traffic sign shape
                if any(detect_shape(c) in ["circle", "triangle", "octagon", "rhombus"] for c in contours):
                    # Preprocess ROI exactly as in training
                    pil_img = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
                    tensor = transform(pil_img).unsqueeze(0).to(device)
                    if half_precision:
                        tensor = tensor.half()

                    with torch.no_grad():
                        out = mamba(tensor)
                        prob = torch.nn.functional.softmax(out, dim=1)
                        conf_val, label_idx = torch.max(prob, dim=1)
                        # Only use predictions with high confidence (> 0.8)
                        if conf_val.item() > 0.8:
                            label = class_labels.get(label_idx.item(), "Unknown")
                            top_preds.append((label, conf_val.item()))
                            log_detection(label)
                            update_detections(label)
                            # Optionally, draw the bounding box on the frame
                            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 3)

            except Exception as e:
                print(f"[!] Mamba Classification Error: {e}")

    # If predictions exist, sort them by confidence (highest first)
    if top_preds:
        top_preds = sorted(top_preds, key=lambda x: x[1], reverse=True)
        # Highlight only the top-1 prediction.
        draw_top_predictions(orig, top_preds)
        
    return orig

def draw_top_detections(frame):
    """
    (Optional) If you want to continue using the history of detections,
    this function draws the top detections counted over the last 30 seconds.
    """
    counter = Counter([label for _, label in detections_history])
    x, y = 10, 100
    for idx, (label, count) in enumerate(counter.most_common(5)):
        text = f"{label}: {count}"
        cv2.putText(frame, text, (x, y + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
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

            processed = process_frame(frame)
            draw_top_detections(processed)  # Optional: shows detections history

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time))
            prev_time = curr_time

            cv2.putText(processed, f"FPS: {fps:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("YOLO + Mamba Real-Time Traffic Sign Detection", processed)

            # Exit on ESC key or if the window is closed
            if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("YOLO + Mamba Real-Time Traffic Sign Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()

        # Auto-restart app if window was closed
        window_titles = [w.title for w in gw.getWindowsWithTitle("YOLO + Mamba Real-Time Traffic Sign Detection")]
        if not window_titles:
            print("🔄 Restarting application...")
            python = sys.executable
            os.execv(python, [python] + sys.argv)

if __name__ == "__main__":
    main()
