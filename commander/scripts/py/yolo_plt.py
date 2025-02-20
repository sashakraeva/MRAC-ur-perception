import os
import sys
import numpy as np
import cv2
import torch
from pyk4a import PyK4A
from pyk4a import Config, ColorResolution, DepthMode

# Define YOLOv5 path explicitly
YOLOV5_PATH = "/dev_ws/src/yolov5"
sys.path.append(YOLOV5_PATH)

# Load YOLOv5 model from local directory
model = torch.hub.load(YOLOV5_PATH, "custom", path=os.path.join(YOLOV5_PATH, "yolov5s.pt"), source="local", device="cpu")

print("✅ YOLO Model Loaded Successfully!")

# Initialize Kinect (Fix applied)
k4a = PyK4A(Config(
    color_resolution=ColorResolution.RES_1080P,
    depth_mode=DepthMode.OFF,  # Disable depth
    synchronized_images_only=False  # Fix: Allow color-only mode
))

k4a.start()

# Open a video window
cv2.namedWindow("Azure Kinect Live Feed", cv2.WINDOW_NORMAL)
print("🎥 Press 'q' to exit.")

try:
    while True:
        # Capture a frame
        capture = k4a.get_capture()

        if capture.color is not None:
            # Convert BGRA to BGR
            frame = cv2.cvtColor(capture.color[:, :, :3], cv2.COLOR_BGRA2BGR)

            # Resize for YOLO input (YOLO expects 640x640)
            frame_yolo = cv2.resize(frame, (640, 640))

            # Convert BGR to RGB (YOLO expects RGB)
            frame_yolo = cv2.cvtColor(frame_yolo, cv2.COLOR_BGR2RGB)

            # YOLOv5 Inference
            results = model(frame_yolo)

            # Get detections
            detections = results.pandas().xyxy[0]

            # Draw bounding boxes
            for _, row in detections.iterrows():
                if row["confidence"] < 0.3:  # Ignore low-confidence detections
                    continue
                
                x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                label = f"{row['name']} {row['confidence']:.2f}"
                
                # Scale bounding boxes back to the original frame size
                h, w, _ = frame.shape
                x1 = int(x1 * (w / 640))
                x2 = int(x2 * (w / 640))
                y1 = int(y1 * (h / 640))
                y2 = int(y2 * (h / 640))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Resize for display and show
            frame_resized = cv2.resize(frame, (1280, 720))
            cv2.imshow("Azure Kinect Live Feed", frame_resized)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Stop Kinect and close windows
    k4a.stop()
    cv2.destroyAllWindows()
