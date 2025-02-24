#!/usr/bin/env python3

import rospy

import os
import sys
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import struct

# **YOLOv5 Path**
YOLOV5_PATH = "/dev_ws/src/yolov5"
sys.path.append(YOLOV5_PATH)

class YOLOV5PointCloudFilter:
    def __init__(self):
        rospy.init_node("yolo_pointcloud_filter", anonymous=True)

        self.bridge = CvBridge()

        # **Subscribe to RGB Camera Feed and PointCloud**
        self.rgb_sub = rospy.Subscriber("/rgb/image_raw", Image, self.image_callback)
        self.pc_sub = rospy.Subscriber("/points2", PointCloud2, self.pc_callback)

        # **Publishers for Masked Image and Filtered Point Cloud**
        self.mask_rgb_pub = rospy.Publisher("/mask_rgb", Image, queue_size=1)
        self.filtered_pc_pub = rospy.Publisher("/filtered_cup_points", PointCloud2, queue_size=1)

        # ✅ **Fix 1: Ensure YOLOv5 Loads with Correct Device**
        # ✅ Fix 1: Ensure YOLOv5 Loads with Correct Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # If CUDA is unavailable, force CPU mode
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()  # Clear GPU memory if available
            except:
                self.device = "cpu"  # Fallback to CPU

        rospy.loginfo(f"🖥 Using Device: {self.device}")

        # ✅ Fix 2: Load YOLOv5 Model Properly
        try:
            self.model = torch.hub.load(YOLOV5_PATH, "custom",
                                        path=os.path.join(YOLOV5_PATH, "best.pt"),
                                        source="local", device=self.device, force_reload=True)

            self.model.to(self.device)
            self.model.eval()
            rospy.loginfo("✅ YOLOv5 Model Loaded Successfully!")

        except Exception as e:
            rospy.logerr(f"❌ Error loading YOLOv5 model: {e}")
            self.model = None  # Prevent crashing if YOLO fails


        self.model = torch.hub.load(YOLOV5_PATH, "custom",
                            path=os.path.join(YOLOV5_PATH, "best.pt"),
                            source="local", device=self.device, force_reload=True)

        self.model.to(self.device)
        self.model.eval()

        # ✅ **Fix 2: Initialize the Mask to Prevent Errors**
        self.mask = None  # Store latest mask for point cloud processing

        # ✅ **Camera Intrinsics (Adjust for Your Camera)**
        self.camera_intrinsics = {
            "fx": 525.0,  # Focal length x
            "fy": 525.0,  # Focal length y
            "cx": 640.0,  # Principal point x
            "cy": 360.0   # Principal point y
        }

        rospy.loginfo("✅ YOLOv5 RGB Tracker and Point Cloud Filter Initialized.")

    def image_callback(self, msg):
        """ Detects objects using YOLOv5 and creates a mask for point cloud filtering. """
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"❌ Error converting RGB image: {e}")
            return

        # ✅ **YOLOv5 Inference**
        results = self.model(rgb_image)
        detections = results.pandas().xyxy[0]

        # ✅ **Create Empty Mask**
        mask = np.zeros_like(rgb_image, dtype=np.uint8)

        # ✅ **Process YOLO Detections**
        for _, row in detections.iterrows():
            if row["confidence"] < 0.3:  # Ignore low-confidence detections
                continue
            
            label = row["name"]

            # Ensure "vase" is classified correctly
            if label == "vase":
                label = "cup"

            # Get Bounding Box
            x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])

            # ✅ **Draw Mask for Detected Cups**
            if label == "cup":
                cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)  # White region for cups

            # ✅ **Draw Bounding Boxes on Image for Debugging**
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_image, f"{label} {row['confidence']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ✅ **Fix 2: Store Mask for Point Cloud Processing**
        self.mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # ✅ **Show Masked Image**
        masked_image = cv2.bitwise_and(rgb_image, mask)
        cv2.imshow("YOLO Masked Objects", masked_image)
        cv2.waitKey(1)

        # ✅ **Publish Masked Image**
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, "bgr8")
            self.mask_rgb_pub.publish(mask_msg)
        except Exception as e:
            rospy.logerr(f"❌ Error publishing masked image: {e}")

    def pc_callback(self, msg):
        """ Filters the point cloud based on the YOLO mask. """
        # ✅ **Fix 3: Avoid Processing if Mask is Not Available**
        if self.mask is None or self.mask.shape == ():
            rospy.logwarn("⚠ Skipping point cloud processing: Mask not available yet.")
            return

        points = []
        mask_height, mask_width = self.mask.shape

        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb_float = point

            # Convert float32 RGB to integer
            rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            # ✅ **Project 3D Point onto 2D Image Plane**
            if z > 0:  # Avoid division by zero
                u = int((x * self.camera_intrinsics["fx"] / z) + self.camera_intrinsics["cx"])
                v = int((y * self.camera_intrinsics["fy"] / z) + self.camera_intrinsics["cy"])

                if 0 <= u < mask_width and 0 <= v < mask_height:
                    if self.mask[v, u] > 0:  # ✅ Keep only points inside detected cups
                        points.append([x, y, z, rgb_float])

        if not points:
            rospy.logwarn("⚠ No filtered points found, skipping publish.")
            return

        # ✅ **Create new PointCloud2 message**
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_link"

        filtered_pc = pc2.create_cloud(msg.header, msg.fields, points)

        # ✅ **Publish the Filtered Point Cloud**
        self.filtered_pc_pub.publish(filtered_pc)
        rospy.loginfo("✅ Filtered PointCloud Published.")

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = YOLOV5PointCloudFilter()
    tracker.run()
