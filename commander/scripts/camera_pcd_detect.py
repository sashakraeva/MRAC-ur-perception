#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import os
import sys
import struct
import open3d as o3d

class CupDetector:

    def __init__(self):
        rospy.init_node("cup_detector", anonymous=True)

        self.bridge = CvBridge()
        self.rgb_image_sub = rospy.Subscriber("/rgb/image_raw", Image, self.image_callback)
        self.depth_image_sub = rospy.Subscriber("/depth_to_rgb/image_raw", Image, self.depth_callback)
        self.pc_sub = rospy.Subscriber("/points2", PointCloud2, self.pc_callback)
        self.cup_pc_pub = rospy.Publisher("/filtered_cup_points", PointCloud2, queue_size=1)

        self.yolo_path = "/dev_ws/src/yolov5"
        sys.path.append(self.yolo_path)

        self.rgb = None
        self.depth = None
        self.pc2 = None

        self.rate = rospy.Rate(16)  # 16Hz update rate

        # Load YOLOv5 model
        try:
            self.model = torch.hub.load(
                self.yolo_path, "custom", 
                path=os.path.join(self.yolo_path, "yolov5s.pt"), 
                source="local", device="cpu"
            )
            rospy.loginfo("YOLO model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}")
            sys.exit(1)

    def detect_cup(self):
        while not rospy.is_shutdown(): 
            if self.rgb is None or self.pc2 is None:
                continue  # Wait until we receive both RGB image and point cloud data
            
            rospy.loginfo("Processing image and point cloud...")

            rgb_image = self.rgb.copy()
            mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)  # Mask for cup regions

            # YOLO detection 
            results = self.model(rgb_image)
            detections = results.pandas().xyxy[0]

            # Filter out detected objects that are cups or vases
            cup_detections = detections[detections["name"].isin(["cup", "vase"])]

            for _, row in cup_detections.iterrows():
                if row["confidence"] < 0.3:
                    continue
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

                # Fill mask with white (255) inside the bounding box
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

            # Apply mask to RGB image (for visualization)
            masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

            # Resize the masked image for display
            scale_percent = 50  # Reduce image size
            width = int(masked_image.shape[1] * scale_percent / 100)
            height = int(masked_image.shape[0] * scale_percent / 100)
            small_masked_image = cv2.resize(masked_image, (width, height), interpolation=cv2.INTER_AREA)

            # Show the masked image
            cv2.imshow("Masked Cups", small_masked_image)
            cv2.waitKey(1)

            # Now apply the mask to the point cloud
            self.filter_and_publish_pc(mask)

            self.rate.sleep()  # Maintain loop rate



    def filter_and_publish_pc(self, mask):
        """Filters self.pc2 using the mask and publishes the new PointCloud2 message."""
        if self.pc2 is None or self.rgb is None or self.depth is None:
            return

        rospy.loginfo("Filtering point cloud based on cup mask...")

        filtered_points = []

        # Camera intrinsic parameters (modify according to your camera)
        fx, fy = 600, 600  # Approximate focal lengths in pixels
        cx, cy = self.rgb.shape[1] // 2, self.rgb.shape[0] // 2  # Image center

        for i, point in enumerate(pc2.read_points(self.pc2, field_names=("x", "y", "z", "rgb"), skip_nans=True)):
            x, y, z, rgb_float = point

            # Convert float RGB to int
            rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
            r = int((rgb_int >> 16) & 0xFF)
            g = int((rgb_int >> 8) & 0xFF)
            b = int(rgb_int & 0xFF)

            # **Ensure RGB values are integers**
            r, g, b = int(r), int(g), int(b)

            # Project 3D point to 2D image coordinates
            img_x = int((fx * x / z) + cx)
            img_y = int((fy * y / z) + cy)

            # Ensure valid image indices
            if 0 <= img_x < mask.shape[1] and 0 <= img_y < mask.shape[0]:
                if mask[img_y, img_x] == 255:  # Check if point is inside the cup mask
                    filtered_points.append([x, y, z, r, g, b])
       
        if len(filtered_points) == 0:
            rospy.logwarn("No filtered points found, skipping publish.")
            return

        # Convert filtered points to a new PointCloud2 message
        elif len(filtered_points) > 0:
            filtered_pc_msg = self.create_pointcloud2(filtered_points)
            self.cup_pc_pub.publish(filtered_pc_msg)
            rospy.loginfo(f"Published filtered PointCloud2 with {len(filtered_points)} cup points.")

    def create_pointcloud2(self, points):
        """Creates a ROS PointCloud2 message from a list of points."""
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "ka_rgb_camera_link"  # ✅ Ensure this matches your TF tree in RViz

        fields = [
            pc2.PointField("x", 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField("y", 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField("z", 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField("r", 12, pc2.PointField.UINT8, 1),
            pc2.PointField("g", 13, pc2.PointField.UINT8, 1),
            pc2.PointField("b", 14, pc2.PointField.UINT8, 1),
        ]

        return pc2.create_cloud(header, fields, points)

    def shutdown(self):
        cv2.destroyAllWindows()

    def image_callback(self, msg):
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

    def depth_callback(self, msg):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"Error in depth callback: {e}")

    def pc_callback(self, msg):
        try:
            self.pc2 = msg  # Store the original ROS PointCloud2 message
            rospy.loginfo("PointCloud2 message received.")
        except Exception as e:
            rospy.logerr(f"Error in point cloud callback: {e}")


def main():
    try:
        cup_detector = CupDetector()
        cup_detector.detect_cup()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt occurred.")
    finally:
        cup_detector.shutdown()

if __name__ == "__main__":
    main()
