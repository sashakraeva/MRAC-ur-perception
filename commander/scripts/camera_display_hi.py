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

class CupDetector:

    def __init__(self):
        rospy.init_node("cup_detector", anonymous=True)

        self.bridge = CvBridge()
        self.rgb_image_sub = rospy.Subscriber("/rgb/image_raw", Image, self.image_callback)
        self.depth_image_sub = rospy.Subscriber("/depth_to_rgb/image_raw", Image, self.depth_callback)
        self.pc_sub = rospy.Subscriber("/points2", PointCloud2, self.pc_callback)
        self.cup_numner = 1

        self.yolo_path = "/dev_ws/src/yolov5"
        sys.path.append(self.yolo_path)

        self.rgb = None
        self.depth = None
        self.pc2 = None

        self.rate = rospy.Rate(10)  # 10Hz update rate

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
            if self.rgb is not None:
                rospy.loginfo("Receiving images...")
                
                rgb_image = self.rgb.copy()
                

                # Run YOLO detection (if needed in future)
                # results = self.model(rgb_image)
                # results.render()  # Draw bounding boxes

                cv2.imshow("RGB Image", rgb_image)
                cv2.waitKey(1)

            self.rate.sleep()  # Maintain loop rate

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
            points = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
                x, y, z, rgb_float = point

                # Convert float32 RGB to an integer
                rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]

                # Extract R, G, B values
                r = (rgb_int >> 16) & 0xFF
                g = (rgb_int >> 8) & 0xFF
                b = rgb_int & 0xFF

                points.append([x, y, z, r, g, b])

            self.pc2 = np.array(points, dtype=np.float32)
        
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
