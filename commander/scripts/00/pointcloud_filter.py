#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import struct

class PointCloudFilter:
    def __init__(self):
        rospy.init_node("pointcloud_filter", anonymous=True)

        self.bridge = CvBridge()
        self.rgb_mask_sub = rospy.Subscriber("/mask_rgb", Image, self.mask_callback)  # ✅ Fixed: Removed duplicate subscription
        self.pc_sub = rospy.Subscriber("/points2", PointCloud2, self.pc_callback)

        self.cup_pc_pub = rospy.Publisher("/filtered_cup_points", PointCloud2, queue_size=1)  # ✅ Fixed publisher

        self.mask = None

        # ✅ **Adjust these intrinsics based on your depth camera**
        self.camera_intrinsics = {
            "fx": 535.4,  # Adjusted Focal length x
            "fy": 539.2,  # Adjusted Focal length y
            "cx": 635.5,  # Adjusted Principal point x
            "cy": 359.5   # Adjusted Principal point y
        }

        rospy.loginfo("PointCloud Filter Node Initialized.")

    def mask_callback(self, msg):
        """ Receives the mask from rgb_tracker.py """
        try:
            mask_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert to BGR
            self.mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for masking
            
            self.mask = cv2.bilateralFilter(self.mask, 9, 75, 75)  # Smooths noise but preserves edges

            
            self.mask = cv2.resize(self.mask, (1280, 720))  # ✅ Ensure correct resolution
            rospy.loginfo("Received RGB Mask from Tracker and Converted to Grayscale.")
        except Exception as e:
            rospy.logerr(f"Error processing RGB Mask: {e}")

    def pc_callback(self, msg):
        """ Processes PointCloud2 and filters it based on the received mask """
        if self.mask is None:
            rospy.logwarn("Mask not received yet. Skipping point cloud processing.")
            return

        points = []
        mask_height, mask_width = self.mask.shape

        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb_float = point

            # ✅ Convert float32 RGB to an integer
            rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            # ✅ **Project 3D point (x, y, z) onto 2D image plane**
            # ✅ **Project 3D point (x, y, z) onto 2D image plane**
            if z > 0:  # Avoid division by zero
                u = int((x * self.camera_intrinsics["fx"] / z) + self.camera_intrinsics["cx"])
                v = int((y * self.camera_intrinsics["fy"] / z) + self.camera_intrinsics["cy"])

                # ✅ Ensure the projected point is within bounds
                if 0 <= u < mask_width and 0 <= v < mask_height:
                    if self.mask[v, u] > 0:  # ✅ Keep only points inside the detected objects
                        points.append([x, y, z, rgb_float])


        if not points:
            rospy.logwarn("No filtered points found, skipping publish.")
            return

        # ✅ **Create new PointCloud2 message**
        header = msg.header  # ✅ Use the original message header for correct timestamps
        filtered_pc = pc2.create_cloud(header, msg.fields, points)  # ✅ Use correct header

        # ✅ **Publish the filtered point cloud**
        self.cup_pc_pub.publish(filtered_pc)  # ✅ Fixed: Correct publisher used
        rospy.loginfo("Filtered PointCloud Published.")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    filter = PointCloudFilter()
    filter.run()
