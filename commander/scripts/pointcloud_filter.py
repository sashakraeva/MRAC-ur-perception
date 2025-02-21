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
        self.rgb_mask_sub = rospy.Subscriber("/rgb/mask", Image, self.mask_callback)  # ✅ Subscribe to mask
        self.pc_sub = rospy.Subscriber("/points2", PointCloud2, self.pc_callback)
        self.filtered_pc_pub = rospy.Publisher("/filtered_points", PointCloud2, queue_size=1)

        self.mask = None
        rospy.loginfo("PointCloud Filter Node Initialized.")

    def mask_callback(self, msg):
        """ Receives the mask from rgb_tracker.py """
        try:
            self.mask = self.bridge.imgmsg_to_cv2(msg, "mono8")  # ✅ Ensure correct format
            rospy.loginfo("Received Mask from RGB Tracker.")
        except Exception as e:
            rospy.logerr(f"Error converting mask: {e}")

    def pc_callback(self, msg):
        """ Receives the raw PointCloud2 and filters it based on the mask. """
        if self.mask is None:
            rospy.logwarn("Mask not received yet. Skipping point cloud processing.")
            return

        points = []
        height, width = self.mask.shape

        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb_float = point

            # Convert float32 RGB to an integer
            rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]

            # Extract R, G, B values
            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            # Project (x, y, z) onto 2D image space
            u = int((x + 1.0) * width / 2.0)
            v = int((1.0 - y) * height / 2.0)

            if 0 <= u < width and 0 <= v < height:
                if self.mask[v, u] > 0:  # If pixel is part of the detected object
                    points.append([x, y, z, rgb_float])

        if not points:
            rospy.logwarn("No filtered points found, skipping publish.")
            return

        # Create new PointCloud2 message
        header = msg.header
        filtered_pc = pc2.create_cloud(header, msg.fields, points)

        # Publish filtered point cloud
        self.filtered_pc_pub.publish(filtered_pc)

        rospy.loginfo("Filtered PointCloud Published.")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    filter = PointCloudFilter()
    filter.run()
