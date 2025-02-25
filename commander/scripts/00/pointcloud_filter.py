#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import struct

class PointCloudFilter:
    def __init__(self):
        rospy.init_node("pointcloud_filter", anonymous=True)

        self.bridge = CvBridge()

        # ✅ Subscribe to mask from the Ball Tracker
        self.mask_sub = rospy.Subscriber("/mask_ball_track", Image, self.mask_callback)

        # ✅ Subscribe to PointCloud2 data
        self.pc_sub = rospy.Subscriber("/points2", PointCloud2, self.pc_callback)

        # ✅ Subscribe to CameraInfo topic
        self.camera_info_sub = rospy.Subscriber("/rgb/camera_info", CameraInfo, self.camera_info_callback)

        # ✅ Publisher for the filtered PointCloud2
        self.filtered_pc_pub = rospy.Publisher("/masked_pointcloud", PointCloud2, queue_size=1)

        self.mask = None
        self.mask_resolution = (1280, 720)  # ✅ Ensure correct resolution
        self.camera_intrinsics = None  # Will be updated dynamically

        rospy.loginfo("PointCloud Filter Node Initialized.")

    def camera_info_callback(self, msg):
        """ Extracts camera intrinsics from CameraInfo topic """
        try:
            self.camera_intrinsics = {
                "fx": msg.K[0],  # Focal length x
                "fy": msg.K[4],  # Focal length y
                "cx": msg.K[2],  # Principal point x
                "cy": msg.K[5]   # Principal point y
            }
        except Exception as e:
            rospy.logerr(f"❌ Error extracting camera intrinsics: {e}")

    def mask_callback(self, msg):
        """ Receives and processes the mask from mask_ball_track """
        try:

            # Convert ROS Image to OpenCV format
            mask_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)


        except Exception as e:
            rospy.logerr(f"❌ Error converting mask: {e}")

    def pc_callback(self, msg):
        """ Filters PointCloud2 based on received mask """
        if self.mask is None:
            return

        if self.camera_intrinsics is None:
            return

        points = []
        mask_height, mask_width = self.mask.shape

        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb_float = point

            if z <= 0.1:  # ✅ Ignore invalid depth values
                continue

            # ✅ Convert float32 RGB to an integer
            rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            # ✅ Project 3D point (x, y, z) onto 2D mask plane using dynamic intrinsics
            u = int((x * self.camera_intrinsics["fx"] / z) + self.camera_intrinsics["cx"])
            v = int((-y * self.camera_intrinsics["fy"] / z) + self.camera_intrinsics["cy"])

            # ✅ Ensure projected point is within mask bounds
            if 0 <= u < mask_width and 0 <= v < mask_height:
                if self.mask[v, u] > 0:  # ✅ Keep only points inside the detected mask
                    points.append([x, y, z, rgb_float])

        if not points:
            rospy.logwarn("No masked points found, skipping publish.")
            return

        # ✅ **Create and publish masked PointCloud2**
        header = msg.header  # ✅ Use the original timestamp
        masked_pc = pc2.create_cloud(header, msg.fields, points)

        self.filtered_pc_pub.publish(masked_pc)

        # get the bounding box of the point cloud
        min_x = min(points, key=lambda x: x[0])[0]
        max_x = max(points, key=lambda x: x[0])[0]
        min_y = min(points, key=lambda x: x[1])[1]
        max_y = max(points, key=lambda x: x[1])[1]
        min_z = min(points, key=lambda x: x[2])[2]
        max_z = max(points, key=lambda x: x[2])[2]

        # create a pose message for the center of the bounding box
        x = (min_x + max_x) / 2
        y = (min_y + max_y) / 2
        z = (min_z + max_z) / 2

        print(f"Bounding Box Center: ({x}, {y}, {z})")





    def run(self):
        rospy.spin()

if __name__ == "__main__":
    filter = PointCloudFilter()
    filter.run()
