#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool

class RGBTracker:
    def __init__(self):
        rospy.init_node("rgb_tracker", anonymous=True)

        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber("/rgb/image_raw", Image, self.image_callback)
        self.mask_pub = rospy.Publisher("/rgb/mask", Image, queue_size=1)  # ✅ Publish mask
        self.tracked_objects = {}  # {ID: (centroid, bbox)}
        self.disappeared = {}  # {ID: frames_missed}
        self.object_ids = ["ID1", "ID2", "ID3"]  # Fixed IDs
        self.max_disappeared = 5  # Frames before removing an object
        self.iou_threshold = 0.3  # IoU threshold for tracking

        rospy.loginfo("RGB Tracker Node Initialized.")

    def image_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting RGB image: {e}")
            return

        # Convert to HSV and apply red color mask
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Find contours of red objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []  # List of (centroid, bbox)
        detected_contours = {}

        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Ignore small objects
                continue

            x, y, w, h = cv2.boundingRect(contour)  # Bounding box
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                detected_objects.append(((cX, cY), (x, y, x + w, y + h)))
                detected_contours[(cX, cY)] = contour

        # Update tracked objects with the new detections
        self.update_tracked_objects(detected_objects, detected_contours, rgb_image)

        # Apply mask to show only detected objects
        masked_image = np.zeros_like(rgb_image)
        for _, contour in detected_contours.items():
            cv2.drawContours(masked_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Convert mask to ROS Image and publish
        gray_mask = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        mask_msg = self.bridge.cv2_to_imgmsg(gray_mask, encoding="mono8")  # Now it's correct

        self.mask_pub.publish(mask_msg)
        rospy.loginfo("Published Mask")

        # Show detected mask
        cv2.imshow("Masked RGB Objects", masked_image)
        cv2.waitKey(1)

    def update_tracked_objects(self, detected_objects, detected_contours, image):
        """
        Match detected objects to existing IDs using IoU and distance.
        """
        if len(detected_objects) == 0:
            # Increase disappeared count for all tracked objects
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.tracked_objects[obj_id]
                    del self.disappeared[obj_id]
            return

        if len(self.tracked_objects) == 0:
            # If no objects are being tracked, register detected ones
            for i, obj in enumerate(detected_objects[:3]):  # Limit to 3 objects
                self.tracked_objects[self.object_ids[i]] = obj
                self.disappeared[self.object_ids[i]] = 0
        else:
            # Compute IoU + Distance matching
            tracked_keys = list(self.tracked_objects.keys())
            tracked_objects = list(self.tracked_objects.values())
            matched = set()

            for tracked_id, (tracked_centroid, tracked_bbox) in zip(tracked_keys, tracked_objects):
                best_match = None
                best_iou = 0

                for i, (new_centroid, new_bbox) in enumerate(detected_objects):
                    if i in matched:
                        continue

                    iou = self.compute_iou(tracked_bbox, new_bbox)
                    distance = np.linalg.norm(np.array(tracked_centroid) - np.array(new_centroid))

                    if iou > best_iou and distance < 100:  # Set max allowed shift
                        best_match = i
                        best_iou = iou

                if best_match is not None:
                    self.tracked_objects[tracked_id] = detected_objects[best_match]
                    self.disappeared[tracked_id] = 0
                    matched.add(best_match)

            # Assign new objects if there are untracked detections
            for i in range(len(self.tracked_objects), min(3, len(detected_objects))):
                if i not in matched:
                    self.tracked_objects[self.object_ids[i]] = detected_objects[i]
                    self.disappeared[self.object_ids[i]] = 0

        # Draw contours and IDs on the image
        for obj_id, (centroid, bbox) in self.tracked_objects.items():
            if centroid in detected_contours:
                cv2.drawContours(image, [detected_contours[centroid]], -1, (0, 255, 0), 2)
            cv2.putText(image, obj_id, (centroid[0], centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def compute_iou(self, boxA, boxB):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = RGBTracker()
    tracker.run()
