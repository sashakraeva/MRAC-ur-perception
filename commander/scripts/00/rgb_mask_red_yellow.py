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
        self.mask_pub = rospy.Publisher("/trigger_pointcloud", Bool, queue_size=1)

        self.tracked_objects = {}  # {ID: (centroid, bbox)}
        self.disappeared = {}  # {ID: frames_missed}
        self.object_ids = ["ID1", "ID2", "ID3"]  # Fixed IDs
        self.max_disappeared = 5  # Frames before removing an object

        rospy.loginfo("RGB Tracker Node Initialized.")

    def image_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting RGB image: {e}")
            return

        # Convert to HSV color space
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Define Red Color Range
        lower_red = np.array([157, 0, 227])  
        upper_red = np.array([179, 255, 255])  
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        # Define Yellow Color Range
        lower_yellow = np.array([14, 7, 233])  
        upper_yellow = np.array([50, 255, 255])  
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Process masks separately before combining
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Smaller kernel for small objects

        # Find contours separately
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []
        detected_contours = {}
        ball_detected = None  # Track yellow ball

        # Process Red Objects
        for contour in contours_red:
            if cv2.contourArea(contour) < 500:  # Ignore small objects
                continue
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                detected_objects.append(((cX, cY), (x, y, x + w, y + h)))
                detected_contours[(cX, cY)] = contour

        # Process Yellow Ball
        for contour in contours_yellow:
            if cv2.contourArea(contour) < 100:  # Lower threshold for small ball
                continue
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                ball_detected = ((cX, cY), (x, y, x + w, y + h))  # Assign to ball

        # Update tracked objects
        self.update_tracked_objects(detected_objects, detected_contours, ball_detected, contours_yellow, rgb_image)

        # Show results
        cv2.imshow("Detected Objects", rgb_image)
        cv2.waitKey(1)

    def update_tracked_objects(self, detected_objects, detected_contours, ball_detected, contours_yellow, image):
        """
        Match detected objects to existing IDs using IoU and distance.
        """
        if len(detected_objects) == 0 and ball_detected is None:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.tracked_objects[obj_id]
                    del self.disappeared[obj_id]
            return

        if len(self.tracked_objects) == 0:
            # Register red objects first
            for i, obj in enumerate(detected_objects[:3]):  
                self.tracked_objects[self.object_ids[i]] = obj
                self.disappeared[self.object_ids[i]] = 0
            
            # Register yellow ball if found
            if ball_detected:
                self.tracked_objects["Ball"] = ball_detected
                self.disappeared["Ball"] = 0
                if contours_yellow:  # Ensure contours exist
                    detected_contours[ball_detected[0]] = contours_yellow[0]  # Store ball contour
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

                    if iou > best_iou and distance < 100:  # Max allowed shift
                        best_match = i
                        best_iou = iou

                if best_match is not None:
                    self.tracked_objects[tracked_id] = detected_objects[best_match]
                    self.disappeared[tracked_id] = 0
                    matched.add(best_match)

            # Assign the yellow ball if detected
            if ball_detected:
                self.tracked_objects["Ball"] = ball_detected
                self.disappeared["Ball"] = 0
                if contours_yellow:  # Ensure contours exist
                    detected_contours[ball_detected[0]] = contours_yellow[0]  # Store ball contour

        # Draw contours and IDs on the image
        for obj_id, (centroid, bbox) in self.tracked_objects.items():
            if obj_id == "Ball":
                color = (0, 0, 255)  # Red for the ball
            else:
                color = (0, 255, 0)  # Green for red objects

            if centroid in detected_contours:
                cv2.drawContours(image, [detected_contours[centroid]], -1, color, 2)
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
