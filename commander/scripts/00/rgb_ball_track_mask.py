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
        self.mask_rgb_pub = rospy.Publisher("/mask_rgb", Image, queue_size=1)
        self.mask_ball_pub = rospy.Publisher("/mask_ball_track", Image, queue_size=1)


        self.tracked_objects = {}  # {ID: (centroid, bbox)}
        self.disappeared = {}  # {ID: frames_missed}
        self.object_ids = ["ID1", "ID2", "ID3"]  # Fixed IDs for cups
        self.max_disappeared = 5  # Frames before removing an object

        self.ball_hidden = False  # Whether the ball is inside a cup
        self.ball_parent_cup = None  # ID of the cup that holds the ball
        self.last_ball_position = None  # Last known position of the ball

        rospy.loginfo("RGB Tracker Node Initialized.")

    def image_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting RGB image: {e}")
            return

        # Convert to HSV color space
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Define Red Color Range (Cups)
        lower_red = np.array([157, 0, 227])  
        upper_red = np.array([179, 255, 255])  
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        # Define Yellow Color Range (Ball)
        lower_yellow = np.array([15, 0, 241])  
        upper_yellow = np.array([97, 255, 255])  
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Process masks separately before combining
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Find contours separately
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []
        detected_contours = {}
        ball_detected = None  # Track yellow ball

        # Process Red Objects (Cups)
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
            if cv2.contourArea(contour) < 50:  # Lower threshold for small ball
                continue
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                ball_detected = ((cX, cY), (x, y, x + w, y + h))  # Assign to ball

        # Update tracked objects with ball-in-cup tracking
        self.update_tracked_objects(detected_objects, detected_contours, ball_detected, contours_yellow, rgb_image)

        # **New Logic: Mask Everything Outside the Contours**
        mask = np.zeros_like(rgb_image, dtype=np.uint8)  # Black background

        # Draw all detected contours for cups as white on the mask
        for contour in detected_contours.values():
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # ✅ **Mask the Ball if Detected**
        if ball_detected:
            ball_x, ball_y = ball_detected[0]  # Extract ball center
            cv2.circle(mask, (ball_x, ball_y), 10, (255, 255, 255), -1)  # White filled circle for the ball

        # Apply mask to show only detected objects
        masked_image = cv2.bitwise_and(rgb_image, mask)

        # **Draw Circles for Red Cups and the Yellow Ball**
        for obj_id, (centroid, _) in self.tracked_objects.items():
            if obj_id.startswith("ID"):  # Red cups
                cv2.circle(masked_image, centroid, 10, (255, 0, 0), -1)  # Blue circle (Cups)
            elif obj_id == "Ball":  # Yellow ball
                cv2.circle(masked_image, centroid, 5, (0, 255, 0), -1)  # Green dot (Ball)
        
        # find the contour of the cup which contains the ball, draw a contour around it
        if self.ball_parent_cup:
            cup_contour = detected_contours.get(self.tracked_objects[self.ball_parent_cup][0])
            if cup_contour is not None:
                cv2.drawContours(masked_image, [cup_contour], -1, (0, 255, 0), 2)

                mask_cup = np.zeros_like(rgb_image, dtype=np.uint8)
                # convert the contour to a mask
                cv2.drawContours(mask_cup, [cup_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                masked_image_cup = cv2.bitwise_and(rgb_image, mask_cup)
                try:
                    mask_msg = self.bridge.cv2_to_imgmsg(masked_image_cup, "bgr8")
                    self.mask_ball_pub.publish(mask_msg)
                except Exception as e:
                    rospy.logerr(f"Error publishing masked image")


 


        # # Show results
        # cv2.imshow("Masked Objects with Markers", masked_image)  # Now only detected objects are visible
        # cv2.waitKey(1)

        try:
            mask_msg = self.bridge.cv2_to_imgmsg(masked_image, "bgr8")
            self.mask_rgb_pub.publish(mask_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing masked image: {e}")

    def update_tracked_objects(self, detected_objects, detected_contours, ball_detected, contours_yellow, image):
        """
        Match detected objects to existing IDs using IoU and distance.
        Ensure that the ball stays inside the correct cup when covered.
        """
        if len(detected_objects) == 0 and ball_detected is None:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.tracked_objects[obj_id]
                    del self.disappeared[obj_id]
            return

        # Assign IDs to cups properly
        for i, obj in enumerate(detected_objects[:3]):
            self.tracked_objects[self.object_ids[i]] = obj
            self.disappeared[self.object_ids[i]] = 0

        # Track ball
        if ball_detected:
            self.tracked_objects["Ball"] = ball_detected
            self.disappeared["Ball"] = 0
            self.ball_hidden = False
            self.ball_parent_cup = None
            self.last_ball_position = ball_detected[0]
        else:
            if not self.ball_hidden:
                for obj_id, (_, cup_bbox) in self.tracked_objects.items():
                    if obj_id.startswith("ID"):  # Ensure it's a cup
                        cup_x1, cup_y1, cup_x2, cup_y2 = cup_bbox
                        if self.last_ball_position and cup_x1 < self.last_ball_position[0] < cup_x2 and cup_y1 < self.last_ball_position[1] < cup_y2:
                            self.ball_hidden = True
                            self.ball_parent_cup = obj_id
                            break

            if self.ball_hidden and self.ball_parent_cup in self.tracked_objects:
                self.tracked_objects["Ball"] = self.tracked_objects[self.ball_parent_cup]  # Assign ball's position to cup
                self.disappeared["Ball"] = 0

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = RGBTracker()
    tracker.run()