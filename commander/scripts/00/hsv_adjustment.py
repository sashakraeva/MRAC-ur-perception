import cv2
import numpy as np

def nothing(x):
    pass

# Open webcam (0 is default camera, change if needed)
cap = cv2.VideoCapture(2)  # Use 1, 2, etc., if you have multiple cameras

# Create a window for HSV adjustments1  
cv2.namedWindow("HSV Adjustments")
cv2.createTrackbar("Lower H", "HSV Adjustments", 0, 179, nothing)
cv2.createTrackbar("Lower S", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower V", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper H", "HSV Adjustments", 179, 179, nothing)
cv2.createTrackbar("Upper S", "HSV Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper V", "HSV Adjustments", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    l_h = cv2.getTrackbarPos("Lower H", "HSV Adjustments")
    l_s = cv2.getTrackbarPos("Lower S", "HSV Adjustments")
    l_v = cv2.getTrackbarPos("Lower V", "HSV Adjustments")
    u_h = cv2.getTrackbarPos("Upper H", "HSV Adjustments")
    u_s = cv2.getTrackbarPos("Upper S", "HSV Adjustments")
    u_v = cv2.getTrackbarPos("Upper V", "HSV Adjustments")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Create mask and apply it
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display results
    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Masked Output", result)

    # Press 'S' to print selected HSV values
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        print(f"Lower HSV: {lower_bound}, Upper HSV: {upper_bound}")
    elif key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
