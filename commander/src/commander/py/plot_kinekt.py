import numpy as np
import cv2
from pyk4a import PyK4A
from pyk4a import Config, ColorResolution

# Initialize the Kinect camera
k4a = PyK4A(Config(color_resolution=ColorResolution.RES_1080P))
k4a.start()

# Open a video window
cv2.namedWindow("Azure Kinect Live Feed", cv2.WINDOW_NORMAL)

print("Press 'q' to exit.")

try:
    while True:
        # Capture a frame
        capture = k4a.get_capture()

        if capture.color is not None:
            # Convert from BGRA to RGB
            frame = capture.color[:, :, :3][:, :, ::-1]  # Remove Alpha, Convert BGR to RGB

            # Resize for better display
            frame_resized = cv2.resize(frame, (1280, 720))

            # Display the video
            cv2.imshow("Azure Kinect Live Feed", frame_resized)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Stop Kinect and close windows
    k4a.stop()
    cv2.destroyAllWindows()
