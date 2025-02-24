import cv2
import numpy as np

# Load the depth image (colored)
depth_image_colored = cv2.imread('/dev_ws/src/commander/scripts/RGB Image_screenshot_24.02.2025.png')
# hsv_depth = cv2.cvtColor(depth_image_colored, cv2.COLOR_BGR2HSV)
hsv_depth = depth_image_colored

# List to store clicked HSV values
hsv_values = []

# Mouse click callback function
def get_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv_depth[y, x]  # Get HSV values at clicked point
        hsv_values.append(pixel)
        print(f'Added HSV at ({x}, {y}): {pixel}')

# Display image and set mouse callback
cv2.imshow('Click on multiple color regions, press any key to finish', hsv_depth)
cv2.setMouseCallback('Click on multiple color regions, press any key to finish', get_hsv_value)
cv2.waitKey(0)
cv2.destroyAllWindows()

# If there are selected points, calculate the optimal HSV range
if len(hsv_values) > 0:
    hsv_values = np.array(hsv_values)

    # Compute mean and std deviation for each HSV channel
    mean_hsv = np.mean(hsv_values, axis=0)
    std_hsv = np.std(hsv_values, axis=0)

    # Define HSV range dynamically
    lower_bound = np.clip(mean_hsv - std_hsv * 1.5, 0, 255).astype(np.uint8)
    upper_bound = np.clip(mean_hsv + std_hsv * 1.5, 0, 255).astype(np.uint8)

    print(f'Optimal HSV range:')
    print(f'Lower Bound: {lower_bound}')
    print(f'Upper Bound: {upper_bound}')
else:
    print('No HSV values selected!')
