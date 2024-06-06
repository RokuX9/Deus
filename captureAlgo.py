import os
import cv2
import numpy as np
import mss
import pyautogui
from datetime import datetime
import time

# Initialize MSS for screen capture
sct = mss.mss()

# Define the part of the screen to capture
monitor = sct.monitors[1]  # Capture the primary monitor

# Create a folder named with the current date
date_str = datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(date_str):
    os.makedirs(date_str)

# Capture 100 frames
for i in range(100):
    
    # Capture the screen
    screen_img = np.array(sct.grab(monitor))

    # Convert the image from BGRA to BGR
    screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)

    # Get the cursor position
    cursor_x, cursor_y = pyautogui.position()

    # Draw the cursor on the screen image
    cursor_color = (0, 0, 255)  # Red color for the cursor
    cursor_size = 10  # Size of the cursor marker
    cv2.circle(screen_img, (cursor_x, cursor_y), cursor_size, cursor_color, -1)

    # Save the image to the folder
    img_path = os.path.join(date_str, f'screen_capture_{i+1:03d}.jpg')
    cv2.imwrite(img_path, screen_img)

    print(f'Saved {img_path}')
    time.sleep(0.1);

print(f'All 100 frames have been saved in the folder: {date_str}')
