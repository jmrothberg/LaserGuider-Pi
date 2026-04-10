import cv2
from picamera2 import Picamera2
import numpy as np
import time
picam2 = Picamera2()
#picam2.preview_configuration.main.size = (1280,720)
picam2.preview_configuration.main.format = "RGB888"
#picam2.preview_configuration.align()
#picam2.configure("preview")
picam2.start()

# Initialize the camera object

picam2.resolution = (640, 480)
picam2.framerate = 32

# Allow the camera to warm up
time.sleep(0.1)

image= picam2.capture_array()

# Display the captured image
cv2.imshow("Image", image)

left_half = image.copy()

# Convert the captured image to grayscale
right_half = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
right_half = cv2.cvtColor(right_half, cv2.COLOR_GRAY2BGR) # can ony combine same size so made grey color
combined_frame = np.hstack((left_half, right_half))
cv2.imshow('frame', combined_frame)

# Display the grayscale image
cv2.imshow("Grayscale", right_half)

# Wait for a key press and clean up
cv2.waitKey(0)
cv2.destroyAllWindows()
