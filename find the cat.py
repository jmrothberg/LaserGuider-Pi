#Program to use camera to find object, and outline it, then put cross hairs in center.
import cv2
import numpy as np
# import RPi.GPIO as GPIO
# import time

# Set up GPIO pins
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(11, GPIO.OUT)

# Load the classifier for object detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up laser pointer color thresholds
lower_red = np.array([0, 0, 200])
upper_red = np.array([50, 50, 255])

# Initialize the camera
camera = cv2.VideoCapture(1)

# Initialize the servo motors
# pan_pin = 12
# tilt_pin = 13
# GPIO.setup(pan_pin, GPIO.OUT)
# GPIO.setup(tilt_pin, GPIO.OUT)
# pan_servo = GPIO.PWM(pan_pin, 50)  # PWM frequency: 50 Hz
# tilt_servo = GPIO.PWM(tilt_pin, 50)
# pan_servo.start(0)
# tilt_servo.start(0)

# Select the object to track
object_name = 'face'

# Set the scan step and delay
step = 10  # degrees
delay = 0.02  # seconds


def segment_object(frame, object_name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if object_name == 'face':
        objects = face_cascade.detectMultiScale(gray, 1.3, 5)
    elif object_name == 'cat':
        # Add cat classifier here
        pass
    elif object_name == 'target':
        ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500 and area < 10000:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                if len(approx) == 8:
                    objects.append(cv2.boundingRect(cnt))

    segmented = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in objects:
        cv2.rectangle(segmented, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return segmented


while True:
    # Capture a frame from the camera
    ret, frame = camera.read()

    # Resize the frame
    frame = cv2.resize(frame, (1000, 500))

    # Split the frame into two halves
    left_half = frame[:, :500]
    right_half = frame[:, 500:]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)

    # Detect the object in the frame
    if object_name == 'face':
        objects = face_cascade.detectMultiScale(gray, 1.3, 5)
    elif object_name == 'cat':
        # Add cat classifier here
        pass
    elif object_name == 'target':
        # You can add more classifiers for different objects
        # Here we're using a simple circular target as an example
        ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500 and area < 10000:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                if len(approx) == 8:
                    objects.append(cv2.boundingRect(cnt))

    if len(objects) > 0:
        # If the object is detected, get its center coordinates
        x, y, w, h = objects[0]
        object_center_x = x + w // 2
        object_center_y = y + h // 2

        # Draw an 'X' at the center of the object
        cv2.line(left_half, (object_center_x - 10, object_center_y - 10), (object_center_x + 10, object_center_y + 10),
                 (0, 255, 0), 2)
        cv2.line(left_half, (object_center_x + 10, object_center_y - 10), (object_center_x - 10, object_center_y + 10),
                 (0, 255, 0), 2)

        # Update the right half with the segmented object
        right_half = segment_object(left_half, object_name)

        # Draw a crosshair at the center of the object on the right half
        cv2.line(right_half, (object_center_x - 10, object_center_y), (object_center_x + 10, object_center_y),
                 (0, 0, 255), 2)
        cv2.line(right_half, (object_center_x, object_center_y - 10), (object_center_x, object_center_y + 10),
                 (0, 0, 255), 2)

    # Combine the left and right halves
    combined_frame = np.hstack((left_half, right_half))

    # Display the frame
    cv2.imshow('frame', combined_frame)

    # Allow the user to switch objects by pressing 'f' for face, 'c' for cat, or 't' for target
    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        object_name = 'face'
    elif key == ord('c'):
        object_name = 'cat'
    elif key == ord('t'):
        object_name = 'target'

    # Exit if 'x' is pressed
    if key == ord('x'):
        break

# Clean up
# pan_servo.stop()
# tilt_servo.stop()
# GPIO.cleanup()
camera.release()
cv2.destroyAllWindows()
