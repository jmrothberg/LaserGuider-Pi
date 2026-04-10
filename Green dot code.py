import cv2
import numpy as np
# import RPi.GPIO as GPIO
# import time

# Set up GPIO pins
# GPIO.setmode(GPIO.BOARD)
# Set up the X and Y axis servos
# x_axis_pin = 12
# y_axis_pin = 13
# GPIO.setup(x_axis_pin, GPIO.OUT)
# GPIO.setup(y_axis_pin, GPIO.OUT)
# x_axis_servo = GPIO.PWM(x_axis_pin, 50)  # PWM frequency: 50 Hz
# y_axis_servo = GPIO.PWM(y_axis_pin, 50)
# x_axis_servo.start(0)
# y_axis_servo.start(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(1)

object_name = 'face'


def segment_object(frame, object_name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if object_name == 'face':
        objects = face_cascade.detectMultiScale(gray, 1.3, 5)
    elif object_name == 'cat':
        pass
    else:
        objects = []

    segmented = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in objects:
        cv2.rectangle(segmented, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return segmented


def detect_green_dot(frame):
    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours: print("got you")

    green_dot_center = None
    max_area = 0
    min_area = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area > min_area:
            max_area = area
            M = cv2.moments(contour)
            if M["m00"] != 0:
                green_dot_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    return green_dot_center


def angle_to_duty_cycle(angle):
    return 7.5 + (angle / 180) * 5
def update_scan_position(scan_position):
    if scan_position['direction'] == 'right':
        scan_position['x'] += 5
        if scan_position['x'] > 45:
            scan_position['x'] = 45
            scan_position['direction'] = 'left'
            scan_position['y'] += 5
    else:
        scan_position['x'] -= 5
        if scan_position['x'] < -45:
            scan_position['x'] = -45
            scan_position['direction'] = 'right'
            scan_position['y'] += 5
    if scan_position['y'] > 45:
        scan_position['y'] = -45

def move_servo_to_center_object(object_center, green_dot_center):
    if object_center and green_dot_center:
        x_diff = object_center[0] - green_dot_center[0]
        y_diff = object_center[1] - green_dot_center[1]

        # Calculate the Euclidean distance between the green dot and the object center
        distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

        # Calculate scaling factor based on the distance (adjust the constants as needed)
        scaling_factor = max(min(distance / 150, 1), 0.1)

        # Adjust the scaling factor as needed to control the step size
        #x_degrees = max(min(x_diff / 20, 45), -45)
        #y_degrees = max(min(y_diff / 20, 45), -45)
        x_degrees = max(min(x_diff * scaling_factor, 45), -45)
        y_degrees = max(min(y_diff * scaling_factor, 45), -45)

        x_duty_cycle = angle_to_duty_cycle(x_degrees)
        y_duty_cycle = angle_to_duty_cycle(y_degrees)

        print(f"Moving {x_degrees} degrees in x direction.")
        print(f"Moving {y_degrees} degrees in y direction.")

        # Uncomment and adjust the following lines based on your servo setup
        # x_axis_servo.ChangeDutyCycle(x_duty_cycle)
        # y_axis_servo.ChangeDutyCycle(y_duty_cycle)

def is_laser_locked_on_target(object_center, green_dot_center, threshold=10):
    if object_center and green_dot_center:
        distance = np.linalg.norm(np.array(object_center) - np.array(green_dot_center))
        return distance <= threshold
    return False


scan_position = {'x': 0, 'y': 0, 'x_dir': 1, 'y_dir': 1}
print ("Centering system")
scan_position = {'x': -45, 'y': -45, 'direction': 'right'}
print ("Preparing for raster scan")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture frame.")
        continue

    frame = cv2.resize(frame, (1000, 500))
    left_half = frame.copy()
    right_half = frame.copy()

    segmented = segment_object(left_half, object_name)
    right_half = np.copy(segmented)

    objects = []
    if object_name == 'face':
        gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
        objects = face_cascade.detectMultiScale(gray, 1.3, 5)

    target_acquired = False
    object_center = None
    if len(objects) > 0:
        x, y, w, h = objects[0]
        object_center_x = x + w // 2
        object_center_y = y + h // 2
        object_center = (object_center_x, object_center_y)

        cv2.line(right_half, (object_center_x - 10, object_center_y - 10), (object_center_x + 10, object_center_y + 10),
                 (0, 255, 0), 2)
        cv2.line(right_half, (object_center_x + 10, object_center_y - 10), (object_center_x - 10, object_center_y + 10),
                 (0, 255, 0), 2)
        target_acquired = True

    combined_frame = np.hstack((left_half, right_half))

    if target_acquired:
        cv2.putText(combined_frame, "Target Acquired", (1050, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Detect the green dot and draw a green rectangle around it

    green_dot_center = detect_green_dot(left_half)
    #print (green_dot_center)
    #green_dot_detected = green_dot_center is not None
    if green_dot_center:
        x, y = green_dot_center
        cv2.rectangle(right_half, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 2)

    if green_dot_center:
        cv2.circle(frame, green_dot_center, 5, (0, 255, 0), -1)
        cv2.putText(combined_frame, "Laser Spot Detected", (1050, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print ("Laser Spot Detected")

        if object_center:
            move_servo_to_center_object(object_center, green_dot_center)
            if is_laser_locked_on_target(object_center, green_dot_center):
                cv2.putText(combined_frame, "Laser locked on target", (1050, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print ("Laser locked on target")
    else:
        update_scan_position(scan_position)
        print(
            f"Moving to scan position: {scan_position['x']} degrees in x direction, {scan_position['y']} degrees in y direction.")
        x_duty_cycle = angle_to_duty_cycle(scan_position['x'])
        y_duty_cycle = angle_to_duty_cycle(scan_position['y'])

        # Uncomment and adjust the following lines based on your servo setup
        # x_axis_servo.ChangeDutyCycle(x_duty_cycle)
        # y_axis_servo.ChangeDutyCycle(y_duty_cycle)


    cv2.imshow('frame', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        object_name = 'face'
    elif key == ord('c'):
        object_name = 'cat'
    elif key == ord('t'):
        object_name = 'target'

    if key == ord('x'):
        break

# Clean up
# pan_servo.stop()
# tilt_servo.stop()
# GPIO.cleanup()
camera.release()
cv2.destroyAllWindows()

'''def detect_green_dot(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_dot_center = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            M = cv2.moments(contour)
            if M["m00"] != 0:
                green_dot_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    return green_dot_center


def detect_green_dot(frame, min_brightness=240, min_area=20):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a green color mask
    green_mask = cv2.inRange(hsv_frame, (36, 0, 0), (86, 255, 255))

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a brightness threshold to keep only the bright pixels
    _, bright_pixels = cv2.threshold(gray_frame, min_brightness, 255, cv2.THRESH_BINARY)

    # Combine the green mask with the brightness threshold to detect the green dot
    green_dot = cv2.bitwise_and(bright_pixels, bright_pixels, mask=green_mask)

    # Find the contours in the green_dot image
    contours, _ = cv2.findContours(green_dot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= min_area:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    green_dot_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    # Draw a green rectangle around the detected laser spot
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    return green_dot_center'''
