# JMR Self aiming laser pointer Revison to average last 5 locations, and allow for disapearance of laser and object
# May 4th.  White box for white circle, green for green.
# Exploring more complex laser detection and putting in information to help fine tune it.
import cv2
import numpy as np
import RPi.GPIO as GPIO
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import time

# don't forget to start the daemon terminal: sudo pigpiod

factory = PiGPIOFactory()

x_axis_servo = Servo(13,pin_factory=factory)
y_axis_servo = Servo(18,pin_factory=factory)

x_axis_servo.value = 0
y_axis_servo.value = 0
time.sleep(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

camera = cv2.VideoCapture(0)
object_name = 'face'

def detect_green_dot(frame, objects):
    # Padding around the detected object
    padding = 100
    x, y, w, h = objects[0]
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image_width, x + w + padding)
    y_end = min(image_height, y + h + padding)

    # Extract the Region of Interest (ROI) from the frame
    roi = frame[y_start:y_end, x_start:x_end]

    # Apply Gaussian blur on ROI and convert it to HSV
    blurred_roi = cv2.GaussianBlur(roi, (7, 7), 0)
    hsv = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)

    # Define color ranges for white and green
    lower_white = np.array([0, 0, 230])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lower_green = np.array([20, 80, 80])
    upper_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours for white and green regions
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_circularity_white = 0.4
    best_intensity_white = 0
    best_center_white = (0, 0)
    best_rect_white = (0,0,0,0)
    best_circularity_green = 0.1
    best_intensity_green = 0
    best_center_green = (0, 0)
    best_rect_green = (0,0,0,0)
    has_green_halo = False

    # Iterate through green contours and draw them on the ROI
    for green_contour in green_contours:
        green_area = cv2.contourArea(green_contour)
        green_perimeter = cv2.arcLength(green_contour, True)
        if green_area > 400 and green_area < 10000 and green_perimeter > 0:
            green_circularity = 4 * np.pi * green_area / (green_perimeter ** 2)
            M_green = cv2.moments(green_contour)

            if M_green["m00"] != 0:
                green_center = (int(M_green["m10"] / M_green["m00"]) + x_start, int(M_green["m01"] / M_green["m00"]) + y_start)

                # Calculate the bounding box for the green contour
                x_green, y_green, w_green, h_green = cv2.boundingRect(green_contour)
                
                # Initialize the mask and draw the green contour on the mask
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [green_contour], -1, 255, -1)

                # Calculate mean intensity of the green contour
                green_intensity = np.mean(cv2.bitwise_and(roi, roi, mask=mask))
                
                # Draw the green contour on the ROI
                cv2.drawContours(roi, [green_contour], -1, (0, 255, 0), 2)

                # Display information about the green contour on the frame
                y_index = np.clip(green_center[1], 0, hsv.shape[0] - 1)
                x_index = np.clip(green_center[0], 0, hsv.shape[1] - 1)
                info_text = f"Size: {w_green}x{h_green}, HSV: {hsv[y_index, x_index]}, Circularity: {green_circularity:.2f}"
                print("Green", info_text)
                text_x = max(0, min(green_center[0] - w // 2, frame.shape[1] - len(info_text) * 9))
                text_y = max(20, min(green_center[1] - h // 2, frame.shape[0]))
                cv2.putText(frame, info_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if green_circularity > best_circularity_green and green_intensity > best_intensity_green:
                    best_circularity_green = green_circularity
                    best_intensity_green = green_intensity
                    best_center_green = green_center
                    best_rect_green = x_green, y_green, w_green, h_green
                    
    # Iterate through white contours and draw them on the ROI
    for white_contour in white_contours:
        area = cv2.contourArea(white_contour)
        perimeter = cv2.arcLength(white_contour, True)
        if area > 100 and area < 10000 and perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            M_white = cv2.moments(white_contour)

            if M_white["m00"] != 0:
                center = (int(M_white["m10"] / M_white["m00"]) + x_start, int(M_white["m01"] / M_white["m00"]) + y_start)
                x_white, y_white, w_white, h_white = cv2.boundingRect(white_contour)

                # Initialize the mask and draw the white contour on the mask
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [white_contour], -1, 255, -1)

                # Calculate mean intensity of the white contour
                intensity = np.mean(cv2.bitwise_and(roi, roi, mask=mask))

                # Draw the white contour on the ROI
                cv2.drawContours(roi, [white_contour], -1, (255, 255, 255), 2)

                # Display information about the white contour on the frame
                y_index = np.clip(center[1], 0, hsv.shape[0] - 1)
                x_index = np.clip(center[0], 0, hsv.shape[1] - 1)
                info_text = f"Size: {w_white}x{h_white}, HSV: {hsv[y_index, x_index]}, Circularity: {circularity:.2f}"
                print("White", info_text)
                text_x = max(0, min(center[0] - w // 2, frame.shape[1] - len(info_text) * 9))
                text_y = max(20, min(center[1] - h // 2, frame.shape[0]))
                cv2.putText(frame, info_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                green_detected = False
                for green_contour in green_contours:
                    green_area = cv2.contourArea(green_contour)

                    if green_area > area:
                        green_detected = True
                        has_green_halo = True
                        M_green = cv2.moments(green_contour)

                        if M_green["m00"] != 0:
                            green_center = (int(M_green["m10"] / M_green["m00"]) + x_start, int(M_green["m01"] / M_green["m00"]) + y_start)

                            if circularity > best_circularity_white and intensity > best_intensity_white:
                                best_circularity_white = circularity
                                best_intensity_white = intensity
                                best_center_white = center
                                best_rect_white = x_white, y_white, w_white, h_white
                if not green_detected and circularity > best_circularity_white and intensity > best_intensity_white:
                    best_circularity_white = circularity
                    best_intensity_white = intensity
                    best_center_white = center
                    best_rect_white = x_white, y_white, w_white, h_white
                    
    result = "no"
    best_center = (0, 0)

    if best_center_white != (0, 0):
        x, y = best_center_white
        size = 10
        result = "white"
        best_center = best_center_white
        cv2.rectangle(roi, best_rect_white, (255, 255, 255), 2)

    if has_green_halo:
        x, y = best_center_white
        size = 10
        result = "green hallow"
        best_center = best_center_white
        cv2.rectangle(roi, best_rect_white, (0, 255, 0), 4)

    if best_center_green != (0, 0) and not has_green_halo:
        x, y = best_center_green
        size = 10
        result = "green"
        best_center = best_center_green
        cv2.rectangle(roi, best_rect_green, (0, 255, 0), 8)

    return result, best_center


def rect_intersection(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    return x_overlap * y_overlap


 # This is for Raspberry Pi servo libray with -1 to 1 being -90 to 90. We work in angles and convert to pixels
def angle_to_servo(angle):
    return angle/90


 # Just used a map instead of simple variables so you can store the direction, but just really x and y servo angles.
def update_scan_position(servo_angles):

    if servo_angles['direction'] == 'right':
        servo_angles['x'] += scan_speed
        if servo_angles['x'] > 45:
            servo_angles['x'] = 45
            servo_angles['direction'] = 'left'
            servo_angles['y'] += scan_speed
    else:
        servo_angles['x'] -= scan_speed
        if servo_angles['x'] < -45:
            servo_angles['x'] = -45
            servo_angles['direction'] = 'right'
            servo_angles['y'] += scan_speed
    if servo_angles['y'] > 34:
        servo_angles['y'] = -34
    return servo_angles


#This is the key function, just needs to be logical, need small movements to not over shoot when noise. Could smooth using previous movements.
def move_servo_close (object_center, green_dot_center, servo_angles, previous_servo_movements, filter_size):

    DEGREES_PER_PIXEL_X = horizontal_FOV / image_width
    DEGREES_PER_PIXEL_Y = vertical_FOV / image_height

    servo_angles['x']  = (object_center [0]- image_width // 2) * DEGREES_PER_PIXEL_X
    servo_angles['y']  = (object_center [1]- image_height //2) * DEGREES_PER_PIXEL_Y

    #print(f"Moving to {servo_angles['x']} ")
    #print(f"Moving to {servo_angles['y']} ")

    x_axis_servo.value= angle_to_servo(servo_angles['x'])
    y_axis_servo.value = angle_to_servo(servo_angles['y'])

    x_servo_pixel = int(image_width // 2 + (servo_angles['x'] ) / DEGREES_PER_PIXEL_X)
    y_servo_pixel = int(image_height // 2 + (servo_angles['y']) / DEGREES_PER_PIXEL_Y)

    #print(x_servo_pixel, y_servo_pixel)
    cv2.circle(right_half, (x_servo_pixel, y_servo_pixel), 50, (255, 0, 0), 2)

    return  servo_angles, previous_servo_movements



#This is the key function, just needs to be logical, need small movements to not over shoot when noise. Could smooth using previous movements.
def move_servo_to_center_object(object_center, green_dot_center, servo_angles, previous_servo_movements, filter_size):

    DEGREES_PER_PIXEL_X = horizontal_FOV / image_width
    DEGREES_PER_PIXEL_Y = vertical_FOV / image_height

    x_diff = object_center[0] - green_dot_center[0]
    y_diff = object_center[1] - green_dot_center[1]

    x_degrees_servo_movement = max(min(x_diff/DEGREES_PER_PIXEL_X, 1), -1)
    y_degrees_servo_movement = max(min(y_diff/DEGREES_PER_PIXEL_Y, 1), -1)

    new_x_angle = servo_angles['x'] + x_degrees_servo_movement
    new_y_angle = servo_angles['y'] + y_degrees_servo_movement

    servo_angles['x'] = max(min(new_x_angle, 45), -45)
    servo_angles['y'] = max(min(new_y_angle, 22.5), -22.5)

    #print(f"Moving to {servo_angles['x']} degrees x.")
    #print(f"Moving to {servo_angles['y']} degrees y.")

    x_axis_servo.value= angle_to_servo(servo_angles['x'])
    y_axis_servo.value = angle_to_servo(servo_angles['y'])

    x_servo_pixel = int(image_width // 2 + (servo_angles['x'] ) / DEGREES_PER_PIXEL_X)
    y_servo_pixel = int(image_height // 2 + (servo_angles['y']) / DEGREES_PER_PIXEL_Y)

    #print(x_servo_pixel, y_servo_pixel)
    cv2.circle(right_half, (x_servo_pixel, y_servo_pixel), 10, (0, 0, 255), 2)

    return  servo_angles, previous_servo_movements


def is_laser_locked_on_target(object_center, green_dot_center, threshold):
    if object_center and green_dot_center:
        distance = np.linalg.norm(np.array(object_center) - np.array(green_dot_center))
        return distance <= threshold
    return False


#simplest fast face recognizer, need to tune, later update with new Deeplearning model
def segment_object(frame, object_name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if object_name == 'face':
        objects = face_cascade.detectMultiScale(gray, 1.3, 5)
    elif object_name == 'body':
        objects = upper_body_cascade.detectMultiScale(gray, 1.3, 5)
    else:
        objects = []

    segmented = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # make it color so you can put red on it :)

    for (x, y, w, h) in objects:
        cv2.rectangle(segmented, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return segmented, objects

print ("Centering system")


# again, just x and y, but saving direciton in same variable.
servo_angles = {'x': 0, 'y': 0, 'direction': 'right'}

print ("Preparing for raster scan")
objects = []
targets = []
target_acquired_time = 0
green_dot_acquired_time = 0
target_acquired = False
green_dot_acquired = False

previous_servo_movements = []
filter_size = 3
horizontal_FOV = 90
vertical_FOV = 68

image_width = 640
image_height = 480

DEGREES_PER_PIXEL_X = horizontal_FOV / image_width
DEGREES_PER_PIXEL_Y = vertical_FOV / image_height

object_center = (0,0)
green_dot_center = (0,0)
threshold = 10
see_green = "no"
scan_speed = 5
while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture frame.")
        continue

    frame = cv2.resize(frame, (image_width, image_height))
    left_half = frame.copy()
    right_half = frame.copy()

    if time.time() - target_acquired_time >= 0.5:
        target_acquired = False
    if time.time() - green_dot_acquired_time >= 0.5:
        green_dot_acquired = False

    segmented, objects = segment_object(left_half, object_name)

    right_half = np.copy(segmented)

    if len(objects) > 0:  #Seeing targets

        x, y, w, h = objects[0]
        object_center_x = x + w // 2
        object_center_y = y + h // 2
        object_center = (object_center_x, object_center_y)

        cv2.line(right_half, (object_center_x - 10, object_center_y - 10), (object_center_x + 10, object_center_y + 10),
                 (0, 255, 0), 2)
        cv2.line(right_half, (object_center_x + 10, object_center_y - 10), (object_center_x - 10, object_center_y + 10),
                 (0, 255, 0), 2)

        cv2.putText(right_half, "Target Acquired", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        target_acquired = True
        if not see_green:
            servo_angles, previous_servo_movements = move_servo_close (object_center, green_dot_center, servo_angles, previous_servo_movements, filter_size)
        target_acquired_time = time.time()

        # Detect the green dot and draw a green rectangle and square around it
        see_green, maybe_green_center = detect_green_dot(left_half, objects)  # Now only looking for green dot in vicinity of the object.

    if see_green != "no":
        #print ("Laser Spot Detected")
        green_dot_center = maybe_green_center
        x, y = green_dot_center
        #cv2.rectangle(right_half, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 2)
        #cv2.circle(right_half, green_dot_center, 20, (0, 255, 0), 3)
        laser_text = "Laser Spot Detected: " + see_green
        cv2.putText(right_half, laser_text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        green_dot_acquired = True
        green_dot_acquired_time = time.time()

    if green_dot_acquired and target_acquired:
        servo_angles, previous_servo_movements = move_servo_to_center_object(object_center, green_dot_center, servo_angles, previous_servo_movements, filter_size)
        if is_laser_locked_on_target(object_center, green_dot_center, threshold):
            cv2.putText(right_half, "Laser locked on target", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(right_half, green_dot_center, 20, (0, 0, 255), 10)
            print("Laser locked on target")

    if not green_dot_acquired:
            update_scan_position(servo_angles)

            x_axis_servo.value = angle_to_servo(servo_angles['x'])
            y_axis_servo.value = -angle_to_servo(servo_angles['y'])

            # Calculate the pixel coordinates for the servo position
            x_servo_pixel = int(image_width // 2 + (servo_angles['x']) / DEGREES_PER_PIXEL_X)
            y_servo_pixel = int(image_height // 2 + (servo_angles['y']) / DEGREES_PER_PIXEL_Y) # was opposite

            # Draw a blue circle in the right frame to show where the servos are pointing
            cv2.circle(right_half, (x_servo_pixel, y_servo_pixel), 10, (255, 0, 0), 2)

    combined_frame = np.hstack((left_half, right_half))
    cv2.imshow('frame', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        object_name = 'face'
        print ("face")
    elif key == ord('c'):
        object_name = 'cat'
        print ("cat")
    elif key == ord('t'):
        object_name = 'bullseye'
        print ("bullseye")
    elif key == ord('b'):
        object_name = 'body'
        print ("body")

    elif key == ord('1'):
        option = 1
        print ("option 1")
    elif key == ord('t'):
        option = ("option 2")
    elif key == ord('b'):
        option = 3
        print("option 3")

    if key == ord('q'):
        break

# Clean up
x_axis_servo.stop()
y_axis_servo.stop()
GPIO.cleanup()
camera.release()
cv2.destroyAllWindows()