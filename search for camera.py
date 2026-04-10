import cv2

def find_camera_index():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        cap.release()
        index += 1
    return index - 1

camera_index = find_camera_index()
cap = cv2.VideoCapture(camera_index)

while True:
    ret, frame = cap.read()
    cv2.imshow('Camera Test', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
