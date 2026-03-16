import cv2
import numpy as np
import pandas as pd
from ximea import xiapi

cam = xiapi.Camera()
print('Opening first camera...')
cam.open_device()

cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)
print('Exposure was set to %i us' % cam.get_exposure())

img = xiapi.Image()
print('Starting data acquisition...')
cam.start_acquisition()
print('Press Q to quit')

cv2.namedWindow('Live')

def nothing(x):
    pass

cv2.createTrackbar('min size', 'Live', 10, 1240, nothing)
cv2.createTrackbar('max size', 'Live', 800, 1240, nothing)
cv2.createTrackbar('blur size', 'Live', 5, 9, nothing)
cv2.createTrackbar('pr2', 'Live', 95, 99, nothing)


def is_right_angle_quadrilateral(approx, cosine_threshold=0.25):
    pts = approx.reshape(4, 2).astype(np.float32)
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i - 1) % 4]
        p2 = pts[(i + 1) % 4]

        v1 = p1 - p0
        v2 = p2 - p0
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return False

        cos_angle = float(np.dot(v1, v2) / denom)
        if abs(cos_angle) > cosine_threshold:
            return False
    return True

while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image_display = cv2.resize(image, (1240, 1240))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    min_size = cv2.getTrackbarPos('min size', 'Live')
    max_size = cv2.getTrackbarPos('max size', 'Live')
    blur_size = cv2.getTrackbarPos('blur size', 'Live')
    pr2 = cv2.getTrackbarPos('pr2', 'Live')/100.0

    if blur_size < 1:
        blur_size = 1
    if blur_size % 2 == 0:
        blur_size += 1

    # image_display = cv2.medianBlur(image_display, 25)
    gray = cv2.cvtColor(image_display, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    # _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(gray, 30, 200)

    threshold = cv2.adaptiveThreshold(edged,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1.5, 20,
                               param1=300, param2=pr2, minRadius=min_size, maxRadius=max_size)

    frame_shapes = []

    if circles is not None:
        circles_rounded = np.uint16(np.around(circles))
        for c in circles_rounded[0, :]:
            cx, cy, r = c[0], c[1], c[2]
            cv2.circle(image_display, (cx, cy), r, (0, 255, 0), 3)
            cv2.putText(image_display, 'Circle', (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            frame_shapes.append({'shape': 'Circle', 'x': cx, 'y': cy, 'area': np.pi * r ** 2})

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        sides = len(approx)

        if sides == 3:
            label = 'Triangle'
        elif sides == 4:
            if not cv2.isContourConvex(approx):
                continue
            if not is_right_angle_quadrilateral(approx):
                continue
            label = 'Rectangle'
        elif sides == 5:
            if not cv2.isContourConvex(approx):
                continue
            label = 'Pentagon'
        elif sides == 6:
            if not cv2.isContourConvex(approx):
                continue
            label = 'Hexagon'
        else:
            continue

        cv2.drawContours(image_display, [approx], 0, (0, 0, 255), 5)

        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        cv2.putText(image_display, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame_shapes.append({'shape': label, 'x': x, 'y': y, 'area': cv2.contourArea(contour)})

    frame_df = pd.DataFrame(frame_shapes, columns=['shape', 'x', 'y', 'area'])
    # print(frame_df) 

    cv2.imshow('Live', image_display)

cv2.waitKey(0)
cv2.destroyAllWindows()

cam.stop_acquisition()
cam.close_device()