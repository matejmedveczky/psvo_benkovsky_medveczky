import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ximea import xiapi
# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number)
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)

print('Exposure was set to %i us' % cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

print('Press Q to quit')

while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image_display = cv2.resize(image, (1240, 1240))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    gray = cv2.cvtColor(image_display, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    
    all_contours = pd.DataFrame({'contour': contours, 'shape': [None] * len(contours)})
    if circles is not None:
        all_contours = all_contours.append({'contour': circles, 'shape': 'Circle'}, ignore_index=True)

 
    circles = np.uint16(np.around(circles))

    # cv2.imshow('Live', image_display)

    # Process each contour
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 500:
            continue
        if i == 0:
            continue

        # Approximate contour shape
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Draw contour
        cv2.drawContours(image_display, [contour], 0, (0, 0, 255), 5)

        # Find center
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        # Detect shape
        sides = len(approx)
        if sides == 3:
            label = 'Triangle'
        elif sides == 4:
            label = 'Quadrilateral'
        elif sides == 5:
            label = 'Pentagon'
        elif sides == 6:
            label = 'Hexagon'
        else:
            label = 'Circle'

        # Label the shape
        cv2.putText(image_display, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Live', image_display)
        # cv2.imshow('shapes', image_display)

cv2.waitKey(0)
cv2.destroyAllWindows()