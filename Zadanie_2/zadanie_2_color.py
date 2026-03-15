from ximea import xiapi
import cv2
import numpy as np

data = np.load("camera_calibration.npz")
mtx = data["mtx"]
dist = data["dist"]

cam = xiapi.Camera()
print('Opening first camera...')
cam.open_device()

cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)

print('Starting data acquisition...')
img = xiapi.Image()
cam.start_acquisition()

kernel = np.ones((5,5),"uint8")

print("Press Q to quit")

while True:

    cam.get_image(img)
    frame = img.get_image_data_numpy()

    frame_bgr = frame[:, :, :3].copy()

    imageFrame = cv2.undistort(frame_bgr, mtx, dist)

    # ---------------- COLOR DETECTION ----------------
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # RED - interval 1
    red_lower1 = np.array([0, 120, 70], np.uint8)
    red_upper1 = np.array([10, 255, 255], np.uint8)

    # RED - interval 2
    red_lower2 = np.array([170, 120, 70], np.uint8)
    red_upper2 = np.array([180, 255, 255], np.uint8)

    red_mask1 = cv2.inRange(hsvFrame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsvFrame, red_lower2, red_upper2)
    red_mask = red_mask1 + red_mask2

    # GREEN
    green_lower = np.array([40, 100, 80], np.uint8)
    green_upper = np.array([160, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # BLUE
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    # --------- zmena zelenej na modrú ---------
    imageFrame[green_mask > 0] = [255,0,0]

    contours,_ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(imageFrame,"Red",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))

    contours,_ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imageFrame,"Green",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))

    contours,_ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(imageFrame,"Blue",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))

    disp = cv2.resize(imageFrame,(640,640))
    cv2.imshow("Color Detection Ximea", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


print('Stopping acquisition...')
cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()
print('Done.')