from ximea import xiapi
import cv2
import numpy as np

def nothing(x):
    pass

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

kernel = np.ones((5, 5), "uint8")

print("Press Q to quit")

# ---------------- WINDOWS ----------------
cv2.namedWindow("Color Detection Ximea")
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 500, 900)

# ---------------- ON/OFF SWITCHES ----------------
cv2.createTrackbar("Tune Red", "Controls", 0, 1, nothing)
cv2.createTrackbar("Tune Green", "Controls", 0, 1, nothing)
cv2.createTrackbar("Tune Blue", "Controls", 0, 1, nothing)

cv2.createTrackbar("Rep G", "Controls", 1, 1, nothing)
cv2.createTrackbar("Rep R", "Controls", 0, 1, nothing)
cv2.createTrackbar("Rep B", "Controls", 0, 1, nothing)

# ---------------- OUTPUT COLOR (BGR) ----------------
# farba, ktorou sa nahradí zelená maska
cv2.createTrackbar("Green-> B", "Controls", 255, 255, nothing)
cv2.createTrackbar("Green-> G", "Controls", 0, 255, nothing)
cv2.createTrackbar("Green-> R", "Controls", 0, 255, nothing)

# farba, ktorou sa nahradí červená maska
cv2.createTrackbar("Red-> B", "Controls", 0, 255, nothing)
cv2.createTrackbar("Red-> G", "Controls", 255, 255, nothing)
cv2.createTrackbar("Red-> R", "Controls", 0, 255, nothing)

# farba, ktorou sa nahradí modrá maska
cv2.createTrackbar("Blue-> B", "Controls", 0, 255, nothing)
cv2.createTrackbar("Blue-> G", "Controls", 255, 255, nothing)
cv2.createTrackbar("Blue-> R", "Controls", 255, 255, nothing)

# ---------------- RED HSV TRACKBARS ----------------
cv2.createTrackbar("R_H_min1", "Controls", 0, 179, nothing)
cv2.createTrackbar("R_H_max1", "Controls", 10, 179, nothing)
cv2.createTrackbar("R_H_min2", "Controls", 170, 179, nothing)
cv2.createTrackbar("R_H_max2", "Controls", 179, 179, nothing)
cv2.createTrackbar("R_S_min", "Controls", 120, 255, nothing)
cv2.createTrackbar("R_S_max", "Controls", 255, 255, nothing)
cv2.createTrackbar("R_V_min", "Controls", 70, 255, nothing)
cv2.createTrackbar("R_V_max", "Controls", 255, 255, nothing)

# ---------------- GREEN HSV TRACKBARS ----------------
cv2.createTrackbar("G_H_min", "Controls", 40, 179, nothing)
cv2.createTrackbar("G_H_max", "Controls", 160, 179, nothing)
cv2.createTrackbar("G_S_min", "Controls", 100, 255, nothing)
cv2.createTrackbar("G_S_max", "Controls", 255, 255, nothing)
cv2.createTrackbar("G_V_min", "Controls", 80, 255, nothing)
cv2.createTrackbar("G_V_max", "Controls", 255, 255, nothing)

# ---------------- BLUE HSV TRACKBARS ----------------
cv2.createTrackbar("B_H_min", "Controls", 94, 179, nothing)
cv2.createTrackbar("B_H_max", "Controls", 120, 179, nothing)
cv2.createTrackbar("B_S_min", "Controls", 80, 255, nothing)
cv2.createTrackbar("B_S_max", "Controls", 255, 255, nothing)
cv2.createTrackbar("B_V_min", "Controls", 2, 255, nothing)
cv2.createTrackbar("B_V_max", "Controls", 255, 255, nothing)

# ---------------- OTHER SETTINGS ----------------
cv2.createTrackbar("Min Area", "Controls", 600, 5000, nothing)

while True:
    cam.get_image(img)
    frame = img.get_image_data_numpy()

    frame_bgr = frame[:, :, :3].copy()
    imageFrame = cv2.undistort(frame_bgr, mtx, dist)

    # ---------------- COLOR DETECTION ----------------
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # ---------------- SWITCH STATES ----------------
    tune_red = cv2.getTrackbarPos("Tune Red", "Controls")
    tune_green = cv2.getTrackbarPos("Tune Green", "Controls")
    tune_blue = cv2.getTrackbarPos("Tune Blue", "Controls")

    replace_green = cv2.getTrackbarPos("Rep G", "Controls")
    replace_red = cv2.getTrackbarPos("Rep R", "Controls")
    replace_blue = cv2.getTrackbarPos("Rep B", "Controls")

    min_area = cv2.getTrackbarPos("Min Area", "Controls")
    if min_area < 1:
        min_area = 1

    # ---------------- OUTPUT COLORS ----------------
    green_out_b = cv2.getTrackbarPos("Green-> B", "Controls")
    green_out_g = cv2.getTrackbarPos("Green-> G", "Controls")
    green_out_r = cv2.getTrackbarPos("Green-> R", "Controls")

    red_out_b = cv2.getTrackbarPos("Red-> B", "Controls")
    red_out_g = cv2.getTrackbarPos("Red-> G", "Controls")
    red_out_r = cv2.getTrackbarPos("Red-> R", "Controls")

    blue_out_b = cv2.getTrackbarPos("Blue-> B", "Controls")
    blue_out_g = cv2.getTrackbarPos("Blue-> G", "Controls")
    blue_out_r = cv2.getTrackbarPos("Blue-> R", "Controls")

    # ---------------- RED MASK ----------------
    if tune_red == 1:
        red_lower1 = np.array([
            cv2.getTrackbarPos("R_H_min1", "Controls"),
            cv2.getTrackbarPos("R_S_min", "Controls"),
            cv2.getTrackbarPos("R_V_min", "Controls")
        ], np.uint8)

        red_upper1 = np.array([
            cv2.getTrackbarPos("R_H_max1", "Controls"),
            cv2.getTrackbarPos("R_S_max", "Controls"),
            cv2.getTrackbarPos("R_V_max", "Controls")
        ], np.uint8)

        red_lower2 = np.array([
            cv2.getTrackbarPos("R_H_min2", "Controls"),
            cv2.getTrackbarPos("R_S_min", "Controls"),
            cv2.getTrackbarPos("R_V_min", "Controls")
        ], np.uint8)

        red_upper2 = np.array([
            cv2.getTrackbarPos("R_H_max2", "Controls"),
            cv2.getTrackbarPos("R_S_max", "Controls"),
            cv2.getTrackbarPos("R_V_max", "Controls")
        ], np.uint8)
    else:
        red_lower1 = np.array([0, 120, 70], np.uint8)
        red_upper1 = np.array([10, 255, 255], np.uint8)
        red_lower2 = np.array([170, 120, 70], np.uint8)
        red_upper2 = np.array([180, 255, 255], np.uint8)

    red_mask1 = cv2.inRange(hsvFrame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsvFrame, red_lower2, red_upper2)
    red_mask = red_mask1 + red_mask2

    # ---------------- GREEN MASK ----------------
    if tune_green == 1:
        green_lower = np.array([
            cv2.getTrackbarPos("G_H_min", "Controls"),
            cv2.getTrackbarPos("G_S_min", "Controls"),
            cv2.getTrackbarPos("G_V_min", "Controls")
        ], np.uint8)

        green_upper = np.array([
            cv2.getTrackbarPos("G_H_max", "Controls"),
            cv2.getTrackbarPos("G_S_max", "Controls"),
            cv2.getTrackbarPos("G_V_max", "Controls")
        ], np.uint8)
    else:
        green_lower = np.array([40, 100, 80], np.uint8)
        green_upper = np.array([160, 255, 255], np.uint8)

    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # ---------------- BLUE MASK ----------------
    if tune_blue == 1:
        blue_lower = np.array([
            cv2.getTrackbarPos("B_H_min", "Controls"),
            cv2.getTrackbarPos("B_S_min", "Controls"),
            cv2.getTrackbarPos("B_V_min", "Controls")
        ], np.uint8)

        blue_upper = np.array([
            cv2.getTrackbarPos("B_H_max", "Controls"),
            cv2.getTrackbarPos("B_S_max", "Controls"),
            cv2.getTrackbarPos("B_V_max", "Controls")
        ], np.uint8)
    else:
        blue_lower = np.array([94, 80, 2], np.uint8)
        blue_upper = np.array([120, 255, 255], np.uint8)

    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # ---------------- MORPHOLOGY ----------------
    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    # ---------------- COLOR REPLACEMENT ----------------
    if replace_green == 1:
        imageFrame[green_mask > 0] = [green_out_b, green_out_g, green_out_r]

    if replace_red == 1:
        imageFrame[red_mask > 0] = [red_out_b, red_out_g, red_out_r]

    if replace_blue == 1:
        imageFrame[blue_mask > 0] = [blue_out_b, blue_out_g, blue_out_r]

    # ---------------- RED CONTOURS ----------------
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imageFrame, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ---------------- GREEN CONTOURS ----------------
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imageFrame, "Green", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ---------------- BLUE CONTOURS ----------------
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, "Blue", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    disp = cv2.resize(imageFrame, (640, 640))
    cv2.imshow("Color Detection Ximea", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

print('Stopping acquisition...')
cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()
print('Done.')