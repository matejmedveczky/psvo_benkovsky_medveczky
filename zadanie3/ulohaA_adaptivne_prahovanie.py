import cv2
import numpy as np

percent = .30

img = cv2.imread("chess_calibration_1.png")
img = cv2.resize(img, (0, 0), None, percent, percent)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

window_size = 7
C = 5
half = window_size // 2

binary = np.zeros_like(gray)

rows, cols = gray.shape

# adaptivne prahovanie - lokalny priemer
for y in range(half, rows - half):
    for x in range(half, cols - half):
        # vyrez okolia
        window = gray[y - half:y + half + 1, x - half:x + half + 1]

        # lokalny priemer
        local_mean = np.mean(window)

        # lokalny prah
        T = local_mean - C

        # binarizacia
        if gray[y, x] > T:
            binary[y, x] = 255
        else:
            binary[y, x] = 0

# OpenCV adaptivne prahovanie na porovnanie
opencv_adaptive = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    window_size,
    C
)

gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

combined1 = np.hstack((img, gray_bgr))
combined2 = np.hstack((binary, opencv_adaptive))

cv2.imshow("Original and gray", combined1)
cv2.imshow("Adaptive Threshold manual and OpenCv", combined2)

"""
cv2.imshow("Original", img)
cv2.imshow("Gray", gray)
cv2.imshow("Adaptive threshold manual", binary)
cv2.imshow("Adaptive threshold OpenCV", opencv_adaptive)
"""

difference = cv2.absdiff(binary, opencv_adaptive)
cv2.imshow("Difference", difference)

cv2.waitKey()
cv2.destroyAllWindows()