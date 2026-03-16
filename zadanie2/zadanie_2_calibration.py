
import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7*5,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('calibration_images/*.png')

for fname in images:
    img = cv.imread(fname)
    #img = cv.resize(img, (0, 0), None, .30, .30)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7,5), None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, (7,5), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100)

cv.destroyAllWindows()
#cv.waitKey()

#-----------Calibration-------------#
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n--- Vnutorne parametre kamery (K / mtx) ---")
print(mtx)

fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

print(f"\nfx = {fx}")
print(f"fy = {fy}")
print(f"cx = {cx}")
print(f"cy = {cy}")

print("\n--- Distorzne koeficienty (dist) ---")
print(dist)

np.savez("camera_calibration.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("\nUlozene do: camera_calibration.npz")

#-----------Undistortion-------------#
img = cv.imread('calibration_images/chessboard_image_6.png')
#img = cv.resize(img, (0, 0), None, .30, .30)
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

cv.waitKey(2000)
image1 = cv.imread("calibresult.png")
cv.imshow('nove', image1)
cv.waitKey()