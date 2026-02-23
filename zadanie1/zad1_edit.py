import cv2
from ximea import xiapi
import numpy as np

img1 = cv2.imread('captured_image_1.png')
img2 = cv2.imread('captured_image_2.png')
img3 = cv2.imread('captured_image_3.png')
img4 = cv2.imread('captured_image_4.png')

height, width = 240, 240 

img1 = cv2.resize(img1, (240, 240))
img2 = cv2.resize(img2, (240, 240))
img3 = cv2.resize(img3, (240, 240))
img4 = cv2.resize(img4, (240, 240))

mosaic = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

mosaic[0:height, 0:width] = img1

mosaic[0:height, width:2*width] = img2  

mosaic[height:2*height, 0:width] = img3

mosaic[height:2*height, width:2*width] = img4   

cv2.imshow("Mosaic 2x2", mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()

# vyhladzovaci kernel
kernel = np.ones((3, 3), dtype=np.float32) / 9.0
mosaic[0:height, 0:width] = cv2.filter2D(
    mosaic[0:height, 0:width],
    ddepth=-1,
    kernel=kernel,
    borderType=cv2.BORDER_DEFAULT
)

# rotacia o 90
part2 = mosaic[0:height, width:2*width].copy()
rotated = np.zeros_like(part2)

for y in range(height):
    for x in range(width):
        rotated[x, width - 1 - y] = part2[y, x]  

mosaic[0:height, width:2*width] = rotated

# cerveny kanal
part3 = mosaic[height:2*height, 0:width]
part3[:, :, 0] = 0  # B
part3[:, :, 1] = 0  # G

cv2.imshow("Mosaic 2x2", mosaic)
cv2.imwrite("mosaic.png", mosaic) 
cv2.waitKey(0)
cv2.destroyAllWindows()

print("dtype:", mosaic.dtype)
print("shape:", mosaic.shape)
print("size:", mosaic.size)