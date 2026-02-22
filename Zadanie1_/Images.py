from fileinput import filename

import cv2
import numpy as np

image1 = cv2.imread("captured_image_1.png") #Load image
image1 = cv2.resize(image1, (0, 0), None, .15, .15)

image2 = cv2.imread("captured_image_2.png") #Load image
image2 = cv2.resize(image2, (0, 0), None, .15, .15)

image3 = cv2.imread("captured_image_3.png") #Load image
image3 = cv2.resize(image3, (0, 0), None, .15, .15)

image4 = cv2.imread("captured_image_4.png") #Load image
image4 = cv2.resize(image4, (0, 0), None, .15, .15)

multiple_images = np.vstack((image1, image3))
multiple_images1 = np.vstack((image2, image4))

mosaic = np.hstack((multiple_images, multiple_images1))

original_mosaic = mosaic.copy()

filename = f'multiple_images_pic.png'
cv2.imwrite(filename, mosaic)

cv2.imshow('Original', original_mosaic)

#-----------------------------------------------------Úloha 1
# rozmery jedného obrázka v mozaike
h, w = image1.shape[:2]

# 1. časť mozaiky = horný-ľavý kvadrant (ROI selektor)
cast = mosaic[0:h, 0:w]

# 3x3 sharpening kernel (zvýrazní hrany, obraz bude "ostrejší")
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])

"""
Ak súčet = 1 → zachová sa jas
Ak súčet > 1 → obraz sa zosvetlí
Ak súčet < 1 → obraz stmavne
Ak súčet = 0 → detekcia hrán
"""

# konvolúcia cez filter2D, padding riešime borderType
cast_filtered = cv2.filter2D(cast, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)

# zapíš späť priamo do už existujúcej mozaiky
mosaic[0:h, 0:w] = cast_filtered


#-----------------------------------------------------Úloha 2
cast2 = mosaic[0:h, w:2*w]

# vytvor nový prázdny obraz (pozor - rozmery sa prehodia)
rotated = np.zeros((w, h, 3), dtype=np.uint8)

# ručné otočenie
for i in range(h):
    for j in range(w):
        rotated[j, h-1-i] = cast2[i, j]

# zapíš späť do mozaiky
mosaic[0:h, w:2*w] = cv2.resize(rotated, (w, h))


#-----------------------------------------------------Úloha 3
cast3 = mosaic[h:2*h, 0:w]

cast3[:, :, 0] = 0   # Blue
cast3[:, :, 1] = 0   # Green

cv2.imshow('Changed', mosaic)

print("Dátový typ:", mosaic.dtype)
print("Rozmer (shape):", mosaic.shape)
#výška, šírka, kanály
print("Veľkosť (počet prvkov):", mosaic.size)

cv2.waitKey()
