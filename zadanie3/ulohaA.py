from fileinput import filename

import cv2
import numpy as np

def load_image(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (400, 400))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_histogram(img):
    hist = np.zeros(256, dtype=int)
    for pixel in img.flatten():
        hist[pixel] += 1
    return hist

def global_threshold(img, T):
    result = np.zeros_like(img)
    result[img >= T] = 255
    return result

def otsu_threshold(img):
    hist = make_histogram(img)
    total_pixels = img.size
    sum_total = np.dot(np.arange(256), hist.flatten())
    sumB = 0
    wB = 0
    max_var_between = 0
    threshold = 0

    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total_pixels - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var_between:
            max_var_between = var_between
            threshold = i

    result = np.zeros_like(img)
    result[img >= threshold] = 255
    return result


def main():
    image = load_image("zadanie3\captured_image_1.png")
    image_grey = global_threshold(image, 128)
    show_image("Global Threshold", image_grey)
    image_otsu = otsu_threshold(image)
    show_image("Otsu Threshold", image_otsu)

if __name__ == "__main__":
    main()