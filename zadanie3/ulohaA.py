from fileinput import filename

import cv2
import numpy as np

def global_threshold(img, T):
    result = np.zeros_like(img)
    result[img >= T] = 255
    return result

def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_image(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (400, 400))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def main():
    image = load_image("zadanie3\captured_image_1.png")
    image_grey = global_threshold(image, 128)
    show_image("Global Threshold", image_grey)

if __name__ == "__main__":
    main()