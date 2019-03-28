import numpy as np
import cv2

def ErosionFilter(img):
    rows, cols = img.shape
    paddedimg = np.ones((rows + 2, cols + 2))
    paddedimg[:, :] = 255
    paddedimg[1:-1, 1:-1] = img.copy()
    newimg = img.copy()
    for x in range(0, rows):
        for y in range(0, cols):
            if img[x, y] != 0:
                newimg[x, y] = ErodePixel(paddedimg, x+1, y+1)

    return np.uint8(newimg)


def ErodePixel(img, i, j):
    kernel = np.zeros((3, 3))
    kernel[:, :] = img[i-1:i+2, j-1:j+2]
    for a in range(0, 2):
        for b in range(0, 2):
            if kernel[a, b] == 0:
                return 0
    return 255


if __name__ == "__main__":
    image = cv2.imread("morphology.png", cv2.IMREAD_GRAYSCALE)
    filteredImg = ErosionFilter(ErosionFilter(image))
    cv2.imshow('Image', np.hstack((image, filteredImg)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
