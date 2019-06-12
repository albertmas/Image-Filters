import cv2
import numpy as np


def ErosionFilter(img, times=1):
    rows, cols = img.shape
    paddedimg = np.ones((rows + 2, cols + 2))
    paddedimg[:, :] = 255
    paddedimg[1:-1, 1:-1] = img.copy()
    newimg = img.copy()
    for x in range(0, rows):
        for y in range(0, cols):
            if img[x, y] != 0:
                newimg[x, y] = ErodePixel(paddedimg, x+1, y+1)

    if times > 1:
        newimg = ErosionFilter(newimg, times-1)

    return np.uint8(newimg)


def ErodePixel(img, i, j):
    kernel = np.zeros((3, 3))
    kernel[:, :] = img[i-1:i+2, j-1:j+2]
    for a in range(0, 2):
        for b in range(0, 2):
            if kernel[a, b] == 0:
                return 0
    return 255


def DilationFilter(img, times=1):
    rows, cols = img.shape
    paddedimg = np.ones((rows + 2, cols + 2))
    paddedimg[:, :] = 0
    paddedimg[1:-1, 1:-1] = img.copy()
    newimg = img.copy()
    for x in range(0, rows):
        for y in range(0, cols):
            if img[x, y] != 255:
                newimg[x, y] = DilatePixel(paddedimg, x+1, y+1)

    if times > 1:
        newimg = DilationFilter(newimg, times-1)

    return np.uint8(newimg)


def DilatePixel(img, i, j):
    kernel = np.zeros((3, 3))
    kernel[:, :] = img[i-1:i+2, j-1:j+2]
    for a in range(0, 2):
        for b in range(0, 2):
            if kernel[a, b] == 255:
                return 255
    return 0


def exercise1():
	# Original image
    img = cv2.imread("binary.png", cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape
	
	# Destination image
    img2 = np.zeros((rows, cols), dtype=np.uint8)

	# TODO: Insert your code here
    img2 = DilationFilter(img)
    img2 = ErosionFilter(img2, 6)
	# Show output
    cv2.imshow("Binary", img)
    cv2.imshow("Binary 2", img2)
    cv2.waitKey(0)
    pass


if __name__ == '__main__':
    exercise1()
