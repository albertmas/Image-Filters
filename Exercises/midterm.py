########################################################################
#                         ALBERT MAS COMPÉS
########################################################################

import cv2
import numpy as np


def GaussianFilter(img, kernel):
    rows, cols, comp = img.shape
    paddedimg = np.zeros((rows+10, cols+10, comp))
    paddedimg[5:-5, 5:-5, :] = img.copy()
    result = img.copy()
    for x in range(0, rows):
        for y in range(0, cols):
            result[x, y] = convolveGaussian(paddedimg, x, y, kernel)
    return result


def convolveGaussian(img, i, j, kernel):
    k_rows, k_cols = kernel.shape
    temporal = img[i:i + k_rows, j:j + k_cols]
    return (temporal * kernel[:, :, np.newaxis]).sum(axis=(0, 1))


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


def exercise1():
    img = np.float64(cv2.imread("noise.jpg", cv2.IMREAD_COLOR))
    rows, cols, comp = img.shape

    # TODO: Write your code here
    h_kernel = np.array([1., 10., 45., 120., 210., 252., 210., 120., 45., 10., 1.])
    full_h_kernel = np.zeros((11, 11))
    full_h_kernel[5, :] = h_kernel
    full_h_kernel /= full_h_kernel.sum(axis=(0, 1))

    blurred1 = GaussianFilter(img, full_h_kernel)
    full_v_kernel = full_h_kernel.transpose()
    blurred2 = GaussianFilter(blurred1, full_v_kernel)

    cv2.imshow("Blurred1", np.uint8(blurred1))
    cv2.imshow("Blurred2", np.uint8(blurred2))
    cv2.imshow("Original", np.uint8(img))
    cv2.waitKey(0)


def exercise2():
    img = cv2.imread("morphology.png", cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    # TODO: Write your code here
    filteredimg = ErosionFilter(ErosionFilter(img))

    cv2.imshow("Output", filteredimg)
    cv2.imshow("Input", img)
    cv2.waitKey(0)


if __name__ == '__main__':

    # Uncomment to execute exercise 1
    exercise1()

    # Uncomment to execute exercise 2
    exercise2()

    pass
