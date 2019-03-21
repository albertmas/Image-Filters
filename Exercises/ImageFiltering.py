import numpy as np
import cv2


def getLinearFilteredPixel(img, i, j, filter_num):
    kernel = np.ones((filter_num, filter_num))
    kernel /= (filter_num * 3)
    addition = [0, 0, 0]
    side = np.uint8((filter_num - 1) / 2)
    for a in range(0, filter_num):
        for b in range(0, filter_num):
            addition += kernel[a, b] * img[i + a - side, j + b - side, :]
    img[i, j, :] = addition
    return img


def getGaussianFilteredPixel(img, i, j, filter_num):
    # numpy.random.normal
    kernel = np.array(([1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]), float)
    kernel /= 273
    addition = [0, 0, 0]
    for a in range(0, 5):
        for b in range(0, 5):
            addition += kernel[a, b] * img[i + a - 2, j + b - 2, :]
    img[i, j, :] = addition
    return img


def getMedianFilteredPixel(img, i, j, filter_num):
    kernel = np.ones(filter_num, filter_num)


def getBilateralFilteredPixel(img, i, j, filter_num):
    kernel = np.array(([1, 4, 7, 4, 1],
                      [4, 16, 26, 16, 4],
                      [7, 26, 41, 26, 7],
                      [4, 16, 26, 16, 4],
                      [1, 4, 7, 4, 1]), float)
    kernel /= 273
    pixel_mean = (0.2126*img[i, j, 0] + 0.7152*img[i, j, 1] + 0.0722*img[i, j, 2])
    side = np.uint8((5 - 1) / 2)
    addition = [0, 0, 0]
    weightsum = 0
    for a in range(0, 5):
        for b in range(0, 5):
            mean = (0.2126*img[i + a - side, j + b - side, 0] + 0.7152*img[i + a - side, j + b - side, 1] + 0.0722*img[i + a - side, j + b - side, 2])
            kernel[a, b] *= ((255 - abs(pixel_mean - mean)) / 255)**2
            weightsum += kernel[a, b]
            addition += kernel[a, b] * img[i + a - side, j + b - side, :]
    addition /= weightsum
    img[i, j, :] = addition
    return img


def getErosionFilteredPixel(img, i, j, filter_num):
    if img[i, j, 0] == 0:
        return img
    kernel = np.ones(filter_num, filter_num)
    side = np.uint8((filter_num - 1) / 2)
    kernel[:, :, :] = img[i - side, j - side, :]
    for a in range(0, 5):
        for b in range(0, 5):
            if kernel[a, b, 0] == 0:
                img[i - side + a, j - side + b, 0] = 0
                return img
    return img


def boxFilter(img, filter_num=3):
    if filter_num % 2 == 0:
        return img
    rows, columns, depth = img.shape
    newimage = img.copy()
    paddedimage = np.zeros((rows + filter_num - 1, columns + filter_num - 1, depth))
    padding = np.uint8((filter_num - 1) / 2)
    a, b, _ = paddedimage.shape
    paddedimage[padding:a-padding, padding:b-padding, :] = img.copy()
    for i in range(0, rows):
        for j in range(0, columns):
            pixel = getGaussianFilteredPixel(paddedimage, i + padding, j + padding, filter_num)
            newimage[i, j, :] = pixel[i + padding, j + padding, :]
    return newimage


if __name__ == "__main__":
    image = cv2.imread("Lenna.png")
    filteredImg = boxFilter(image, 5)
    cv2.imshow('Image', image)
    cv2.imshow('Filtered image', filteredImg)
    cv2.imwrite('Lenna_Bilateral_2.png', filteredImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def getFilteredPixel(img, i, j, filter_num):
#     kernel = np.ones((filter_num * 3, filter_num * 3))
#     kernel /= 9
#     addition = [0, 0, 0]
#     for a in range(0, filter_num * 3):
#         for b in range(0, filter_num * 3):
#             addition += kernel[a, b] * img[i + a - filter_num, j + b - filter_num, :]
#     img[i, j, :] = addition
#     return img
#
#
# def boxFilter(img, filter_num):
#     rows, columns, depth = img.shape
#     newimage = img.copy()
#     paddedimage = np.zeros((rows + filter_num * 2, columns + filter_num * 2, depth))
#     paddedimage[filter_num:-filter_num, filter_num:-filter_num, :] = img
#     for i in range(0, rows):
#         for j in range(0, columns):
#             newimage[i, j, :] = getFilteredPixel(paddedimage, i + filter_num, j + filter_num, filter_num)[i + filter_num, j + filter_num, :]
#     return newimage
#
#
# if __name__ == "__main__":
#     image = cv2.imread("Lenna.png")
#     cv2.imshow('Image', image)
#     filteredImg = boxFilter(image, 3)
#     cv2.imshow('Filtered image', filteredImg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
