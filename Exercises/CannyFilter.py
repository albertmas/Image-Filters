import numpy as np
import cv2

def SobelFilter(img):
    Gx = SobelFilterGx(img)
    Gy = SobelFilterGy(img)
    G_result = np.sqrt((Gx**2 + Gy**2))

    return np.uint8(G_result)


def SobelFilterGx(img):
    h_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return np.uint8(applyFilter(img, h_kernel))


def SobelFilterGy(img):
    v_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return np.uint8(applyFilter(img, v_kernel))


def ScharrFilter(img):
    h_kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    v_kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

    Gx = applyFilter(img, h_kernel)
    Gy = applyFilter(img, v_kernel)

    G_result = np.sqrt((Gx**2 + Gy**2))

    return np.uint8(G_result)


def GaussianFilter(img):
    kernel = np.array(([1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]), float)
    kernel /= 273
    return np.uint8(applyFilter(img, kernel))


def applyFilter(img, filter):
    newimg = np.float64(img.copy())
    rows, columns = img.shape
    f_rows, f_columns = filter.shape
    f_rows_half = np.uint8(f_rows / 2)
    f_columns_half = np.uint8(f_columns / 2)
    for x in range(0, rows):
        for y in range(0, columns):
            submat = img[max(0, x-f_rows_half):min(rows, x+f_rows_half+1), max(0, y-f_columns_half):min(columns, y+f_columns_half+1)]
            f_submat = filter[max(f_rows_half-x, 0):f_rows-max(0, x+f_rows_half-rows+1), max(f_columns_half-y, 0):f_columns-max(0, y+f_columns_half-columns+1)]
            newimg[x, y] = np.sum(submat*f_submat)
    return newimg


def CannyFilter(img):
    newimg = GaussianFilter(img)
    newimg = SobelFilter(newimg)
    return newimg


if __name__ == "__main__":
    image = cv2.imread("Hermione.jpg", cv2.IMREAD_GRAYSCALE)
    filteredImg = CannyFilter(image)
    cv2.imshow('Image', np.hstack((image, filteredImg)))
    # cv2.imwrite('Hermione_Canny.png', filteredImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
