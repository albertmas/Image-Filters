import numpy as np
import cv2


def SobelFilter(img):
    h_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    v_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Gx = applyFilter(img, h_kernel)
    Gy = applyFilter(img, v_kernel)
    G_result = np.sqrt((Gx**2 + Gy**2))

    return np.uint8(G_result)


def SobelFilterSplit(gx, gy):
    G_result = np.sqrt((gx**2 + gy**2))
    return G_result


def SobelFilterGx(img):
    h_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return applyFilter(img, h_kernel)


def SobelFilterGy(img):
    v_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return applyFilter(img, v_kernel)


def ScharrFilter(img):
    h_kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    v_kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

    Gx = applyFilter(img, h_kernel)
    Gy = applyFilter(img, v_kernel)

    G_result = np.sqrt((Gx**2 + Gy**2))

    return np.uint8(G_result)


def GaussianFilter(img):
    kernel = np.array(([2, 4, 5, 4, 2],
                       [4, 9, 12, 9, 4],
                       [5, 12, 15, 12, 5],
                       [4, 9, 12, 9, 4],
                       [2, 4, 5, 4, 2]), float)
    kernel /= 159
    return applyFilter(img, kernel)


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


def getEdgeDirection(gx, gy):
    result = np.arctan2(gx, gy)
    rows, columns = result.shape
    for x in range(0, rows):
        for y in range(0, columns):
            if result[x, y] < 0:
                result[x, y] += np.pi
            if abs(result[x, y]) < np.pi/8:
                result[x, y] = 0
            elif abs(abs(result[x, y]) - np.pi/4) < np.pi/8:
                result[x, y] = np.pi/4
            elif abs(abs(result[x, y]) - np.pi/2) < np.pi/8:
                result[x, y] = np.pi/2
            elif abs(abs(result[x, y]) - np.pi*3/4) < np.pi/8:
                result[x, y] = np.pi*3/4
            elif abs(abs(result[x, y]) - np.pi) < np.pi/8:
                result[x, y] = 0
    return result


def getLines(img_G, directions_G):
    img_M = img_G.copy()
    img_M[:, :] = 0
    rows_M, columns_M = img_M.shape
    paddedimg = np.zeros((rows_M+2, columns_M+2))
    paddedimg[1:-1, 1:-1] = img_G
    for x in range(0, rows_M):
        for y in range(0, columns_M):
            if directions_G[x, y] == 0:
                if (img_G[x, y] > paddedimg[x+1+1, y+1]) & (img_G[x, y] > paddedimg[x-1+1, y+1]):
                    img_M[x, y] = img_G[x, y]
            elif directions_G[x, y] == np.pi/4:
                if (img_G[x, y] > paddedimg[x+1+1, y-1+1]) & (img_G[x, y] > paddedimg[x-1+1, y+1+1]):
                    img_M[x, y] = img_G[x, y]
            elif directions_G[x, y] == np.pi/2:
                if (img_G[x, y] > paddedimg[x+1, y+1+1]) & (img_G[x, y] > paddedimg[x+1, y-1+1]):
                    img_M[x, y] = img_G[x, y]
            elif directions_G[x, y] == np.pi*3/4:
                if (img_G[x, y] > paddedimg[x+1+1, y+1+1]) & (img_G[x, y] > paddedimg[x-1+1, y-1+1]):
                    img_M[x, y] = img_G[x, y]

    return img_M


def HysteresisThreshold(img, minimum, maximum):
    rows, columns = img.shape
    newimg = np.zeros((rows, columns))
    for x in range(0, rows):
        for y in range(0, columns):
            if img[x, y] < minimum:
                newimg[x, y] = 0
            elif img[x, y] > maximum:
                newimg[x, y] = 255
            else:
                for a in range(-1, 1):
                    for b in range(-1, 1):
                        if img[x+a, y+b] > maximum:
                            newimg[x, y] = 255
    return newimg


def CannyFilter(img):
    blurred = GaussianFilter(img)
    # cv2.imshow('blurred', np.uint8(blurred))
    Gx = SobelFilterGx(blurred)
    # cv2.imshow('Gx', np.uint8(Gx))
    Gy = SobelFilterGy(blurred)
    # cv2.imshow('Gy', np.uint8(Gy))
    G_img = SobelFilterSplit(Gx, Gy)
    # cv2.imshow('G', np.uint8(G_img))
    directions = getEdgeDirection(Gx, Gy)
    # cv2.imshow('direc', np.uint8(np.rad2deg(directions)))
    M_img = getLines(G_img, directions)
    # cv2.imshow('M', np.uint8(M_img))
    M_img_filtered = HysteresisThreshold(M_img, 30, 50)
    return np.uint8(M_img_filtered)


if __name__ == "__main__":
    image = cv2.imread("Machine.png", cv2.IMREAD_GRAYSCALE)
    filteredImg = CannyFilter(image)
    cv2.imshow('Image', np.hstack((image, filteredImg)))
    # cv2.imwrite('Machine_Canny.png', filteredImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
