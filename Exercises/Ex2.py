import numpy as np
import cv2


def ex_1():
    print("Exercise 1")
    img = cv2.imread('Sonic.jpg')
    print(type(img[0, 0, 0]))
    print(img.shape)
    print(img.ndim)
    cv2.imshow('Sonic', img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return


def ex_2():
    print("Exercise 2")
    img = cv2.imread('Sonic.jpg')
    img = np.float64(img)
    img[:, :, :] /= 255
    img[:, :, :] *= 255
    cv2.imshow('Sonic', np.uint8(img))
    cv2.waitKey(0)
    return


def ex_3():
    print("Exercise 3")
    img = cv2.imread('Sonic.jpg')

    cv2.imshow('Sonic', img)
    cv2.waitKey(0)
    return


def ex_4():
    print("Exercise 4")
    return


def ex_5():
    print("Exercise 5")
    return


def ex_6():
    print("Exercise 6")
    return


def ex_7():
    print("Exercise 7")
    return


# if __name__ == "__main__":
#     ex_3()
