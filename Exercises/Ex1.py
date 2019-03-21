import numpy as np
import cv2

print("Exercise 1")
vec1 = np.zeros(10)
print(vec1)

print("Exercise 2")
vec1[4] = 1
print(vec1)

print("Exercise 3")
vec2 = np.arange(10, 40)
print(vec2)

print("Exercise 4")
mat1 = np.arange(1., 10.)
mat1 = mat1.reshape(3, 3)
print(mat1)

print("Exercise 5")
mat2 = np.flip(mat1, 1)
print(mat2)

print("Exercise 6")
mat3 = np.flip(mat1, 0)
print(mat3)

print("Exercise 7")
matId = np.identity(3)
print(matId)

print("Exercise 8")
# np.random.seed(101)
matRand = np.random.rand(3, 3)
print(matRand)

print("Exercise 9")
vecRand = np.random.rand(10)
print(vecRand.mean())

print("Exercise 10")
matFramed = np.ones((10, 10))
matFramed[1:-1, 1:-1] = 0
print(matFramed)

print("Exercise 11")
vec3 = np.arange(1, 6)
mat4 = np.zeros((5, 5))
mat4[:, :] = vec3
print(mat4)

print("Exercise 12")
mat5 = np.random.randint(0, 200, 9)
print(mat5)
mat5 = np.float64(mat5.reshape(3, 3))
print(mat5)

print("Exercise 13")
matRand1 = np.random.rand(5, 5)
matRand1 -= matRand1.mean()
print(matRand1)

print("Exercise 14")
matRand2 = matRand1.copy()
matRand2[0, :] -= matRand2[0, :].mean()
matRand2[1, :] -= matRand2[1, :].mean()
matRand2[2, :] -= matRand2[2, :].mean()
matRand2[3, :] -= matRand2[3, :].mean()
matRand2[4, :] -= matRand2[4, :].mean()
print(matRand2)

print("Exercise 15")
matRand3 = np.random.rand(5, 5)
print(matRand3)
mat6 = matRand3.flatten()
closest = (np.abs(mat6 - 0.5)).argmin()
print(mat6[closest])

print("Exercise 16")
matRand4 = np.random.randint(0, 10, 9).reshape(3, 3)
print(matRand4)
vec4 = matRand4.reshape(9)
print(len(vec4[vec4 > 5]))

print("Exercise 17")
data = [0, 0, 0]
img = np.zeros((64, 64, 3), np.uint8)
for i in range(0, 64):
    img[:, i, 0] = i / 64 * 255
    img[:, i, 1] = i / 64 * 255
    img[:, i, 2] = i / 64 * 255
cv2.imshow('Fade', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Exercise 18")
img = img.transpose(1, 0, 2)
cv2.imshow('Fade', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Exercise 19")
data = [0, 255, 225]
img = np.zeros((64, 64, 3), np.uint8)
img[:, :, :] = data
cv2.imshow('Yellow', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Exercise 20")
img = np.zeros((64, 64, 3), np.uint8)
img[:, :, :] = [255, 255, 225]
img[:32, :32, :] = [0, 255, 225]
img[32:64, 32:64, :] = [255, 255, 0]
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Exercise 21")
img = cv2.imread('elephant.jpg')
img[::2, :, :] = [0, 0, 0]
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Exercise 22")
img = cv2.imread('elephant.jpg')
img[:, ::2, :] = [0, 0, 0]
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Exercise 23")
img = cv2.imread('elephant.jpg')
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_height, img_length, _ = img.shape
img = np.float64(img)
img[:, :, :] /= 255
img[:, :, :] /= 2
img[:, :, :] *= 255
cv2.imshow('Image', np.uint8(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
