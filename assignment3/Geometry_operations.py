import cv2
import numpy as np

# 读取图像
img = cv2.imread('image_processing/standard_test_images/lena_color_512.tif', flags=1)

# 显示原始图像
cv2.imshow('Original', img)
cv2.waitKey()

# 图像平移
rows, cols = img.shape[:2]
M = np.float32([[1, 0, int(cols/2)], [0, 1, int(rows/2)]])
img_translate = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('Translation', img_translate)
cv2.waitKey()

# 图像镜像
img_mirror = cv2.flip(img, 1)
cv2.imshow('Mirror', img_mirror)
cv2.waitKey()

# 图像旋转
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
img_rotate = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('Rotation', img_rotate)
cv2.waitKey()

# 图像平移后镜像
M = np.float32([[1, 0, int(cols/2)], [0, -1, int(rows/2)]])
img_translate_mirror = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('Translation Mirror', img_translate_mirror)
cv2.waitKey()

# 图像平移后旋转
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
M = np.float32([[M[0, 0], M[0, 1], int(cols/2)], [M[1, 0], M[1, 1], int(rows/2)]])
img_translate_rotate = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('Translation Rotation', img_translate_rotate)
cv2.waitKey()

# 图像镜像后平移
M = np.float32([[1, 0, int(cols/2)], [0, 1, int(rows/2)]])
img_mirror_translate = cv2.flip(img, 1)
img_mirror_translate = cv2.warpAffine(img_mirror_translate, M, (cols, rows))
cv2.imshow('Mirror Translation', img_mirror_translate)
cv2.waitKey()

# 图像镜像后旋转
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
M = np.float32([[M[0, 0], -M[0, 1], cols], [-M[1, 0], M[1, 1], int(rows/2)]])
img_mirror_rotate = cv2.flip(img, 1)
img_mirror_rotate = cv2.warpAffine(img_mirror_rotate, M, (cols, rows))
cv2.imshow('Mirror Rotation', img_mirror_rotate)
cv2.waitKey()

cv2.waitKey(0)
cv2.destroyAllWindows()
