import cv2
import numpy as np


def apply_watershed(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用阈值处理
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 对图像进行形态学操作，以便更好地定位图像中的区域
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 确定前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 找到不确定区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记不同的区域
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 应用分水岭算法
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # 将分割线标记为红色

    return image


# 加载图像
image = cv2.imread("F:/PythonProject/image_processing/standard_test_images/peppers_color.tif")

# 应用分水岭算法
result = apply_watershed(image)

# 保存结果图像
output_path = "F:/PythonProject/image_processing/watershed/results2.jpg"
cv2.imwrite(output_path, result)

