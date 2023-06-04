import numpy as np
import cv2
import matplotlib.pyplot as plt

def otsu_threshold(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = image.shape[0] * image.shape[1]
    max_variance = 0
    threshold = 0

    for i in range(256):
        w0 = np.sum(hist[:i]) / total_pixels
        w1 = np.sum(hist[i:]) / total_pixels

        if w0 == 0 or w1 == 0:
            continue

        u0 = np.sum(hist[:i] * np.arange(i)) / (w0 * total_pixels)
        u1 = np.sum(hist[i:] * np.arange(i, 256)) / (w1 * total_pixels)

        variance = w0 * w1 * (u0 - u1) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = i

    return threshold

def iterative_threshold(image, initial_threshold=127, max_iterations=100, epsilon=0.1):
    threshold = initial_threshold
    iteration = 0

    while True:
        lower = image[image <= threshold]
        higher = image[image > threshold]
        new_threshold = (np.mean(lower) + np.mean(higher)) / 2

        if abs(new_threshold - threshold) < epsilon or iteration >= max_iterations:
            break

        threshold = new_threshold
        iteration += 1

    return int(round(threshold))

# 加载灰度图像
image = cv2.imread("F:/PythonProject/image_processing/standard_test_images/lena_gray_256.tif", cv2.IMREAD_GRAYSCALE)

# 大津法阈值分割
otsu_thresh = otsu_threshold(image)
otsu_result = cv2.threshold(image, otsu_thresh, 255, cv2.THRESH_BINARY)[1]

# 迭代法阈值分割
iter_thresh = iterative_threshold(image)
iter_result = cv2.threshold(image, iter_thresh, 255, cv2.THRESH_BINARY)[1]

# 显示结果
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(otsu_result, cmap='gray')
axes[1].set_title('Otsu Thresholding')
axes[1].axis('off')
axes[2].imshow(iter_result, cmap='gray')
axes[2].set_title('Iterative Thresholding')
axes[2].axis('off')
plt.savefig("F:/PythonProject/image_processing/Threshold_segmentation/results/comparison.jpg")
plt.show()

# 计算性能差别
otsu_pixels = np.count_nonzero(otsu_result)
iter_pixels = np.count_nonzero(iter_result)

otsu_ratio = otsu_pixels / (image.shape[0] * image.shape[1])
iter_ratio = iter_pixels / (image.shape[0] * image.shape[1])

print(f"Otsu Thresholding Pixels: {otsu_pixels}")
print(f"Otsu Thresholding Ratio: {otsu_ratio:.4f}")
print(f"Iterative Thresholding Pixels: {iter_pixels}")
print(f"Iterative Thresholding Ratio: {iter_ratio:.4f}")
