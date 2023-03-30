from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 导入灰度图像
image = Image.open("图像处理/standard_test_images/lena_gray_512.tif")

# 线性点运算
linear_image = np.array(image) * 3

# 分段线性点运算
segmented_image = np.array(image)
segmented_image[segmented_image < 128] = segmented_image[segmented_image < 128] * 0.5
segmented_image[segmented_image >= 128] = segmented_image[segmented_image >= 128] * 1.5

# 非线性点运算
nonlinear_image = np.array(image) ** 0.5

# 绘制四张图片
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(linear_image, cmap='gray')
axes[0, 1].set_title('Linear Transformed Image')
axes[1, 0].imshow(segmented_image, cmap='gray')
axes[1, 0].set_title('Segmented Linear Transformed Image')
axes[1, 1].imshow(nonlinear_image, cmap='gray')
axes[1, 1].set_title('Nonlinear Transformed Image')

# 添加大标题
fig.suptitle('Point Operation')

plt.savefig("图像处理/Point_operation/results/result.png", dpi=600)
