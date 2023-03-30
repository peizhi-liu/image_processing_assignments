from PIL import Image
import os

# 读取原始图片
img = Image.open("图像处理/standard_test_images/lena_color_512.tif")

# 定义分辨率和灰度级别
resolutions = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # 不同的分辨率比例
grayscale_levels = [2, 4, 8, 16, 32, 64, 128, 256]  # 不同的灰度级别

# 定义保存路径
save_path = "图像处理/resolution/results"

# 如果保存路径不存在，则创建它
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 创建一个新的空白图像
num_images = len(resolutions) * len(grayscale_levels)
width, height = img.size
total_width = width * len(grayscale_levels)
total_height = height * len(resolutions)
result_img = Image.new('RGB', (total_width, total_height))

# 遍历所有的分辨率和灰度级别，并生成新图片
for i, res in enumerate(resolutions):
    for j, gray_level in enumerate(grayscale_levels):
        # 计算新图片的大小
        new_width = int(width * res)
        new_height = int(height * res)

        # 调整图片大小
        new_img = img.resize((new_width, new_height))

        # 转换灰度级别
        new_img = new_img.convert('L')  # 转换为灰度图像
        new_img = new_img.quantize(colors=gray_level)  # 转换为指定灰度级别的图像

        # 粘贴新图片到结果图像中
        x = j * width
        y = i * height
        result_img.paste(new_img, (x, y))

        # 保存新图片
        file_name = f"new_image_{res}_{gray_level}.jpg"
        file_path = os.path.join(save_path, file_name)
        if new_img.mode == 'P' or new_img.mode == 'RGBA':
            new_img = new_img.convert('RGB')
        new_img.save(file_path)

# 保存结果图像
result_img.save(os.path.join(save_path, "result_image.jpg"))
