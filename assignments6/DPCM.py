import numpy as np
import cv2
from math import log10
from skimage.metrics import structural_similarity as ssim

def dpcm_encode(image):
    rows, cols = image.shape
    prediction = np.zeros((rows, cols), dtype=np.int32)
    error = np.zeros((rows, cols), dtype=np.int32)

    for i in range(1, rows):
        for j in range(1, cols):
            prediction[i][j] = image[i-1][j-1]
            error[i][j] = image[i][j] - prediction[i][j]

    return prediction, error

def dpcm_decode(prediction, error):
    rows, cols = prediction.shape
    decoded = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(1, rows):
        for j in range(1, cols):
            decoded[i][j] = prediction[i][j] + error[i][j]

    return decoded

def quantize(image, bits):
    levels = 2 ** bits
    scale = 255 / (levels - 1)
    return (np.round(image / scale) * scale).astype(np.uint8)

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    psnr = 10 * log10(255**2 / mse)
    return psnr

# Load grayscale image
image = cv2.imread("F:/PythonProject/image_processing/standard_test_images/lena_gray_256.tif", cv2.IMREAD_GRAYSCALE)

# Encode
prediction, error = dpcm_encode(image)

# Quantize and decode
quantizers = [1, 2, 4, 8]  # Different quantizers
reconstructed_images = []
psnr_values = []
ssim_values = []

for bits in quantizers:
    quantized_error = quantize(error, bits)
    reconstructed_image = dpcm_decode(prediction, quantized_error)
    reconstructed_images.append(reconstructed_image)
    psnr = calculate_psnr(image, reconstructed_image)
    ssim_value = ssim(image, reconstructed_image, data_range=image.max() - image.min())
    psnr_values.append(psnr)
    ssim_values.append(ssim_value)

# Print results
for i, bits in enumerate(quantizers):
    print(f"Quantizer: {bits}-bit")
    print(f"PSNR: {psnr_values[i]}")
    print(f"SSIM: {ssim_values[i]}")
    print()

# You can save the reconstructed images here, for example:
cv2.imwrite("F:/PythonProject/image_processing/DPCM/results/reconstructed_8_bit.jpg", reconstructed_images[3])
