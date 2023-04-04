import numpy as np
from scipy import fftpack
from PIL import Image


# Load the image
im = Image.open("F:/PythonProject/image_processing/standard_test_images/cameraman.tif")
im_array = np.array(im)

# Compute the Fourier transforms of the image
im_fft = fftpack.fft2(im_array)

# Method 1: Restore the image using the magnitude and phase spectra
magnitude = np.abs(im_fft)
phase = np.angle(im_fft)
restored1 = np.real(fftpack.ifft2(magnitude * np.exp(1j * phase)))
restored1 = (restored1 / np.max(restored1) * 255).astype(np.uint8)
Image.fromarray(restored1).save("F:/PythonProject/image_processing/Fourier_transform/restored1.png")

# Method 2: Restore the image using the magnitude spectrum and zero phase
magnitude = np.abs(im_fft)
phase = np.zeros_like(im_fft)
restored2 = np.real(fftpack.ifft2(magnitude * np.exp(1j * phase)))
restored2 = (restored2 / np.max(restored2) * 255).astype(np.uint8)
Image.fromarray(restored2).save("F:/PythonProject/image_processing/Fourier_transform/restored2.png")

# Method 3: Restore the image using the given amplitude and phase spectra
amplitude = 5
phase = np.angle(im_fft)
restored3 = np.real(fftpack.ifft2(amplitude * np.exp(1j * phase)))
restored3 = (restored3 / np.max(restored3) * 255).astype(np.uint8)
Image.fromarray(restored3).save("F:/PythonProject/image_processing/Fourier_transform/restored3.png")

