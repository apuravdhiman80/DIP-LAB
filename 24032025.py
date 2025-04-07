import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_frequency_filter(image, filter_type):
    image_float = np.float32(image)
    fft = np.fft.fft2(image_float)
    fft_shifted = np.fft.fftshift(fft)
    rows, cols = image.shape
    mask = np.zeros((rows, cols), np.float32)
    center_row, center_col = rows // 2, cols // 2
    mask[center_row-1:center_row+2, center_col-1:center_col+2] = 1
    
    if filter_type == "lowpass":
        mask = mask
    elif filter_type == "highpass":
        mask = 1 - mask
    elif filter_type == "lowpass_gaussian":
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
    elif filter_type == "highpass_gaussian":
        mask = 1 - cv2.GaussianBlur(mask, (3, 3), 0)
    
    fft_filtered = fft_shifted * mask
    fft_ishifted = np.fft.ifftshift(fft_filtered)
    image_filtered = np.fft.ifft2(fft_ishifted)
    image_filtered = np.abs(image_filtered)
    
    return image_filtered

image_path = "image.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Unable to load image from {image_path}. Please check the file path.")
    exit()

image_lowpass = apply_frequency_filter(image, "lowpass")
image_highpass = apply_frequency_filter(image, "highpass")
image_lowpass_gaussian = apply_frequency_filter(image, "lowpass_gaussian")
image_highpass_gaussian = apply_frequency_filter(image, "highpass_gaussian")

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Input Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(image_lowpass, cmap="gray")
plt.title("Low Pass Filter")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(image_highpass, cmap="gray")
plt.title("High Pass Filter")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(image_lowpass_gaussian, cmap="gray")
plt.title("Lowpass Gaussian Filter")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(image_highpass_gaussian, cmap="gray")
plt.title("Highpass Gaussian Filter")
plt.axis("off")

plt.tight_layout()
plt.show()
