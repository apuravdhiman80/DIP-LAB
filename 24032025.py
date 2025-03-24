import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_filter(img, mask_type='lowpass', sigma=10):
    img_out = np.zeros_like(img, dtype=np.float32)

    for i in range(3):
        channel = img[:, :, i]
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.ones((rows, cols), dtype=np.float32)

        if mask_type == 'lowpass':
            mask[:] = 0
            mask[crow-1:crow+2, ccol-1:ccol+2] = 1

        elif mask_type == 'highpass':
            mask[crow-1:crow+2, ccol-1:ccol+2] = 0

        elif mask_type == 'gaussian_lowpass':
            y, x = np.ogrid[:rows, :cols]
            d2 = (x - ccol)**2 + (y - crow)**2
            mask = np.exp(-d2 / (2 * sigma**2))

        elif mask_type == 'gaussian_highpass':
            y, x = np.ogrid[:rows, :cols]
            d2 = (x - ccol)**2 + (y - crow)**2
            mask = 1 - np.exp(-d2 / (2 * sigma**2))

        fshift_filtered = fshift * mask

        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        img_out[:, :, i] = img_back

    return np.clip(img_out, 0, 255).astype(np.uint8)

img_bgr = cv2.imread('image.jpg')
if img_bgr is None:
    raise FileNotFoundError("Make sure 'image.jpg' is in the same directory.")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

lowpass = apply_filter(img_rgb, 'lowpass')
highpass = apply_filter(img_rgb, 'highpass')
gauss_lp = apply_filter(img_rgb, 'gaussian_lowpass', sigma=10)
gauss_hp = apply_filter(img_rgb, 'gaussian_highpass', sigma=10)

plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(img_rgb), plt.title("Original")
plt.axis('off')
plt.subplot(232), plt.imshow(lowpass), plt.title("Low Pass Filter (3x3)")
plt.axis('off')
plt.subplot(233), plt.imshow(highpass), plt.title("High Pass Filter (3x3)")
plt.axis('off')
plt.subplot(234), plt.imshow(gauss_lp), plt.title("Gaussian Low Pass")
plt.axis('off')
plt.subplot(235), plt.imshow(gauss_hp), plt.title("Gaussian High Pass")
plt.axis('off')
plt.tight_layout()
plt.show()
