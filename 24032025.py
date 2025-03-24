import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply frequency domain filtering
def apply_frequency_filter(image, filter_type):
    # Convert image to float32 for FFT
    image_float = np.float32(image)
    
    # Perform FFT and shift zero frequency to the center
    fft = np.fft.fft2(image_float)
    fft_shifted = np.fft.fftshift(fft)
    
    # Create a 3x3 mask
    rows, cols = image.shape
    mask = np.zeros((rows, cols), np.float32)
    
    # Define the 3x3 mask center
    center_row, center_col = rows // 2, cols // 2
    mask[center_row-1:center_row+2, center_col-1:center_col+2] = 1  # 3x3 mask
    
    # Apply filter type
    if filter_type == "lowpass":
        mask = mask  # Low Pass Filter (3x3 mask)
    elif filter_type == "highpass":
        mask = 1 - mask  # High Pass Filter (3x3 mask)
    elif filter_type == "lowpass_gaussian":
        mask = cv2.GaussianBlur(mask, (3, 3), 0)  # Lowpass Gaussian Filter
    elif filter_type == "highpass_gaussian":
        mask = 1 - cv2.GaussianBlur(mask, (3, 3), 0)  # Highpass Gaussian Filter
    
    # Apply the mask to the shifted FFT
    fft_filtered = fft_shifted * mask
    
    # Shift back and perform inverse FFT
    fft_ishifted = np.fft.ifftshift(fft_filtered)
    image_filtered = np.fft.ifft2(fft_ishifted)
    image_filtered = np.abs(image_filtered)  # Take the magnitude
    
    return image_filtered

# Read the input image
image_path = "image.jpg"  # Ensure the image path is correct
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

# Check if the image is loaded correctly
if image is None:
    print(f"Error: Unable to load image from {image_path}. Please check the file path.")
    exit()

# Apply filters
image_lowpass = apply_frequency_filter(image, "lowpass")
image_highpass = apply_frequency_filter(image, "highpass")
image_lowpass_gaussian = apply_frequency_filter(image, "lowpass_gaussian")
image_highpass_gaussian = apply_frequency_filter(image, "highpass_gaussian")

# Display the results
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