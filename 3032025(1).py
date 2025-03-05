import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(image, title):
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

def ensure_256_bins(ref_hist):
    """
    Ensure that the reference histogram has exactly 256 values.
    If the reference histogram has fewer values, interpolate to get 256.
    """
    ref_hist = np.array(ref_hist)
    if len(ref_hist) == 256:
        return ref_hist  # Already correct size

    # Generate new 256-bin histogram using linear interpolation
    x_original = np.linspace(0, 255, len(ref_hist))  # Original indices
    x_new = np.arange(256)  # Target 256 indices
    ref_hist_interpolated = np.interp(x_new, x_original, ref_hist)

    return ref_hist_interpolated

def histogram_specification(source_img, ref_hist):
    matched_img = np.zeros_like(source_img)

    for i in range(3):
        source_hist, bins = np.histogram(source_img[:, :, i].flatten(), 256, [0, 256])
        source_cdf = source_hist.cumsum()
        source_cdf_normalized = source_cdf / source_cdf[-1]

        # Ensure reference histogram has 256 values
        ref_hist[i] = ensure_256_bins(ref_hist[i])

        # Normalize reference histogram
        ref_cdf_normalized = np.cumsum(ref_hist[i] / ref_hist[i].sum())

        # Interpolate CDF values to match lengths
        source_cdf_normalized = np.interp(np.linspace(0, 1, 256), source_cdf_normalized, np.linspace(0, 255, 256))
        ref_cdf_normalized = np.interp(np.linspace(0, 1, 256), ref_cdf_normalized, np.linspace(0, 255, 256))

        mapping = np.interp(source_cdf_normalized, ref_cdf_normalized, bins[:-1])
        matched_img[:, :, i] = np.interp(source_img[:, :, i].flatten(), bins[:-1], mapping).reshape(source_img[:, :, i].shape)

    return matched_img

# Load image
image = cv2.imread("moon.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load reference histogram from Excel
ref_data = pd.read_excel("HistogramSpecificationData.xlsx")

# Ensure reference histograms have exactly 256 values using interpolation
ref_hist = [
    ensure_256_bins(ref_data["Red"].values),
    ensure_256_bins(ref_data["Green"].values),
    ensure_256_bins(ref_data["Blue"].values)
]

# Histogram Equalization
image_equalized = np.zeros_like(image_rgb)
for i in range(3):
    image_equalized[:, :, i] = cv2.equalizeHist(image_rgb[:, :, i])

# Perform histogram specification
image_specified = histogram_specification(image_rgb, ref_hist)

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 3, 2)
plot_histogram(image_rgb, "Original Histogram")

plt.subplot(3, 3, 4)
plt.imshow(image_equalized)
plt.title("Equalized Image")
plt.axis("off")

plt.subplot(3, 3, 5)
plot_histogram(image_equalized, "Equalized Histogram")

plt.subplot(3, 3, 7)
plt.imshow(image_specified.astype(np.uint8))
plt.title("Specified Image")
plt.axis("off")

plt.subplot(3, 3, 8)
plot_histogram(image_specified, "Specified Histogram")

plt.tight_layout()
plt.show()

print("Conclusion:")
print("1. Histogram equalization improves contrast and spreads out the dynamic range.")
print("2. Histogram specification modifies the histogram to match a reference, providing targeted brightness or contrast.")
