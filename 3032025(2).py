import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_histogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    return hist, cdf, cdf_normalized

def histogram_specification(source, ref_cdf):
    source_hist, _, source_cdf = calculate_histogram(source)
    mapping = np.interp(source_cdf, ref_cdf, np.arange(256))
    specified_image = mapping[source].astype(np.uint8)
    return specified_image

image = cv2.imread("moon.jpg")
if image is None:
    raise FileNotFoundError("moon.jpg not found!")\

gray_image = np.mean(image, axis=2).astype(np.uint8)

ref_data = pd.read_excel("HistogramSpecificationData.xlsx")
ref_gray_cdf = np.cumsum(np.histogram(ref_data["Red"], 256, [0, 256])[0])
ref_gray_cdf = ref_gray_cdf / ref_gray_cdf[-1]

specified_image = histogram_specification(gray_image, ref_gray_cdf)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(gray_image, cmap="gray")
axes[0, 0].set_title("Original Grayscale Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(specified_image, cmap="gray")
axes[0, 1].set_title("Histogram Specified Image")
axes[0, 1].axis("off")

original_hist, _, _ = calculate_histogram(gray_image)
specified_hist, _, _ = calculate_histogram(specified_image)

axes[1, 0].plot(original_hist, color="black")
axes[1, 0].set_title("Original Image Histogram")

axes[1, 1].plot(specified_hist, color="black")
axes[1, 1].set_title("Histogram Specified Image Histogram")

plt.tight_layout()
plt.show()
