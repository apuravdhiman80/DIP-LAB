import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Scene.jpg')
if image is None:
    raise FileNotFoundError("Scene.jpg not found in the working directory.")

b, g, r = cv2.split(image)

low_threshold = 50
high_threshold = 150

edges_b = cv2.Canny(b, low_threshold, high_threshold)
edges_g = cv2.Canny(g, low_threshold, high_threshold)
edges_r = cv2.Canny(r, low_threshold, high_threshold)

edges_merged = cv2.merge((edges_b, edges_g, edges_r))

def plot_image_and_histogram(title, img, is_gray=False):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    if is_gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'{title} Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if is_gray:
        plt.hist(img.ravel(), 256, [0, 256], color='black')
    else:
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            plt.hist(img[:, :, i].ravel(), 256, [0, 256], color=col, alpha=0.5)
    plt.title(f'{title} Histogram')
    plt.tight_layout()
    plt.show()

plot_image_and_histogram("Original", image)
plot_image_and_histogram("Canny Edge Detected", edges_merged, is_gray=False)
