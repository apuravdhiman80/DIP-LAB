import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization_rgb(image):
    r, g, b = cv2.split(image)
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    return cv2.merge([r_eq, g_eq, b_eq])

def plot_histogram(image, ax, title):
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
    ax.set_title(title)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.grid(True)

def main():
    image = cv2.imread("image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    equalized_image = histogram_equalization_rgb(image)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(equalized_image)
    axes[0, 1].set_title("Equalized Image")
    axes[0, 1].axis('off')
    plot_histogram(image, axes[1, 0], "Histogram of Original Image")
    plot_histogram(equalized_image, axes[1, 1], "Histogram of Equalized Image")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
