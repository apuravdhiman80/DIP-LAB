import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def apply_median_filter(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    filtered_image = np.zeros_like(image)
    for i in range(3):
        filtered_image[:, :, i] = median_filter(image[:, :, i], size=3)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(filtered_image)
    axes[1].set_title('Filtered Image (Median 3x3)')
    axes[1].axis('off')
    
    plt.show()
    
image_path = 'image.jpg'
apply_median_filter(image_path)
