import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def apply_laplacian_filter(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image. Check the file path.")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    laplacian_kernel = np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]])
    
    filtered_image = image.copy()
    for i in range(3):
        filtered_plane = convolve(image[:, :, i], laplacian_kernel)
        sharpened_plane = image[:, :, i] - filtered_plane
        filtered_image[:, :, i] = np.clip(sharpened_plane, 0, 255)
    
    filtered_image = filtered_image.astype(np.uint8)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(filtered_image)
    axes[1].set_title('Filtered Image (Laplacian 3x3)')
    axes[1].axis('off')
    
    plt.show()
    
image_path = 'image.jpg'
apply_laplacian_filter(image_path)
