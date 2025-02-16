import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the color image (in BGR format)
image = cv2.imread('C:/Users/MC/Desktop/Grayscale.png')  # Make sure the image is in color (RGB or BGR)

# Apply median filtering to each channel separately (R, G, B)
denoised_image = np.zeros_like(image)  # Create an empty array to store the result

# Loop through each channel (BGR order in OpenCV)
for i in range(3):  # 0 - Blue, 1 - Green, 2 - Red
    denoised_image[:, :, i] = cv2.medianBlur(image[:, :, i], 3)  # 5 is the kernel size

# Show the original and denoised images
plt.figure(figsize=(10,5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for display
plt.title("Original Image")
plt.axis('off')

# Denoised image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for display
plt.title("Denoised Image (Median Filter)")
plt.axis('off')

plt.show()
