# --------------------------- CODE BIEN ---------------------------
'''
Ce code est pour appliquer le median filtering sur une image grayscaled
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('C:/Users/MC/Desktop/Grayscale.png', cv2.IMREAD_GRAYSCALE)  # Assuming the image is grayscale

# Apply median filtering
denoised_image = cv2.medianBlur(image, 5)  # 5 is the kernel size, you can adjust it

# Show the original and denoised images
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(denoised_image, cmap='gray')
plt.title("Denoised Image (Median Filter)")
plt.axis('off')

plt.show()
