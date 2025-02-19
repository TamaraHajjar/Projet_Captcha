import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the color image (in BGR format)
image = cv2.imread('C:/Users/MC/Desktop/PFE S5/dataset/abacus_captcha_imgs/0b75a677690bb65d8f6d0a70507e8a1e_checkbox_image.png')  # Make sure the image is in color (RGB or BGR)

# Apply Gaussian filtering to smooth the image
smoothed_image = cv2.GaussianBlur(image, (3, 3), 0)

# Sharpen the image using unsharp masking
sharpened_image = cv2.addWeighted(image, 1.5, smoothed_image, -0.5, 0)

# Show the original, smoothed, and sharpened images
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for display
plt.title("Original Image")
plt.axis('off')

# Smoothed (Gaussian filtered) image
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB))
plt.title("Smoothed Image (Gaussian Blur)")
plt.axis('off')

# Sharpened image
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.title("Sharpened Image (Unsharp Masking)")
plt.axis('off')

plt.show()
