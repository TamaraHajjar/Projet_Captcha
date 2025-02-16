# --------------------------- CODE BIEN ---------------------------
'''
Code qui transforme une image en Grayscale et applique CLAHE
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the color image (in BGR format)
image = cv2.imread('C:/Users/MC/Desktop/PFE S5/Code/data/abacus_label_imgs/1be2d3dcb893c2caf014d42f195ba799_text_image.png')  # Make sure to replace with your image path

# Convert the BGR image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)
# Apply Otsu's Thresholding (automatically finds the best threshold value)
#_, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_clahe = clahe.apply(gray_image)
print(enhanced_clahe.shape)

# Normalize the image
# enhanced_clahe = enhanced_clahe.astype(np.float32) / 255.0
# print(enhanced_clahe.shape)

# # Create a figure and two subplots
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # (rows, columns)

# # Show Image 1
# ax[0].imshow(gray_image)
# ax[0].set_title("Grayscaled Image")
# ax[0].axis("off")  # Hide axes

# # Show Image 2
# ax[1].imshow(enhanced_clahe)
# ax[1].set_title("Processed Image CLAHE")
# ax[1].axis("off")  # Hide axes

# # Show the plot
# plt.show()

# Show the grayscale image
plt.figure(figsize=(6, 6))
plt.imshow(enhanced_clahe, cmap='gray')
plt.title("CLAHE Image")
plt.axis('off')
plt.show()

# Optionally, save the grayscale image
#cv2.imwrite('grayscale_image.png', gray_image)
