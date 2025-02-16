

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the color image (in BGR format)
image = cv2.imread('C:/Users/MC/Desktop/PFE S5/Code/data/abacus_label_imgs/76a41a564b2441cbd8bd3742af12da67_text_image.png')  # Make sure to replace with your image path

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY) 
ret, thresh2 = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV) 
ret, thresh3 = cv2.threshold(gray_image, 120, 255, cv2.THRESH_TRUNC) 
ret, thresh4 = cv2.threshold(gray_image, 120, 255, cv2.THRESH_TOZERO) 
ret, thresh5 = cv2.threshold(gray_image, 120, 255, cv2.THRESH_TOZERO_INV) 
  
# Apply morphological operations (dilation) to connect fragmented characters
kernel = np.ones((3, 3), np.uint8)
dilated_image_thresh1 = cv2.dilate(thresh1, kernel, iterations=1)
dilated_image_thresh2 = cv2.dilate(thresh2, kernel, iterations=1)
dilated_image_thresh3 = cv2.dilate(thresh3, kernel, iterations=1)
dilated_image_thresh4 = cv2.dilate(thresh4, kernel, iterations=1)
dilated_image_thresh5 = cv2.dilate(thresh5, kernel, iterations=1)

dilated_image_gray_image = cv2.dilate(gray_image, kernel, iterations=1)

# # Show the grayscale image
# plt.figure(figsize=(6, 6))
# plt.imshow(thresh1, cmap='gray')
# plt.title("THRESH_BINARY Image")
# plt.imshow(thresh2, cmap='gray')
# plt.title("THRESH_BINARY_INV Image")
# plt.imshow(thresh3, cmap='gray')
# plt.title("THRESH_TRUNC Image")
# plt.imshow(thresh4, cmap='gray')
# plt.title("THRESH_TOZERO Image")
# plt.imshow(thresh5, cmap='gray')
# plt.title("THRESH_TOZERO_INV Image")
# plt.axis('off')
# plt.show() 


# Create a figure and 2x3 grid of subplots (2 rows, 3 columns)
fig, ax = plt.subplots(2, 3, figsize=(12, 8))  # Adjust the figsize if necessary

# Show Image 1: Grayscaled Image
ax[0, 0].imshow(dilated_image_thresh1, cmap='gray')  # Using 'gray' colormap to ensure grayscale
ax[0, 0].set_title("THRESH_BINARY")
ax[0, 0].axis("off")  # Hide axes

# Show Image 2: Processed Image (CLAHE)
ax[0, 1].imshow(dilated_image_thresh2, cmap='gray')  # Assuming enhanced_clahe is a grayscale image
ax[0, 1].set_title("THRESH_BINARY_INV")
ax[0, 1].axis("off")  # Hide axes

# Show Image 3: Thresholded Image 1 (Binary)
ax[0, 2].imshow(dilated_image_thresh3, cmap='gray')
ax[0, 2].set_title("THRESH_TRUNC")
ax[0, 2].axis("off")

# Show Image 4: Thresholded Image 2 (Binary Inverted)
ax[1, 0].imshow(dilated_image_thresh4, cmap='gray')
ax[1, 0].set_title("THRESH_TOZERO")
ax[1, 0].axis("off")

# Show Image 5: Thresholded Image 3 (Truncated)
ax[1, 1].imshow(dilated_image_thresh5, cmap='gray')
ax[1, 1].set_title("THRESH_TOZERO_INV")
ax[1, 1].axis("off")

# Hide the empty subplot (1, 2)
ax[1, 2].imshow(dilated_image_gray_image, cmap='gray')
ax[1, 2].set_title("gray image")
ax[1, 2].axis("off")

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()


# Show the plot
plt.show()

# # De-allocate any associated memory usage   
# if cv2.waitKey(0) & 0xff == 27:  
#     cv2.destroyAllWindows()  