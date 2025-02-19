# ---------------------------  NOT USED YET ---------------------------
''' Ce fichier sert à transformer les lbl images to binary images'''


# import cv2 as cv
# import matplotlib.pyplot as plt 

# # Read the colored image
# image = cv.imread("C:/Users/MC/Desktop/PFE S5/Code/data/Train_Labels/airplane/76a41a564b2441cbd8bd3742af12da67_text_image.png")  # Load your image

# # Step 1: Convert to Grayscale
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# # Step 3: Apply Thresholding (convert grayscale to binary)
# _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

# plt.imshow(thresh, cmap="gray")
# plt.axis("off")  # Hide axis labels

# plt.show()  # Show the processed images


# # # Now pass `thresh` to your function
# # filtered_images = calculate_corner_border(thresh)


# import cv2 as cv
# import os
# import operator
# from PIL import Image

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt 
# from torchvision import transforms
# from PIL import Image

# # Load the image
# image = cv2.imread('C:/Users/MC/Desktop/PFE S5/Code/data/Train_Labels/airplane/73ffcfee4cc78a378c089c5262daef43_text_image.png')
# image = Image.fromarray(image)

# #Transfrom the read image to grayscale
# grayscale_transform = transforms.Grayscale()
# gray = grayscale_transform(image)

# Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(gray)
# plt.show()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# # Threshold to make the background black (adjust threshold value as needed)
# _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)


# plt.imshow(gray)
# plt.show() 

# # Apply the mask to make the background black
# gray_with_black_bg = cv2.bitwise_and(gray, gray, mask=mask)


# plt.imshow(gray_with_black_bg)
# plt.show() 
# # Save or show the image
# cv2.imwrite('grayscale_black_bg.jpg', gray_with_black_bg)
# cv2.imshow('Grayscale with Black Background', gray_with_black_bg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# def calculate_corner_border(thresh, nrootdir="C:/Users/MC/Desktop/PFE S5/Code/data/cut_image/"):
#     # Read the original image for visualization (optional)
#     #show_img = cv.imread('temp.jpg')

#     # Find contours (still works on grayscale but less accurate)
#     contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     # Store contour information
#     new_contours = []
#     cur_contours = []
#     filter_container = []

#     # Extract bounding boxes and sort contours left-to-right
#     for i in contours:
#         x, y, w, h = cv.boundingRect(i)
#         cur_contours.append([x, y, w, h])
    
#     contours = sorted(cur_contours, key=operator.itemgetter(0))

#     for i in range(len(contours)):  
#         x, y, w, h = contours[i]
#         newimage = thresh[y:y+h, x:x+w]  # Extract character region

#         # Ignore very small areas (likely noise)
#         if h * w < 25:
#             continue

#         # Adaptive border color based on the image background
#         border_color = int(newimage.mean())  # Take average gray level
#         top, bottom, left, right = [1] * 4

#         # Add border while maintaining grayscale compatibility
#         newimage = cv.copyMakeBorder(newimage, top, bottom, left, right, cv.BORDER_CONSTANT, value=border_color)

#         # Resize while maintaining details
#         newimage = cv.resize(newimage, (30, 60), interpolation=cv.INTER_AREA)

#         # Ensure output directory exists
#         if not os.path.isdir(nrootdir):
#             os.makedirs(nrootdir)

#         # Save the processed image
#         image_path = os.path.join(nrootdir, f"{i}.png")
#         cv.imwrite(image_path, newimage)

#         # Convert to PIL for returning
#         filter_container.append(Image.open(image_path))

#     return filter_container


# # Load the colored image
# image = cv.imread("C:/Users/MC/Desktop/PFE S5/Code/data/Train_Labels/airplane/73ffcfee4cc78a378c089c5262daef43_text_image.png")

# # Convert to grayscale (no thresholding!)
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# # Pass to function
# processed_images = calculate_corner_border(gray)

# # Display results
# for img in processed_images:
#     img.show()  # Opens each processed character as a PIL image

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load the image
image = Image.open("C:/Users/MC/Desktop/PFE S5/Code/data/Train_Labels/banknote/0bb281b845f0eb07c8c42289208ef5d2_text_image.png")

# Convert to grayscale using torchvision transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to 1-channel grayscale
    transforms.ToTensor()  # Convert to tensor for further processing
])

gray_tensor = transform(image)

# Convert tensor to NumPy array (scale 0-255)
gray_np = (gray_tensor.squeeze().numpy() * 255).astype(np.uint8)

# Apply thresholding (everything below 128 becomes black, above becomes white)
threshold_value = 120
binary_np = np.where(gray_np > threshold_value, 255, 0).astype(np.uint8)

# Convert back to PIL Image
binary_image = Image.fromarray(binary_np)

# Save or show the processed image
#binary_image.save("grayscale_thresholded.jpg")
binary_image.show()
