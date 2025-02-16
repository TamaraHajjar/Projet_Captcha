import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os
from glob import glob

# Custom transform for applying Gaussian filtering
class GaussianBlurTransform:
    def __init__(self, kernel_size=5, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        img = np.array(img)  # Convert PIL image to NumPy array
        img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
        return Image.fromarray(img)  # Convert back to PIL image

# Custom transform for applying Median filtering
class MedianFilterTransform:
    def __init__(self, kernel_size=5):
        """
        :param kernel_size: Must be an odd integer (e.g., 3, 5, 7).
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer.")
        self.kernel_size = kernel_size

    def __call__(self, img):
        img = np.array(img)  # Convert PIL image to NumPy array
        img = cv2.medianBlur(img, self.kernel_size)  # Apply median filtering
        return Image.fromarray(img)  # Convert back to PIL image

# Define the gaussian transformation pipeline
transform_gaussian = transforms.Compose([
    GaussianBlurTransform(kernel_size=3, sigma=1.0),  # Apply Gaussian blur
    transforms.Grayscale(),  # Convert RGB to grayscale
    transforms.ToTensor(),  # Convert to tensor (scales values to [0,1])
    transforms.Normalize(mean=[0.35115125], std=[0.25548151])  # Normalize (adjust as needed)
])

# Define the median transformation pipeline
transform_median = transforms.Compose([
    MedianFilterTransform(kernel_size=3),
    transforms.Grayscale(),  # Convert RGB to grayscale
    transforms.ToTensor(),  # Convert to tensor (scales values to [0,1])
    transforms.Normalize(mean=[0.35115125], std=[0.25548151])  # Normalize (adjust as needed)
])

# Load an example image
#image = Image.open("C:/Users/MC/Desktop/PFE S5/Code/segmented_captchas/3b9b3bf40721eb834b3f1567abe51c0a_checkbox_image_circle_1.png")  # Load RGB image

seg_cap = Path("C:/Users/MC/Desktop/PFE S5/Code/segmented_captchas").glob("*.png")

# Apply gaussian filtering as transformation
save_dir = "gaussian_captchas"
os.makedirs(save_dir, exist_ok=True)

for sc in seg_cap:
    # Load image as PIL Image
    image = Image.open(sc).convert("RGB")  # Ensure RGB format
    transformed_image_gaussian = transform_gaussian(image)  # Convert to string if needed
    save_path = os.path.join(save_dir, sc.name + '_transformed_image_gaussian')  # Save with the same filename
    transformed_image_gaussian.save(save_path)  # Save the image
    
# Apply median filtering as transformation       
save_dir = "median_captchas"
os.makedirs(save_dir, exist_ok=True)  

for sc in seg_cap:
    image = Image.open(sc).convert("RGB")  # Ensure RGB format
    transformed_image_median = transform_median(image)  # Convert to string if needed
    save_path = os.path.join(save_dir, sc.name + '_transformed_image_median')  # Save with the same filename
    transformed_image_median.save(save_path)  # Save the image
    
# # Apply transformations
# transformed_image = transform(image)

# # Steps to display the image
# transformed_image_cpu = transformed_image.cpu()  # Move tensor to CPU
# transformed_image_np = transformed_image_cpu.numpy()

# # If the image has the shape (channels, height, width), transpose it to (height, width, channels)
# if transformed_image_np.ndim == 3:
#     transformed_image_np = transformed_image_np.transpose(1, 2, 0)

# # If the image is grayscale (only one channel), remove the channel dimension
# if transformed_image_np.shape[-1] == 1:
#     transformed_image_np = transformed_image_np.squeeze(-1)

# plt.imshow(transformed_image_np, cmap='gray')
# plt.title(f"Transformed image ")
# plt.axis('off')  # Hide the axes
# plt.show()
                
# Print tensor shape to verify
#print(transformed_image.shape)  # Output should be [1, H, W] for grayscale images
