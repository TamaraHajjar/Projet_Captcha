import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from glob import glob

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

transform_median = transforms.Compose([
    MedianFilterTransform(kernel_size=3),
    transforms.ToTensor(),
])

# Define source and destination folders
source_folder = "C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/Train_Captchas_Copy/"
destination_folder = "C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/second_median_filtering/"

# Process images in all subdirectories
for root, _, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".png"):  # Assuming PNG images, change if needed
            image_path = os.path.join(root, file)
            
            # Open the image
            image = Image.open(image_path)
            
            # Apply the transformation
            transformed_image = transform_median(image)
            
            # Convert tensor back to PIL image
            transformed_image_pil = transforms.ToPILImage()(transformed_image)
            
            # Create corresponding subfolder structure in destination folder
            relative_path = os.path.relpath(root, source_folder)
            save_folder = os.path.join(destination_folder, relative_path)
            os.makedirs(save_folder, exist_ok=True)
            
            # Save the transformed image
            save_path = os.path.join(save_folder, file)
            transformed_image_pil.save(save_path)
