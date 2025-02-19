# --------------------------- CODE BIEN ---------------------------
'''
Ce code est pour appliquer le median filtering sur toutes les images de labels
Résultats dans le folder labels_median_filtering
'''
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
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
    transforms.Grayscale(),
    MedianFilterTransform(kernel_size=3),
    transforms.ToTensor(),
    ])

# Define source and destination folders
source_folder = "C:/Users/MC/Desktop/PFE S5/Code/data/abacus_label_imgs/"
destination_folder = "C:/Users/MC/Desktop/PFE S5/Code/labels_median_filtering/"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get a list of all image files in the source folder (you can adjust the file extensions if needed)
image_paths = glob(os.path.join(source_folder, "*.png"))  # Assuming PNG images, change if needed

# Loop through each image file
for image_path in image_paths:
    # Open the image
    image = Image.open(image_path)
    
    # Apply the transformation
    transformed_image = transform_median(image)
    
    # Convert tensor back to PIL image
    transformed_image_pil = transforms.ToPILImage()(transformed_image)

    # Get the file name from the path and create the save path
    file_name = os.path.basename(image_path)
    save_path = os.path.join(destination_folder, file_name)

    # Save the transformed image
    transformed_image_pil.save(save_path)