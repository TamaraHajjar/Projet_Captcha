import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt
from pathlib import Path
import os
from glob import glob

class UnsharpMasking:
    def __init__(self, radius, amount):
        self.radius = radius
        self.amount = amount

    def __call__(self, img):
        img = np.array(img)  # Convert PIL image to NumPy array
        img_sharp = unsharp_mask(img, self.radius, self.amount)
        return Image.fromarray(img_sharp)  # Convert back to PIL image

transform_unsharp_mask = transforms.Compose([
    transforms.Grayscale(),
    UnsharpMasking(radius=2, amount=1),
    transforms.ToTensor(),
    ])

# Define source and destination folders
source_folder = "C:/Users/MC/Desktop/PFE S5/Code/data/Train_labels/banknote/"
destination_folder = "C:/Users/MC/Desktop/PFE S5/Code/data/banknote_unsharp_mask"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get a list of all image files in the source folder (you can adjust the file extensions if needed)
image_paths = glob(os.path.join(source_folder, "*.png"))  # Assuming PNG images, change if needed

# Loop through each image file
for image_path in image_paths:
    # Open the image
    image = Image.open(image_path)
    
    # Apply the transformation
    transformed_image = transform_unsharp_mask(image)
    
    # Convert tensor back to PIL image
    transformed_image_pil = transforms.ToPILImage()(transformed_image)
    print("transformed_image_pil Type:", transformed_image_pil.dtype)

    # Get the file name from the path and create the save path
    file_name = os.path.basename(image_path)
    save_path = os.path.join(destination_folder, file_name)

    # Save the transformed image
    #transformed_image_pil.save(save_path)