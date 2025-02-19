# --------------------------- CODE BIEN ---------------------------
'''
3ème étape
Ce code est pour appliquer le median filtering sur toutes les images segmentées des captchas
Résultat dans le folder Train_Captchas_unsharp_masking
Type des I/O: Reads images using OpenCV, applies transformations, and saves using Matplotlib's imsave().
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.filters import unsharp_mask
import os 

def unsharp_masking(img_path, radius, amount):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_sharp = unsharp_mask(img, radius, amount)
    return img_sharp

# Source and destination folders
source_folder = "C:/Users/MC/Desktop/PFE S5/Code/data/second_median_filtering/"
destination_folder = "C:/Users/MC/Desktop/PFE S5/Code/data/second_median_filtering_unsharp_masking/"

# Get a list of all image files in subdirectories
image_paths = glob(os.path.join(source_folder, "**", "*.png"), recursive=True)

# Dictionary to track image counts per subfolder
folder_image_count = {}

# Loop through each image file
for img_pth in image_paths:
    # Generate relative path and corresponding destination path
    rel_path = os.path.relpath(img_pth, source_folder)
    subfolder = os.path.dirname(rel_path) # C:/Users/MC/Desktop/PFE S5/Code/data/Train_Captchas/airplane/005286.png
    
    # Ensure destination subdirectory exists
    dest_subfolder = os.path.join(destination_folder, subfolder)
    os.makedirs(dest_subfolder, exist_ok=True)
    
    # Count images per subfolder
    if subfolder not in folder_image_count:
        folder_image_count[subfolder] = 1
    else:
        folder_image_count[subfolder] += 1
    
    # Generate new file name
    new_filename = f"{subfolder.replace(os.sep, '_')}_{folder_image_count[subfolder]}.png"
    save_path = os.path.join(dest_subfolder, new_filename)
    
    # Apply unsharp masking
    img_sharp = unsharp_masking(img_pth, radius=2, amount=1)
    
    # Save the sharpened image
    plt.imsave(save_path, img_sharp)
    
    print(f"Processed: {img_pth} -> {save_path}")

