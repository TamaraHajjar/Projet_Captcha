# --------------------------- CODE BIEN ---------------------------
'''
4ème étape:
Sur les images unsharp masking, on applique un masque de luminance et un masque de pie 
slice pour rogner les images pour passer de la forme rectangulaire --> circulaire
Cette étape est nécessaire pour s'affranchir des larges zones noires inutiles dans l'img
et que le modèle de deep learning puisse apprendre correctement
Résultat dans le folder Train_Captchas_UM_Circular_Crop
'''

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from glob import glob

# Folder with images after unsharp masking (contains subfolders)
#source_folder = "C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/Train_Captchas_unsharp_masking/"
source_folder = "C:/Users/MC/Desktop/PFE S5/figures_article/"

# Folder where the processed images will be saved
#destination_folder = "C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/Train_Captchas_UM_Circular_Crop/"
destination_folder = "C:/Users/MC/Desktop/PFE S5/figures_article/"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Function to process and apply luminance + pie slice
def process_image(image_path, save_path):
    # Open the image
    img = Image.open(image_path)

    # Get image dimensions
    height, width = img.size

    # Create a new image in 'L' (luminance) mode
    lum_img = Image.new('L', [height, width], 0)

    # Draw a pie slice on the luminance image
    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(0, 0), (height, width)], 0, 360, fill=255, outline="white")

    # Convert images to numpy arrays
    img_arr = np.array(img)
    lum_img_arr = np.array(lum_img)

    # Ensure the luminance image has the same number of dimensions as the RGB image
    lum_img_arr = np.expand_dims(lum_img_arr, axis=-1)  # Add a channel dimension

    # Stack the image and luminance array along the third axis (channels)
    final_img_arr = np.concatenate((img_arr, lum_img_arr), axis=-1)

    # Convert final image array to an Image object
    img_final = Image.fromarray(final_img_arr.astype(np.uint8))

    # Save the final image
    img_final.save(save_path)

# Loop through the source folder and subfolders
for subdir, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".png"):  # Check for PNG files (or adjust extension if needed)
            image_path = os.path.join(subdir, file)

            # Create corresponding subfolder in destination folder
            relative_path = os.path.relpath(subdir, source_folder)  # Get relative path
            destination_subfolder = os.path.join(destination_folder, relative_path)
            os.makedirs(destination_subfolder, exist_ok=True)  # Create subfolder if it doesn't exist

            # Get the save path for the processed image
            save_path = os.path.join(destination_subfolder, file)

            # Process and save the image
            process_image(image_path, save_path)

            print(f"Processed and saved: {save_path}")
