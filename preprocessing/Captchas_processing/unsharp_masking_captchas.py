# --------------------------- CODE BIEN ---------------------------
'''
3ème étape
Ce code est pour appliquer le unsharp mask sur toutes les images segmentées des captchas
Résultat dans le folder Train_Captchas_unsharp_masking
Type des I/O: Reads images using OpenCV, applies transformations, and saves using Matplotlib's imsave().
'''
import numpy as np
from PIL import Image
from skimage.filters import unsharp_mask
import os
from glob import glob
from torchvision import transforms

class UnsharpMaskTransform:
    def __init__(self, radius, amount):
        self.radius = radius
        self.amount = amount

    def __call__(self, img): 
        img = np.array(img)
        img_sharp = unsharp_mask(img, self.radius, self.amount)
        
        # Convert to uint8 before converting to image
        img_sharp = np.uint8(np.clip(img_sharp * 255, 0, 255))
        
        return Image.fromarray(img_sharp)

transform_unsharp_mask = transforms.Compose([
    UnsharpMaskTransform(radius=2, amount=1),
    transforms.ToTensor(),
])

# Source and destination folders
#source_folder = "C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/Train_Captchas/"
source_folder = "C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/one_img_seg/median_filtering/"
#destination_folder = "C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/Train_Captchas_unsharp_masking_test/"
destination_folder = "C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/unsharp_masking/"


# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Dictionary to count images per subfolder
folder_image_count = {}

# Loop through the source folder and subfolders
for subdir, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".png"):  # Check for PNG files
            image_path = os.path.join(subdir, file)
            
            # Open the image
            image = Image.open(image_path)

            # Apply the transformation
            transformed_image = transform_unsharp_mask(image)
            
            # Convert tensor back to PIL image
            transformed_image_pil = transforms.ToPILImage()(transformed_image)

            # Get the relative subfolder name
            subfolder = os.path.basename(subdir)
            
            # Count images per subfolder
            if subfolder not in folder_image_count:
                folder_image_count[subfolder] = 1
            else:
                folder_image_count[subfolder] += 1
            
            # Generate the new file name
            new_filename = f"{subfolder}_{folder_image_count[subfolder]}.png"
            
            # Create the corresponding subfolder in the destination folder
            destination_subfolder = os.path.join(destination_folder, subfolder)
            os.makedirs(destination_subfolder, exist_ok=True)  # Create subfolder if it doesn't exist

            # Generate the save path for the transformed image
            save_path = os.path.join(destination_subfolder, new_filename)

            # Save the transformed image
            transformed_image_pil.save(save_path)

            print(f"Processed and saved: {save_path}")
