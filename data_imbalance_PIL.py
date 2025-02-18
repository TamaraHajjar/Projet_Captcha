# --------------------------- CODE BIEN ---------------------------
'''
Ce code est pour gérer le data imbalance avec des images PIL.
It uses transform.RandomRotation.
Cependant, les imgs obtenues ont un peu une qualité dégradée.(results are save in Train_Captchas_unsharp_masking_copy_1)
'''

import os
import random
from glob import glob
from PIL import Image
from torchvision import transforms

# Define rotation angles: multiples of 12 up to 348
ROTATION_ANGLES = [12 * i for i in range(1, 29)]  # [12, 24, 36, ..., 348]

def rotate_image_torchvision(image_path, angles):
    """Loads an image, applies rotation using torchvision.transforms."""
    img = Image.open(image_path)  # Open with PIL
    
    rotated_images = []
    for angle in angles:
        transform = transforms.RandomRotation([angle, angle])  # Fixed rotation
        rotated_img = transform(img)  # Apply rotation
        rotated_images.append((rotated_img, angle))
    
    return rotated_images

def augment_images_torchvision(subfolder_path, subfolder_name, target_count=200):
    """Augment images to reach the target count using rotation."""
    image_paths = glob(os.path.join(subfolder_path, "*.png"))
    image_count = len(image_paths)

    if image_count >= target_count:
        print(f"{subfolder_path} already has {image_count} images, skipping augmentation.")
        return

    images_needed = target_count - image_count
    next_number = max(
        [int(os.path.basename(img).split("_")[-1].split(".")[0]) for img in image_paths], default=0
    ) + 1

    # First, go through each image and generate **3 unique rotated versions**
    for img_path in image_paths:
        if images_needed <= 0:
            break
        
        selected_angles = random.sample(ROTATION_ANGLES, 3)  # Pick 3 unique angles
        rotated_images = rotate_image_torchvision(img_path, selected_angles)

        for rotated_img, angle in rotated_images:
            if images_needed <= 0:
                break
            new_filename = f"{subfolder_name}_{next_number}.png"
            new_path = os.path.join(subfolder_path, new_filename)
            rotated_img.save(new_path)  # Save rotated image
            next_number += 1
            images_needed -= 1

    # If we still need more images, randomly select existing ones and apply rotation
    while images_needed > 0:
        selected_image = random.choice(image_paths)
        selected_angle = random.choice(ROTATION_ANGLES)  # Pick a random rotation

        rotated_img = rotate_image_torchvision(selected_image, [selected_angle])[0][0]
        new_filename = f"{subfolder_name}_{next_number}.png"
        new_path = os.path.join(subfolder_path, new_filename)
        rotated_img.save(new_path)  # Save rotated image
        next_number += 1
        images_needed -= 1

    print(f"Augmentation completed for {subfolder_path}.")

# Example: Process each subfolder
source_folder = "C:/Users/MC/Desktop/PFE S5/Code/data/Train_Captchas_unsharp_masking_copy_1/"
subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
for subfolder in subfolders:
    subfolder_name = os.path.basename(subfolder)
    augment_images_torchvision(subfolder, subfolder_name)
