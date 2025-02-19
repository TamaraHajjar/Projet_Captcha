# --------------------------- CODE BIEN ---------------------------
'''
Ce code marche très bien avec des images OpenCV.
Une autre version sera pour le format d'images PIL
'''
import os
import random
import shutil
import cv2
import numpy as np
from glob import glob
import re

# Source folder
#source_folder = "C:/Users/MC/Desktop/PFE S5/Code/data/Train_Captchas_unsharp_masking_copy/"
source_folder = "C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/Train_Captchas_Copy/"

ROTATION_ANGLES = [12 * i for i in range(1, 29)]  # [12, 24, 36, ..., 348]

def get_next_image_number(image_paths, subfolder_name):
    numbers = []
    pattern = re.compile(rf"{re.escape(subfolder_name)}_(\d+)\.png")
    
    for img_path in image_paths:
        match = pattern.search(os.path.basename(img_path))
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) if numbers else 0


def rotate_image(image_path, angle):
    """Loads an image and applies a specific rotation."""
    img = cv2.imread(image_path)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def augment_images_to_target(subfolder_path, subfolder_name, target_count=200):
    # Get all image paths in the subfolder
    image_paths = glob(os.path.join(subfolder_path, "*.png"))
    image_count = len(image_paths)
    
    # Check if augmentation is needed
    if image_count >= target_count:
        print(f"{subfolder_path} already has {image_count} images, skipping augmentation.")
        return
    
    # Calculate how many images are needed
    images_needed = target_count - image_count
    next_number = get_next_image_number(image_paths, subfolder_name) + 1
    
    print(f"{subfolder_path} has {image_count} images, adding {images_needed} more.")
    
    #rotation_index = 0
    for img_path in image_paths:
        random_angle = 0
        for _ in range(3):  # Generate 3 rotated versions per image
            if images_needed <= 0:
                break
            random_1 = random.choice(ROTATION_ANGLES) # choose a random angle among the 29 angles in the ROTATION_ANGLES list
            while random_angle == random_1:
                random_1 = random.choice(ROTATION_ANGLES)
            random_angle = random_1
            #angle = ROTATION_ANGLES[rotation_index % len(ROTATION_ANGLES)]
            rotated_img = rotate_image(img_path, random_angle)
            new_filename = f"{subfolder_name}_{next_number}.png"
            new_path = os.path.join(subfolder_path, new_filename)
            cv2.imwrite(new_path, rotated_img)
            next_number += 1
            images_needed -= 1
            #rotation_index += 1
        if images_needed <= 0:
            break
    
    # If we still need more images, randomly select images and apply rotation
    while images_needed > 0:
        selected_image = random.choice(image_paths)
        random_angle = 0
        random_1 = random.choice(ROTATION_ANGLES) # choose a random angle among the 29 angles in the ROTATION_ANGLES list
        while random_angle == random_1:
            random_1 = random.choice(ROTATION_ANGLES)
        random_angle = random_1
        #angle = ROTATION_ANGLES[rotation_index % len(ROTATION_ANGLES)]
        rotated_img = rotate_image(selected_image, random_angle)
        new_filename = f"{subfolder_name}_{next_number}.png"
        new_path = os.path.join(subfolder_path, new_filename)
        cv2.imwrite(new_path, rotated_img)
        next_number += 1
        images_needed -= 1
        #rotation_index += 1
    
    print(f"Augmentation completed for {subfolder_path}.")

# Get all subfolders
subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

# Process each subfolder
for subfolder in subfolders:
    subfolder_name = os.path.basename(subfolder)
    augment_images_to_target(subfolder, subfolder_name)
