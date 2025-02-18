# --------------------------- CODE BIEN ---------------------------
'''
3ème étape (probablement à ne pas utiliser)
Ce fichier permet d'appliquer CLAHE sur les images segmentées 
(ici j'ai essayé sur une image:C:\Users\MC\Desktop\PFE S5\Code\data\clahe on captcha img)

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# # Load the image --> it will be loaded as a numpy array in BGR format (H, W, C)
img_path = glob("C:/Users/MC/Desktop/PFE S5/Code/data/segmented_captchas_median_filtering/0b75a677690bb65d8f6d0a70507e8a1e_checkbox_image_circle_2.png")

for image in img_path:
    img = cv2.imread(image)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    L, A, B = cv2.split(lab)

    # Apply CLAHE only on the L channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)

    # Merge back the LAB channels
    lab_clahe = cv2.merge((L_clahe, A, B))

    # Convert back to BGR color space
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    # Convert back to RGB for color convinience
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plt.imshow(img, cmap='gray')
    # plt.title(f"image ")
    # plt.axis('off')  # Hide the axes
    # plt.show()

    # plt.imshow(colored, cmap='gray')
    # plt.title(f"CLAHE image ")
    # plt.axis('off')  # Hide the axes
    # plt.show()

    # Create a figure and two subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # (rows, columns)

    # Show Image 1
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")  # Hide axes

    # Show Image 2
    ax[1].imshow(enhanced)
    ax[1].set_title("Processed Image clip_limit = 2")
    ax[1].axis("off")  # Hide axes

    # Show the plot
    plt.show()
