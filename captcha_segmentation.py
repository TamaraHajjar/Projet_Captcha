# --------------------------- CODE BIEN ---------------------------
'''
1ère étape:
Ce fichier est pour segmenter les captchas en 15 mini images
Résultat dans le folder segmented_captchas

'''
import cv2
import numpy as np
import os
import glob
from pathlib import Path

def extract_circles(image_path):
    #print(image_path)
    # Load the image --> it will be loaded as a numpy array in BGR format (H, W, C)
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    # Extract the original filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Define the dimensions of the circles and offsets
    circle_width, circle_height = 64, 64
    
    offset_x, offset_y = 3, 3
    num_rows, num_cols = 3, 5

    count = 0
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the top-left corner of each circle
            x1 = offset_x + col * (circle_width+1)
            y1 = offset_y + row * (circle_height+1)
            x2 = x1 + circle_width
            y2 = y1 + circle_height

            # Extract each individual circle
            circle_image = image[y1:y2, x1:x2]

            # Save each extracted circle with the original filename prefix
            count += 1
            if not os.path.exists("test_seg"):
                os.makedirs("test_seg")  # Create the directory if it doesn't exist
            cv2.imwrite(f"test_seg/{filename}_circle_{count}.png", circle_image)
            
            # Draw the circle on the original image (optional)
            #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 128, 255), 1)

if __name__ == "__main__":
    files = Path("C:/Users/MC/Desktop/PFE S5/Code/data/abacus_captcha_imgs").glob("*.png")

    for fpath in files:
        extract_circles(str(fpath))  # Convert to string if needed


        