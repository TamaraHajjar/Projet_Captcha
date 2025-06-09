'''
1ère étape:
Ce fichier est pour segmenter les captchas en 15 mini images
Les 15 images ainsi obtenues sont ensuite aussi rognées sous forme de cercles

Cette étape est nécessaire pour s'affranchir des larges zones noires inutiles dans l'img
et que le modèle de deep learning puisse apprendre correctement
Version finale de la segmentation
'''

import cv2
import numpy as np
import os
import glob
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def extract_circles(image_path, output_folder):
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
    num_rows, num_cols = 3, 5  # Adjust as needed

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

            # Create directory if it doesn't exist
            output_dir = os.path.join(output_folder, "extracted_circles")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save each extracted circle with the original filename prefix
            count += 1
            circle_filename = f"{filename}_circle_{count}.png"
            cv2.imwrite(os.path.join(output_dir, circle_filename), circle_image)

            # Process the circle for luminance and pie slice
            process_image_for_luminance(os.path.join(output_dir, circle_filename))

def process_image_for_luminance(image_path):
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
    img_final.save(image_path)

def main():
    source_folder = "C:/Users/data/one_img_seg"
    output_folder = "C:/Users/one_img_seg"  # This is where the circles will be saved

    # Loop through the source folder and extract circles
    files = Path(source_folder).glob("*.png")
    for fpath in files:
        extract_circles(str(fpath), output_folder)

if __name__ == "__main__":
    main()
