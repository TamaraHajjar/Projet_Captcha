'''
fichier pour comparer le contenu de 2 folders
'''
import os
from collections import defaultdict

def get_all_images(folder):
    image_files = set()
    image_paths = defaultdict(list)
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith('.png'):
                image_files.add(f)
                image_paths[f].append(os.path.join(dirpath, f))
    return image_files, image_paths

def compare_folders(folder1, folder2):
    images1 = {f for f in os.listdir(folder1) if f.endswith('.png')}
    images2, image_paths = get_all_images(folder2)
    
    missing_images = images1 - images2
    
    return missing_images, image_paths

def find_duplicates(image_paths):
    duplicates = {k: v for k, v in image_paths.items() if len(v) > 1}
    return duplicates

if __name__ == "__main__":
    folder1 = "C:/Users/MC/Desktop/PFE S5/Code/data/segmented_captchas_median_filtering"  # Replace with actual path
    folder2 = "C:/Users/MC/Desktop/PFE S5/Code/data/test"  # Replace with actual path
    
    missing, image_paths = compare_folders(folder1, folder2)
    
    if missing:
        print("Missing images in folder2:")
        for img in missing:
            print(img)
    else:
        print("No images are missing.")
    
    duplicates = find_duplicates(image_paths)
    if duplicates:
        print("Duplicate images found:")
        for img, paths in duplicates.items():
            print(f"{img} is found in:")
            for path in paths:
                print(f"  {path}")
    else:
        print("No duplicate images found.")
