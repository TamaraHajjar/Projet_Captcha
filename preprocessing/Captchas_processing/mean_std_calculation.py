# --------------------------- CODE BIEN MAIS NON UTILISÉ ---------------------------
'''
J'avais utilisé ce code pour calculer le mean et std des images des captchas que j'ai
transformés en grayscale


'''
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from load_data import LoadDataset

# >-------------------------------------------------------<
#ICI J'AI CALCULER LE MEAN ET STD DES CAPTCHAS EN GRAYSCALE
# >-------------------------------------------------------<

# Compute mean and std for grayscale
def compute_mean_std(dataset):
    mean = 0.
    std = 0.
    total_images = len(dataset)
    
    # Loop through the dataset to accumulate pixel values
    for image in dataset:
        mean += image.mean()  # Compute mean for grayscale (only 1 channel)
        std += image.std()  # Compute std for grayscale (only 1 channel)

    # Average the computed mean and std
    mean /= total_images
    std /= total_images
    
    return mean, std

# Load dataset and compute mean/std
dataset = LoadDataset('C:/Users/MC/Desktop/PFE S5/Code/segmented_captchas/')
mean, std = compute_mean_std(dataset)
print(f"Mean: {mean.item()}, Std: {std.item()}")

#Mean: 0.3511512577533722, Std: 0.2554815113544464
#Mean: 0.35115125, Std: 0.25548151