# --------------------------- CODE BIEN MAIS NON UTILISÉ ---------------------------
'''
J'avais utilisé ce code pour calculer le mean et std des images des captchas que j'ai
transformés en grayscale

'''
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from load_data import LoadDataset

# Compute mean and std for RGB images
def compute_mean_std(dataset):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = len(dataset)
    
    # Loop through dataset to accumulate pixel values per channel
    for image in dataset:
        if isinstance(image, tuple):  # If dataset returns (image, label), extract only image
            image = image[0]
        
        mean += torch.tensor([image[:, :, i].mean() for i in range(3)])
        std += torch.tensor([image[:, :, i].std() for i in range(3)])
    
    # Compute average for each channel
    mean /= total_images
    std /= total_images
    
    return mean, std

# Load dataset and compute mean/std
dataset = LoadDataset('C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/Train_Captchas_UM_augmented_with_fct')
mean, std = compute_mean_std(dataset)
print(f"Mean: {mean.tolist()}, Std: {std.tolist()}")

#Mean_gray: 0.3511512577533722, Std: 0.2554815113544464
#Mean_gray: 0.35115125, Std: 0.25548151