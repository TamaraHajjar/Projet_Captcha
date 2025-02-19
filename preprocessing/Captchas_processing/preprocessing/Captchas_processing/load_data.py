# --------------------------- CODE BIEN ---------------------------
'''
Ce code est pour loader les images des captchas, appelé dans le fichier mean_std_calculation.py
'''
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def load_CAPTCHAS_TrainingSet(data_path):
    data = []
    image_path = os.listdir(data_path)
           
    for img in image_path:
            try:
                image = cv2.imread(data_path + img)
                # Convert BGR image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((32, 32))
                data.append(size_image)
            except AttributeError:
                print("Error loading image:", img)
    print("\n")    
    return data  # Return a list of PIL Image objects

def LoadDataset(path):
    # Load dataset from images
    x_train = load_CAPTCHAS_TrainingSet(path)
    
    # Apply transformations: ToTensor, Normalize
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert RGB to grayscale
        transforms.ToTensor(),  # Converts PIL Image to Tensor
    ])
    
    train_data = CustomDataset(x_train, transform=transform_train)
    
    return train_data # in train_data images are tensors

class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        # Apply the transformation (e.g., ToTensor, Normalize) if any
        if self.transform:
            image = self.transform(image)  # Apply transformations if any

        return image
