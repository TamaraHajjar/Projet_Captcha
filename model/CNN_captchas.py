# -*- coding: utf-8 -*-

#To implement an object recognition CAPTCHA classifier using a Convolutional Neural Network (CNN) based on the VGG architecture in PyTorch, follow these steps:

#Import Necessary Libraries:
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from preprocessing.Captchas_processing.data_loading import LoadDataset
from model.vgg16 import selectModelandTrainingParameters
from config.settings import NUM_CLASSES, DEVICE, NUM_EPOCHS

import matplotlib.pyplot as plt
import torchvision.utils as vutils


#dataset_path = '/home/elhajjta/Projet/data/Train_Captchas_UM_CC_DataAug/'
dataset_path = 'C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/Train_Captchas_UM_CC_DataAug'
# Load dataset and split into train, validation, and test sets
train_dataset, val_dataset, test_dataset = LoadDataset(dataset_path)

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print('/nloading data done.../n')

# Get a batch of images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Display images
plt.figure(figsize=(10, 5))
plt.axis("off")
plt.title("Sample Training Images")
plt.imshow(vutils.make_grid(images[:8], nrow=4).permute(1, 2, 0))  # Show first 8 images
plt.show()


#Load Pre-trained VGG16 Model:
model, criterion, optimizer = selectModelandTrainingParameters()

# Geler tous les paramètres
for param in model.parameters():
    param.requires_grad = False

# Unfreeze Block 4 
for i, layer in enumerate(model.features):
    if isinstance(layer, nn.Conv2d) and 24 <= i <= 28:
        for param in layer.parameters():
            param.requires_grad = True

# Décongeler les paramètres de la dernière couche
for param in model.classifier.parameters():
    param.requires_grad = True

print('Start training...')

#Train the Model:
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    epoch_loss = train_loss / len(train_loader.dataset)
    epoch_acc = 100 * (train_correct / train_total)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}')

    #scheduler.step()

# Evaluate the Model on Validation Set
model.eval()
val_correct = 0
val_total = 0
val_loss = 0.0

with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc="Evaluating on Validation Set"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

val_acc = 100* (val_correct / val_total)
val_loss = val_loss / len(val_loader.dataset)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}')

# Final Test Evaluation
test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        #print(f'outputs: {outputs}')
        loss = criterion(outputs, labels)

        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        #print(f'predicted : {predicted}')
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * (test_correct / test_total)
test_loss = test_loss / len(test_loader.dataset)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}')
