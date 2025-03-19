import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from config.settings import NUM_CLASSES, DEVICE

def VGG16Model():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1, progress=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=NUM_CLASSES)
    return model

def selectModelandTrainingParameters():
    model = VGG16Model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print('Model used: ', str(VGG16Model))
    return model, criterion, optimizer
    