import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from config.settings import NUM_CLASSES, DEVICE

# COMMENTER CES 2 FONCTIONS LORSQUE JE TRAVAILLE AVEC WEIGHTS=NONE
def RESNET18Model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def selectModelandTrainingParameters():
    model = RESNET18Model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print('Model used: ', str(RESNET18Model))

