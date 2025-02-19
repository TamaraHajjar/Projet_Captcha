import cv2
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

dataset_path = "C:/Users/MC/Desktop/PFE S5/data_in_folder_Code/data/Train_Captchas_UM_augmented_with_fct"

#-----------------------------------------------------------------------------------
def get_dict_labels(dataset_path):
    '''
    This method aims to convert class names to numeric labels (for training).
    params: path to the classes
    output: {'airplane': 0, 'automobile': 1, 'banknote': 2, 'bird': 3, 'boxing': 4, 'burgerking': 5, 'bus': 6, 'cat': 7, 'cloud': 8, 'coin': 9, 'crosswalk': 10, 'deer': 11, 'dog': 12, 'firehydrant': 13, 'frog': 14, 'gold': 15, 'horse': 16, 'pyramids': 17, 'road': 18, 'ship': 19, 'stair': 20, 'trafficlight': 21, 'tree': 22}
    '''
    classes_dict = {}
    i = 0
    Classes = os.listdir(dataset_path)
    Classes.sort() # Sort the classes in alphabetical order

    for classe in Classes:  
        classes_dict[classe] = i  # Assign class name as value with index as key
        i += 1
    return classes_dict
#-----------------------------------------------------------------------------------

def load_CAPTCHAS_TrainingSet(dataset_path):
    train_images=[]
    train_labels=[]

    labels_dict = get_dict_labels(dataset_path)

    for classes in os.listdir(dataset_path): # classes takes these values: airplane, automobile...
        
        for img in os.listdir(os.path.join(dataset_path,classes)): # for each img in airplane class

            img_path= os.path.join(dataset_path,classes,img) # construction du path vers une image: 'C:.../airplane/airplane_1.png'/

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            train_images.append(image)
            train_labels.append(labels_dict[classes])

    train_labels=torch.LongTensor(train_labels)
    train_images=torch.FloatTensor(train_images)

    # normalizing the dataset
    train_images=train_images/255

    # shuffeling the training dataset
    num_samples = train_images.size(0)
    indices = torch.randperm(num_samples)
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    train_images=train_images.permute(0,3,1,2)  # CNN expects input in the format (batch_size, channels, height, width). 

    return train_images, train_labels
#-----------------------------------------------------------------------------------

def LoadDataset(dataset_path):
    # Load dataset from images
    x_train, y_train = load_CAPTCHAS_TrainingSet(dataset_path)
    
    # Apply transformations: ToTensor, Normalize
    transform_train = transforms.Compose([
        #transforms.Grayscale(num_output_channels=1),  # Convert RGB to grayscale
        transforms.ToTensor(),  # Converts PIL Image to Tensor
    ])
    
    train_data = CustomDataset(x_train, transform=transform_train)
    
    return train_data # in train_data images are tensors
