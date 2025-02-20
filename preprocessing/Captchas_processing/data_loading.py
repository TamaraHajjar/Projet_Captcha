import cv2
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, Subset

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

def load_CAPTCHAS_DatasetSet(dataset_path):
    data_images=[]
    data_labels=[]

    labels_dict = get_dict_labels(dataset_path)

    for classes in os.listdir(dataset_path): # classes takes these values: airplane, automobile...
        
        for img in os.listdir(os.path.join(dataset_path,classes)): # for each img in airplane class

            img_path= os.path.join(dataset_path,classes,img) # construction du path vers une image: 'C:.../airplane/airplane_1.png'/

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data_images.append(image)
            data_labels.append(labels_dict[classes])

    data_labels=torch.LongTensor(data_labels)
    #data_images=torch.FloatTensor(data_images)
    data_images = torch.tensor(np.array(data_images), dtype=torch.float32)

    # normalizing the dataset
    data_images=data_images/255

    # shuffeling the training dataset
    num_samples = data_images.size(0)
    indices = torch.randperm(num_samples)
    data_images = data_images[indices]
    data_labels = data_labels[indices]

    data_images=data_images.permute(0,3,1,2)  # CNN expects input in the format (batch_size, channels, height, width). 

    return data_images, data_labels
#-----------------------------------------------------------------------------------

def LoadDataset(dataset_path):
    # Load dataset from images
    x_data, y_data = load_CAPTCHAS_DatasetSet(dataset_path)

    # Apply transformations: Resize, Rotation, ToTensor, Normalize
    transform_data = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomRotation(144),  # Any multiple of 12
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Create dataset
    full_dataset = CustomDataset(x_data, y_data, transform=transform_data)

    # Define split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Compute split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Ensure total matches

    # Split dataset
    train_subset, val_subset, test_subset = random_split(full_dataset, [train_size, val_size, test_size])

    # Convert to SubsetToDataset for easy access to data & labels
    train_dataset = SubsetToDataset(train_subset)
    val_dataset = SubsetToDataset(val_subset)
    test_dataset = SubsetToDataset(test_subset)

    return train_dataset, val_dataset, test_dataset
#-----------------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        return image, label
#-----------------------------------------------------------------------------------
#   
class SubsetToDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        
        # Collect and store the data and labels corresponding to the subset indices
        self.train_data, self.data_labels = self._collect_data_and_labels()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]
    
    def _collect_data_and_labels(self):
        datas = []
        lbls = []
        for idx in self.indices:
            image, label = self.dataset[idx]
            datas.append(image)
            lbls.append(label)
        return torch.stack(datas), torch.tensor(lbls)

