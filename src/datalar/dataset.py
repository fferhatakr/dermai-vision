import torch
import torchvision
from torchvision import transforms
import random
from collections import defaultdict
from torch.utils.data import Dataset

#This function is used to load the dataset and split it into training and validation sets.
def get_data_loaders(data_path, batch_size=32):
    #The mean and standard deviation values are defined.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    #We refine the training set through specific stages to make it understandable for our model.
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20), 
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    #We are also running the verification set through most of the same processes.
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)

    ])

    #We load the dataset from the specified path and apply the transformations.
    full_dataset_train = torchvision.datasets.ImageFolder(data_path, transform=train_transforms)
    full_dataset_val = torchvision.datasets.ImageFolder(data_path, transform=val_transforms)

    #We split the dataset into training and validation sets.
    total_data = len(full_dataset_train)
    train_size = int(total_data * 0.85)
    
    #We shuffle the dataset and split it into training and validation sets.
    indices = torch.randperm(total_data).tolist()
    train_set = torch.utils.data.Subset(full_dataset_train, indices[:train_size])
    val_set = torch.utils.data.Subset(full_dataset_val, indices[train_size:])

    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size, 
        shuffle=True, #The reason it is “true” is that we want the model to see the data in different orders each time, just like shuffling a deck of cards.
        num_workers=4, #You can think of it as if we were employing four workers. The goal is to increase speed.
        pin_memory=True #We say “True” to speed up data transfer to the graphics card.
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,  #The reason it is “false” is that papers are not mixed up during the exam.
        num_workers=4, #You can think of it as if we were employing four workers. The goal is to increase speed.
        pin_memory=True #We say “True” to speed up data transfer to the graphics card.
    )

    
    print(f"Total Data: {total_data}")
    print(f"Train Set: {len(train_set)} image")
    print(f"Test Set: {len(val_set)} image")
    
    return train_loader, val_loader





class TripletDermaDataset(Dataset):
        def __init__(self,data_path):
            self.data = data_path
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
            self.index = torchvision.datasets.ImageFolder(self.data,transform=self.transform)
            
            self.label_to_indices=defaultdict(list) 

            for i,label in enumerate(self.index.targets):
                self.label_to_indices[label].append(i)
            
            
            self.all_classes = list(self.label_to_indices.keys())


        def __len__(self):
            
            return len(self.index)

        def __getitem__(self,idx):
            anchor_img , anchor_label = self.index[idx]
            positive_list = self.label_to_indices[anchor_label]
            positive_idx = random.choice(positive_list)
            positive_img,_ = self.index[positive_idx]

            negative_classes = self.all_classes.copy()
            negative_classes.remove(anchor_label)
            random_negative_class = random.choice(negative_classes)
            negative_list = self.label_to_indices[random_negative_class]

            negative_idx = random.choice(negative_list)
            negative_img,_ = self.index[negative_idx]

            return anchor_img,positive_img,negative_img


            