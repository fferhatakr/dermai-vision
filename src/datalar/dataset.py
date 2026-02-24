import torch
import torchvision
from torchvision import transforms


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