import torch
import torchvision
from torchvision import transforms

def get_data_loaders(data_path, batch_size=32):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20), 
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)

    ])

   
    full_dataset_train = torchvision.datasets.ImageFolder(data_path, transform=train_transforms)
    full_dataset_val = torchvision.datasets.ImageFolder(data_path, transform=val_transforms)

    
    total_data = len(full_dataset_train)
    train_size = int(total_data * 0.85)
    
    indices = torch.randperm(total_data).tolist()
    train_set = torch.utils.data.Subset(full_dataset_train, indices[:train_size])
    val_set = torch.utils.data.Subset(full_dataset_val, indices[train_size:])

    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    
    print(f"Total Data: {total_data}")
    print(f"Train Set: {len(train_set)} image")
    print(f"Test Set: {len(val_set)} image")
    
    return train_loader, val_loader