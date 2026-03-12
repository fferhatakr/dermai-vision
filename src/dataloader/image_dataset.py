
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader ,WeightedRandomSampler
import numpy as np



mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5), 
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(), # Zorunlu: Erasing'den önce
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    transforms.Normalize(mean=mean, std=std)
])

val_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])



def get_data_loaders(train_path,val_path, batch_size=16):
    

    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=val_transforms)

    targets = train_dataset.targets
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[targets]
    
   
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

  
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True 
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader