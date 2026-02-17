import torch
import torchvision
from torchvision import transforms

def veri_yukleyicileri_getir(veri_yolu, batch_size=32):
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20), 
        transforms.ColorJitter(),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

   
    full_dataset_train = torchvision.datasets.ImageFolder(veri_yolu, transform=train_transforms)
    full_dataset_val = torchvision.datasets.ImageFolder(veri_yolu, transform=val_transforms)

    
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

    
    print(f"Toplam Veri: {total_data}")
    print(f"Eğitim Seti: {len(train_set)} resim")
    print(f"Test Seti: {len(val_set)} resim")
    
    return train_loader, val_loader