
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

"""
Certain steps are taken to prevent the model from overfitting or relying on rote learning 
during training. For example, 
rotating the image, turning half of the image black, experimenting with colors, etc.
"""
train_album = A.Compose([
    A.Resize(300, 300),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
    A.OneOf([
        A.CLAHE(clip_limit=2.0, p=1.0),           # Kontrast iyileştirme
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    ], p=0.5),
    A.OneOf([
        A.GaussNoise(noise_scale_range=(0.02, 0.1), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),  # Bulanıklık
    ], p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(10, 20), hole_width_range=(10, 20), fill=0, p=0.3),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

"""
Here, we conduct training as if the model were taking an actual exam. To give an example, 
the questions the model encounters during the exam must be clear and follow a standard format. For this reason, 
virtually no “Compose” operations are performed.
"""
val_album = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

"""
A general architecture that automatically generates tags from the folder structure.
Albumentations -> Numpy
"""
class AlbumentationsDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, album_transform=None, **kwargs):
        super().__init__(root, **kwargs)
        self.album_transform = album_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = np.array(image)  
        
        if self.album_transform:
            augmented = self.album_transform(image=image)
            image = augmented["image"]  
        
        return image, target
    
train_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    transforms.Normalize(mean=mean, std=std)
])

val_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])