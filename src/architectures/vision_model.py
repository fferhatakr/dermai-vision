import torch.nn as nn
import torch
import torchvision.models as models
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights



    

"""
Some may wonder why we rejected DenseNet. Our aim is to release 
this project as a mobile application. We chose MobileNet because 
DenseNet could experience significant optimisation issues on certain devices.
"""
#Define the fifth version of the model.
class  DermaScanModel(nn.Module):
    def __init__(self): 
        super().__init__()


        pretrained = torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        self.backbone = pretrained.features 
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1024,7)
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    


#Most recently used class 07.03.2026
class DermaScanModelV3(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(DermaScanModelV3, self).__init__()
        
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        
        in_features = self.backbone.classifier[1].in_features
        
       
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4), 
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512), 
            nn.SiLU(), 
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
    
    
    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True