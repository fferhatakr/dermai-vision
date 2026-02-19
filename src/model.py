import torch.nn as nn
import torch
import torchvision.models as models
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

class SkinCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer =nn.Sequential(
        nn.Flatten(),
        nn.Linear(150528,224),
        nn.ReLU(),
        nn.Linear(224,7)
    )
    def forward(self, x):
       x = self.layer(x)
       return x
    
num_classes = 7
class SkinCancerModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(50176,128)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        
        x = self.flatten(x)

       
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        
        return x




class SkinCancerResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.resnet18_model = models.resnet18(weights='IMAGENET1K_V1')
        

        for param in self.resnet18_model.parameters():
            param.requires_grad = True

        original_fc_layer = self.resnet18_model.fc
        num_features = original_fc_layer.in_features
        num_classes = 7
        new_fc_layer = nn.Linear(in_features=num_features, out_features=num_classes)
        self.resnet18_model.fc = new_fc_layer

    def forward(self, x):
        return self.resnet18_model(x)
    


