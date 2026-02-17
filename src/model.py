import torch.nn as nn
import torch

class SkinCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.katman =nn.Sequential(
        nn.Flatten(),
        nn.Linear(150528,224),
        nn.ReLU(),
        nn.Linear(224,7)
    )
    def forward(self, x):
       x = self.katman(x)
       return x
    

