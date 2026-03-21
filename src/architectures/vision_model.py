import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)


class DermaScanModel(nn.Module):
    """Legacy MobileNetV3 model — kept for backward compatibility."""
    def __init__(self):
        super().__init__()
        import torchvision
        pretrained = torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        self.backbone = pretrained.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1024, 7)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class DermaScanModelV3(nn.Module):
    """EfficientNet-B3 backbone — current production model."""
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


class DermaScanModelV4(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(DermaScanModelV4, self).__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        for param in self.backbone.parameters():
            param.requires_grad = True

        in_features = self.backbone.classifier[2].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),       
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),                        
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True