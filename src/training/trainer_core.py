
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
# from src.architectures.vision_model import SkinCancerMobileNet
import torch.nn.functional as F
from src.architectures.vision_model import DermaScanModelV3
from torchvision.ops import sigmoid_focal_loss
from torchmetrics import F1Score ,Accuracy
import numpy as np
num_classes = 8

class DermatologLightning(pl.LightningModule):
    def __init__(self, class_weights= None): 
        super().__init__()

        
        self.model=DermaScanModelV3(num_classes=num_classes) 
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        self.accuracy = Accuracy(task="multiclass", num_classes=8)
        self.activations = None
        self.gradients = None

        self.model.backbone.features[-2].register_forward_hook(self.save_activations)
        self.model.backbone.features[-2].register_full_backward_hook(self.backward_gradients)

        self.f1_per_class = F1Score(task="multiclass", num_classes=8, average=None)

    def save_activations(self, module, input, output):
        print("Activation shape:", output.shape)
        self.activations = output

    def backward_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self,x):

        return self.model(x)
    
    def compute_loss(self, logits, labels):
        
        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=0.1,
            reduction='none'
            )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** 2.0) * ce_loss
        
        return focal_loss.mean()
    
    def training_step(self,batch,batc_idx):
        
        images,labels = batch
        logits = self(images)
        loss = self.compute_loss(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self,batch,batch_idx):
        
        images,labels = batch
        logits = self(images)
        loss = self.compute_loss(logits,labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels)
        f1_scores = self.f1_per_class(preds, labels)

        self.log('val_f1_MEL', f1_scores[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    
    def generate_gradcam(self, input_image,target_class = None):
        
        if input_image.grad is not None:
            input_image.grad.zero_()
        
        
        logits = self.forward(input_image)

        if target_class is None:
            pred_idx = torch.argmax(logits, dim=1).item()
        else:
            pred_idx = torch.tensor(target_class)
        
        score = logits[:, pred_idx]
        
        
        self.zero_grad()
        score.backward(retain_graph=True)

        
        maps = self.activations 
        derivative = self.gradients  

        weights = derivative.mean(dim=[2,3], keepdim=True)
        multiplication_table = maps * weights

        
        single_map = multiplication_table.sum(dim=1, keepdim=True)
        positive_map = F.relu(single_map)
        cam = positive_map.squeeze().detach().cpu().numpy()
        h, w = cam.shape
        margin_h = int(h * 0.10) 
        margin_w = int(w * 0.10) 
        cam[:margin_h, :] = 0  
        cam[-margin_h:, :] = 0 
        cam[:, :margin_w] = 0  
        cam[:, -margin_w:] = 0 
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.power(cam, 2.0)
        max_val = positive_map.max()
        normal_map = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return normal_map
    
    
    def configure_optimizers(self):
       
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        
      
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    





class TripletLightning(pl.LightningModule):
    def __init__(self,learning_rate,margin_value): 
        super().__init__()
        self.learning_rate = learning_rate
        self.model=DermaScanModelV3() 
        self.criterion = nn.TripletMarginLoss(
            margin=margin_value,
            p=2
            )
        
        self.activations = None
        self.gradients = None

        self.model.backbone.register_forward_hook(self.save_activations)
        self.model.backbone.register_full_backward_hook(self.backward_gradients)


    def save_activations(self, module, input, output):
        print("Activation shape:", output.shape)
        self.activations = output

    def backward_gradients(self,module, grad_input, grad_output):

        self.gradients = grad_output[0]





    def forward(self,x):
        x = self.model.backbone(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x,1)


        return x
    

    def training_step(self,batch,batc_idx):
      
        anchor,positive,negative = batch
        anchor_embedding =self(anchor)
        positive_embedding =self(positive)
        negative_embedding = self(negative)

    
        loss = self.criterion(anchor_embedding,positive_embedding,negative_embedding)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)


        return loss
    
    def validation_step(self,batch,batch_idx):

        anchor,positive,negative = batch
        anchor_embedding =self.forward(anchor)
        positive_embedding =self.forward(positive)
        negative_embedding = self.forward(negative)

    
        loss = self.criterion(anchor_embedding,positive_embedding,negative_embedding)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)


        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        return optimizer
    

    def generate_gradcam(self, input_image):
        vector = self.forward(input_image) 
        total_energy = vector.sum() 
        total_energy.backward() 

        maps = self.activations 

        derivative = self.gradients  
        weights = derivative.mean(dim=[2,3], keepdim=True)
        multiplication_table = maps * weights
        
        single_map = multiplication_table.sum(dim=1, keepdim=True)
        positive_map = F.relu(single_map)
        normal_map = positive_map / positive_map.max()

    
        return normal_map