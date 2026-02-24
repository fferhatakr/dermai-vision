#Import the necessary libraries
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from models.model import SkinCancerMobileNet



num_classes = 7

class DermatologLightning(pl.LightningModule):
    def __init__(self, class_weights): 
        super().__init__()

        #We've Launched Our Model
        self.model=SkinCancerMobileNet() 

        #We defined the loss function and provided its weights as parameters.
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        #We automatically calculate the success rate
        self.accuracy =Accuracy(task="multiclass",num_classes=num_classes)

    def forward(self,x):

        return self.model(x)
    

    def training_step(self,batch,batc_idx):
        
        #We are adding the images and tags to the batch
        images,labels = batch

        #Predictions, Loss Value and Accuracy Calculated
        predictions = self(images)
        loss = self.criterion(predictions,labels)
        acc =self.accuracy(predictions,labels)

        #These are the ready-made codes that should be there and that we should see in the terminal.
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        #Return the loss value to the system so that it can perform the training.
        return loss
    
    def validation_step(self,batch,batch_idx):
        #It continues with the same logic as the training_step steps.
        images,labels = batch
        predictions= self(images)
        #Loss Value and Accuracy Calculated
        loss = self.criterion(predictions,labels)
        acc = self.accuracy(predictions,labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    #We define the optimizer and the learning rate
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-5)
        return optimizer