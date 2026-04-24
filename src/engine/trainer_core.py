import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score, Accuracy
import numpy as np

num_classes = 8


class DermatologLightning(pl.LightningModule):
    def __init__(self, class_weights=None, backbone="efficientnet_b3", lr=1e-4, max_epochs=25):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights'])

        self.lr = lr
        self.max_epochs = max_epochs

        
        if backbone == "efficientnet_b3":
            from src.architectures.vision_model import DermaScanModelV3
            self.model = DermaScanModelV3(num_classes=num_classes)
        elif backbone == "convnext_tiny":
            from src.architectures.vision_model import DermaScanModelV4
            self.model = DermaScanModelV4(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

        self.activations = None
        self.gradients = None

        self._register_hooks(backbone)

    def _register_hooks(self, backbone):
    
        try:
            if backbone == "efficientnet_b3":
                self.model.backbone.features[-1].register_forward_hook(self.save_activations)
                self.model.backbone.features[-1].register_full_backward_hook(self.backward_gradients)
            elif backbone == "convnext_tiny":
                self.model.backbone.features[-1].register_forward_hook(self.save_activations)
                self.model.backbone.features[-1].register_full_backward_hook(self.backward_gradients)
        except Exception as e:
            print(f"GradCAM hook warning: {e}")

    def save_activations(self, module, input, output):
        self.activations = output

    def backward_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.model(x)

    """
    Focal Loss calculates. Unlike standard cross-entropy,
    it reduces the weight of easy examples that the model is already certain about using (1-pt)².
    This allows the model to focus more on rare and difficult classes.
    Additionally, class_weights corrects for class imbalance."""
    def compute_loss(self, logits, labels):
        ce_loss = F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** 2.0) * ce_loss
        return focal_loss.mean()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.compute_loss(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.compute_loss(logits, labels)
        preds = torch.argmax(logits, dim=1)

        acc = self.val_accuracy(preds, labels)
        f1_scores = self.val_f1_per_class(preds, labels)

        self.log('val_f1_MEL', f1_scores[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=3
        )
        
        """
        We start by taking big steps at the beginning to help them learn better
        and then we slow down our pace to ensure more effective learning"""
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs-3 , eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[3]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def generate_gradcam(self, input_image, target_class=None):
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

        weights = derivative.mean(dim=[2, 3], keepdim=True)
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
        normal_map = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return normal_map