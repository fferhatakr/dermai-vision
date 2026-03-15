import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from src.api.models import DEVICE

def apply_tta(image):
    base_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tta_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images = [base_transform(image)]
    for _ in range(4): 
        images.append(tta_transform(image))
    return torch.stack(images).to(DEVICE)

def apply_vignette(image_pil, sigma=180):
    img_cv = np.array(image_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    rows, cols = img_cv.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, sigma)
    kernel_y = cv2.getGaussianKernel(rows, sigma)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask.astype(np.float32) / mask.max()
    mask_3ch = np.dstack([mask] * 3)
    vignette_img = (img_cv * mask_3ch).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(vignette_img, cv2.COLOR_BGR2RGB))