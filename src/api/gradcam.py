import torch
import numpy as np
import cv2
import base64
from torchvision import transforms
from src.api.models import DEVICE

def generate_heatmap(lightning_model, processed_image, original_image):
    try:
        single_input = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(processed_image).unsqueeze(0).to(DEVICE)
        hm = lightning_model.generate_gradcam(single_input)
        if isinstance(hm, torch.Tensor): 
            hm = hm.squeeze().detach().cpu().numpy()
        hm = cv2.resize(hm, (original_image.size[0], original_image.size[1]))
        hm_uint8 = np.uint8(255 * hm)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        _, buf = cv2.imencode(".png", hm_color)
        return base64.b64encode(buf).decode("utf-8")
    except:
        return ""