import torch
import numpy as np
import cv2
import base64
import io
from PIL import Image
from torchvision import transforms
from src.api.models import DEVICE

def generate_heatmap(lightning_model, processed_image, original_image):
    """
    EfficientNet-B3 tabanlı Grad-CAM ısı haritası üretir ve orijinal resmin üzerine bindirir.
    """
    try:
        # 1. Preprocessing (Eğitim parametreleri ile birebir uyumlu)
        single_input = transforms.Compose([
            transforms.Resize((300, 300)), # EfficientNet-B3 standart giriş boyutu
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(processed_image).unsqueeze(0).to(DEVICE)

        # 2. Grad-CAM Üretimi
        # lightning_model içindeki generate_gradcam fonksiyonunu çağırır
        hm = lightning_model.generate_gradcam(single_input)
        
        if isinstance(hm, torch.Tensor): 
            hm = hm.squeeze().detach().cpu().numpy()

        # 3. Isı Haritasını Orijinal Resim Boyutuna Getir
        hm = cv2.resize(hm, (original_image.size[0], original_image.size[1]))
        hm_uint8 = np.uint8(255 * hm)
        
        # JET renk paleti uygula (Sıcak bölgeler kırmızı, soğuklar mavi)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

        # 4. Isı Haritasını Resmin Üzerine Bindir (Overlay)
        # Orijinal PIL resmini OpenCV formatına çevir
        orig_cv = np.array(original_image.convert('RGB'))
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_RGB2BGR)

        # %60 orijinal resim, %40 ısı haritası karışımı (Şeffaflık)
        overlay_img = cv2.addWeighted(orig_cv, 0.6, hm_color, 0.4, 0)

        # 5. Base64 Formatına Çevir ve Gönder
        _, buf = cv2.imencode(".png", overlay_img)
        return base64.b64encode(buf).decode("utf-8")
        
    except Exception as e:
        print(f"Grad-CAM Error: {str(e)}")
        return ""