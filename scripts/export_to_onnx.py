import torch
import torch.onnx
import os
import glob
import sys
sys.path.append(os.getcwd())
from src.training.trainer_core import TripletLightning



def convert_to_onnx():
    print("We are checking the file path and trying to find the most recent file.")
    checkpoints = glob.glob("models/*.ckpt") + glob.glob("lightning_logs/*/checkpoints/*.ckpt")
    if not checkpoints:
        print("Error: No .ckpt files were found!")
        return
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    output_onnx_path = "models/derma_vision_large_v1.onnx"

    try:
        model = TripletLightning.load_from_checkpoint(latest_ckpt)
        model.eval()
        model.to('cpu')
    except Exception as e:
        print(f"Error:An error occurred while loading the model")
        return
    

    fake_data = torch.rand(1,3,224,224,requires_grad=True).to('cpu')

    try:
        torch.onnx.export(
            model,
            fake_data,
            output_onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input_image'],
            output_names = ['embedding'],
            dynamic_axes={
                'input_image': {0:'batch_size'},
                'embedding': {0:'batch_size'}
            }
        )
    except Exception as e:
        print("Error Export")

    if os.path.exists(output_onnx_path):
        file_size = os.path.getsize(output_onnx_path) / (1024 * 1024)
        print(f"Model Path:{output_onnx_path}")
        print(f" Filze Size: {file_size:.2f} MB")
        print(f"ONNX Runtime is used in src/api/main.py")
    else:
        print("Error:The file could not be created.")
if __name__ == "__main__":
    convert_to_onnx