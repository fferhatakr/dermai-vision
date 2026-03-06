import torch
import torch.onnx
import os
import glob
import sys
sys.path.append(os.getcwd())
from src.engine.train_class_v2 import UltimateDermatolog as ModelClass



def convert_to_onnx():
    
    checkpoint_path = "models/production/classifier_v3_best.ckpt"
    onnx_path = "models/production/classifier_v3.onnx"

    try:
        dummy_weights = torch.ones(7).float()
        model = ModelClass.load_from_checkpoint(
            checkpoint_path,
            class_weights=dummy_weights
            
        )
        model.eval()
        model.to('cpu')

    except Exception as e:
        print("\n" + "!"*30)
        print(f" Detail: {e}")
        import traceback
        traceback.print_exc()
        return
    

    dummy_input = torch.rand(1,3,224,224).to('cpu')

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input_image'],
            output_names = ['logits'],
            dynamic_axes={
                'input_image': {0:'batch_size'},
                'logits': {0:'batch_size'}
            }
        )
    except Exception as e:
        print("\n" + "!"*30)
        print(f"Export error detail: {e}")
        import traceback
        traceback.print_exc()
        

    if os.path.exists(onnx_path):
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"Model Path:{onnx_path}")
        print(f" Filze Size: {file_size:.2f} MB")
        print(f"ONNX Runtime is used in src/api/main.py")
    else:
        print("Error:The file could not be created.")
if __name__ == "__main__":
    convert_to_onnx()