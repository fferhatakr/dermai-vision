"""
This script does the following:
1. Loads the trained .ckpt model
2. Converts the model to ONNX format
3. Compares PyTorch vs ONNX outputs (correctness test)
4. Performs speed comparison
"""
import torch
import numpy as np
import time
import onnxruntime as ort
import sys
import os
import hydra
from omegaconf import DictConfig

sys.path.append(os.getcwd())
from engine.trainer_core import DermatologLightning

DEVICE = torch.device("cpu")


def export_to_onnx(ckpt_path, onnx_path, image_size):
    print("Model Loading")

    model = DermatologLightning.load_from_checkpoint(
        ckpt_path, 
        strict=False
    )
    model.to(DEVICE)
    model.eval()


    dummy_input = torch.randn(1, 3, image_size,image_size).to(DEVICE)

    print("It is being exported to ONNX.")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print(f"Onnx saved: {onnx_path}")
    return model, dummy_input


def verify_outputs(pytorch_model, dummy_input,onnx_path):

    print("Verification in progress")

    with torch.no_grad():
        pytorh_output= pytorch_model(dummy_input).numpy()

        session = ort.InferenceSession(onnx_path ,providers=["CPUExecutionProvider"])
        onnx_output = session.run(["logits"], {"image": dummy_input.numpy()})[0]


        max_diff = np.max(np.abs(pytorh_output - onnx_output))
        print(f"Max differences: {max_diff}")

        if max_diff < 1e-4:
            print("Succesful")
        else:
            print("Error")

def benchmark_speed(pytorch_model , dummy_input , onnx_path,n_runs=100):

    with torch.no_grad():
        _ = pytorch_model(dummy_input)
        start = time.time()
        for _ in range(n_runs):
            pytorch_model(dummy_input)
        pytorch_time = (time.time() - start ) / n_runs * 1000

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_data = dummy_input.numpy()
    _ = session.run(["logits"], {"image": input_data})
    start = time.time()
    for _ in range(n_runs):
        session.run(["logits"], {"image": input_data})
    onnx_time = (time.time() - start) / n_runs * 1000

    print(f"Pytorch CPU: {pytorch_time:.2f} ms/image")
    print(f"ONNX Runtime CPU: {onnx_time:.2f} ms/image")
    print(f"Acceleration: {pytorch_time / onnx_time:.2f}x")

@hydra.main(version_base=None, config_path= "configs", config_name="config")
def main(cfg: DictConfig):

    ckpt_path = cfg.paths.checkpoint_dir
    onnx_path = cfg.paths.onnx_path
    image_size = cfg.model.image_size

    pytorch_model, dummy_input = export_to_onnx(ckpt_path, onnx_path, image_size)
    verify_outputs(pytorch_model,dummy_input,onnx_path)
    benchmark_speed(pytorch_model,dummy_input,onnx_path)
if __name__ ==  "__main__":
    main()
