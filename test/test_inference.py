import torch
import glob
import os
import sys


sys.path.append(os.getcwd())
from src.training.trainer_core import TripletLightning

def test_model_discovery_and_loading():

    path1 = glob.glob("lightning_logs/version_*/checkpoints/*.ckpt")
    path2 = glob.glob("models/*.ckpt*") 
    checkpoints = path1 + path2
    

    if not checkpoints:
        print("WARNING: The .ckpt model file to be tested could not be found.")
        return 

    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f"The model being tested: {latest_ckpt}")
    

    try:

        model = TripletLightning.load_from_checkpoint(
            latest_ckpt,
            learning_rate=0.001,
            margin_value=1.0,
            map_location=torch.device('cpu')
        )
        model.eval()
        
        assert model is not None, "Model could not be loaded (None returned)!"
        
    except Exception as e:
        assert False, f"An error occurred while loading the model: {e}"