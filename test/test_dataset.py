import torch
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

def test_processed_data_exists():

    path_lower = ROOT_DIR / "Data" / "processed" / "processed_data.pt"

    path_upper = ROOT_DIR / "Data" / "Processed" / "processed_data.pt"
    
    final_path = None
    if path_lower.exists():
        final_path = path_lower
    elif path_upper.exists():
        final_path = path_upper
    
   
    assert final_path is not None, f"VNO DATA FILE! We looked here and there but couldn't find it:\n1. {path_lower}\n2. {path_upper}\nPlease 'src/utils/rebuild_data.py' run."

    print(f" Data found: {final_path}")

def test_data_content_format():

    path_lower = ROOT_DIR / "Data" / "processed" / "processed_data.pt"
    path_upper = ROOT_DIR / "Data" / "Processed" / "processed_data.pt"
    
    data_path = path_lower if path_lower.exists() else path_upper
    
    if data_path.exists():
        data = torch.load(data_path)
        assert len(data) > 0, "The data set is empty!"
        first_sample = data[0]
        assert len(first_sample) == 2, "The format is incorrect (Image, Tag)."
        assert isinstance(first_sample[0], torch.Tensor), "The image is not in Tensor format.."