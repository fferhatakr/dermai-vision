import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
from tqdm import tqdm
import sys

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning
from src.dataloader.image_dataset import val_transforms

DATA_DIR = "data/processed/kfold_data"
CSV_PATH = "data/raw/ISIC_2019_Training_Metadata.csv"
MODEL_PATH = "models/kfold_models/best_model.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    model = DermatologLightning.load_from_checkpoint(MODEL_PATH,strict=False)
    model.to(DEVICE)
    model.eval()

    dataset = datasets.ImageFolder(DATA_DIR,transform=val_transforms)
    dataloader= DataLoader(dataset,batch_size=32,shuffle=False, num_workers=4)

    
    preds = []
    targets = []

    
    with torch.no_grad(): 
        for images, labels in tqdm(dataloader, desc="CNN Probabilities are being calculated"):
            
            
            images=images.to(DEVICE)
            logits = model(images)
            pred = F.softmax(logits,dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.numpy())
            

    print("Finish.")

    image_ids = []
    for path, label in dataset.samples:
        name = os.path.basename(path) 
        just_name = name.split('.')[0] 
        image_ids.append(just_name)

    class_name = ['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']
    df = pd.DataFrame(preds,columns=class_name)
    df['image'] = image_ids
    df['targets'] = targets

    big_df = pd.read_csv(CSV_PATH)
    df_final = pd.merge(df,big_df,on='image')

    mean_age = df_final['age_approx'].median()
    df_final['age_approx'] = df_final['age_approx'].fillna(mean_age)
    df_final['sex'] = df_final['sex'].fillna('unknown')
    df_final['anatom_site_general'] = df_final['anatom_site_general'].fillna('unknown')
    df_final.to_csv("data/processed/meta_dataset_fold4.csv", index=False)
    print("Meta-Dataset successfully created")


if __name__ == "__main__":
    main()