import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F

sys.path.append(os.getcwd())
from src.training.trainer_core import DermatologLightning
from src.dataloader.image_dataset import val_transforms


CSV_PATH = "data/processed/full_metadata.csv"
DATA_PATH = "data/processed/just_stain"
MODEL_PATH = "models/kfold_models/best_modelv2.ckpt"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEL_CLASS_IDX = 0
USE_TTA = True  

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
"""
This script:

1. Loads the model
2. Creates a validation set
3. Generates predictions using TTA
4. Optimizes the threshold for melanoma
5. Analyzes performance
6. Visualizes the results

TTA= Test time augmentation
If we want to present the same image to the model in a different way—to tell a story—
we can think of it as if five doctors were making a joint decision
"""

tta_transforms = [
    val_transforms, 
    transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomRotation(degrees=(90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomRotation(degrees=(270, 270)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
]


def evaluate_with_tta(model, val_indices, data_path, device):
    all_probs = []
    all_true = []

    for t_idx, t in enumerate(tta_transforms):
        dataset = datasets.ImageFolder(data_path, transform=t)
        val_sub = Subset(dataset, val_indices)
        loader = DataLoader(val_sub, batch_size=16, shuffle=False, num_workers=0)

        fold_probs = []
        fold_true = []

        with torch.no_grad():
            desc = f"TTA {t_idx+1}/{len(tta_transforms)}" if t_idx > 0 else "Original"
            for images, labels in tqdm(loader, desc=desc):
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                fold_probs.extend(probs.cpu().numpy())
                if t_idx == 0:
                    fold_true.extend(labels.numpy())

        all_probs.append(np.array(fold_probs))
        if t_idx == 0:
            all_true = np.array(fold_true)

    
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs, all_true


def evaluate_standard(model, val_indices, data_path, device):

    dataset = datasets.ImageFolder(data_path, transform=val_transforms)
    val_sub = Subset(dataset, val_indices)
    loader = DataLoader(val_sub, batch_size=16, shuffle=False, num_workers=0)

    all_probs = []
    all_true = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_true.extend(labels.numpy())

    return np.array(all_probs), np.array(all_true)


def find_optimal_threshold(all_probs, all_true, preds_standard):
    mel_probs = all_probs[:, MEL_CLASS_IDX]
    thresholds = np.arange(0.05, 0.50, 0.01)

    best_threshold = 0.5
    best_f1 = 0
    best_recall = 0

    print(f"\n{'Threshold':>8} | {'Recall':>8} | {'Precision':>10} | {'F1':>8} | {'False Alarm':>12}")
    print("-" * 60)

    for thresh in thresholds:
        preds_thresh = preds_standard.copy()
        preds_thresh[mel_probs >= thresh] = MEL_CLASS_IDX

        tp = np.sum((preds_thresh == MEL_CLASS_IDX) & (all_true == MEL_CLASS_IDX))
        fp = np.sum((preds_thresh == MEL_CLASS_IDX) & (all_true != MEL_CLASS_IDX))
        fn = np.sum((preds_thresh != MEL_CLASS_IDX) & (all_true == MEL_CLASS_IDX))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if recall >= 0.80 and f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_recall = recall

        if thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
            print(f"{thresh:>8.2f} | {recall:>7.1%} | {precision:>9.1%} | {f1:>7.3f} | {fp:>12}")

    return best_threshold, best_recall, best_f1


def main():
    df = pd.read_csv(CSV_PATH)
    df['image'] = df['image'].str.replace('_downsampled', '')
    df['lesion_id'] = df['lesion_id'].fillna(df['image'])
    df.set_index('image', inplace=True)

    full_dataset = datasets.ImageFolder(DATA_PATH)
    clean_file_names = []
    for f in full_dataset.imgs:
        raw_name = os.path.splitext(os.path.basename(f[0]))[0]
        clean_name = raw_name.replace('_downsampled', '')
        clean_file_names.append(clean_name)

    valid_imgfolder_indices = []
    valid_targets = []
    valid_groups = []

    for img_idx, name in enumerate(clean_file_names):
        if name in df.index:
            valid_imgfolder_indices.append(img_idx)
            valid_targets.append(df.loc[name, 'targets'])
            valid_groups.append(df.loc[name, 'lesion_id'])

    valid_imgfolder_indices = np.array(valid_imgfolder_indices)
    valid_targets = np.array(valid_targets)
    valid_groups = np.array(valid_groups)

    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(len(valid_targets)), valid_targets, groups=valid_groups)):
        if fold > 0:
            break

        real_val_indices = valid_imgfolder_indices[val_idx].tolist()

       
        print(f"Model loading: {MODEL_PATH}")
        model = DermatologLightning.load_from_checkpoint(MODEL_PATH, strict=False)
        model.to(DEVICE)
        model.eval()

        
        if USE_TTA:
            print(f"\nEvaluation with TTA ({len(tta_transforms)})...")
            all_probs, all_true = evaluate_with_tta(model, real_val_indices, DATA_PATH, DEVICE)
        else:
            print("\nStandard Evaluation")
            all_probs, all_true = evaluate_standard(model, real_val_indices, DATA_PATH, DEVICE)

        preds_standard = np.argmax(all_probs, axis=1)

        print("\n" + "=" * 60)
        mode = "TTA" if USE_TTA else "STANDARD"
        print(f"{mode} EVALUATION")
        print("=" * 60)
        print(classification_report(all_true, preds_standard, target_names=full_dataset.classes))

        best_threshold, best_recall, best_f1 = find_optimal_threshold(all_probs, all_true, preds_standard)
        print(f"\nOptimal MEL threshold: {best_threshold:.2f}")
        print(f"At this threshold, recall: {best_recall:.1%}, F1: {best_f1:.3f}")

        
        mel_probs = all_probs[:, MEL_CLASS_IDX]
        preds_optimized = preds_standard.copy()
        preds_optimized[mel_probs >= best_threshold] = MEL_CLASS_IDX

        print(f"\nTHRESHOLD ({best_threshold:.2f}):")
        print(classification_report(all_true, preds_optimized, target_names=full_dataset.classes))

        
        cm_standard = confusion_matrix(all_true, preds_standard)
        cm_optimized = confusion_matrix(all_true, preds_optimized)

        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        sns.heatmap(cm_standard, annot=True, fmt='d', cmap='Blues',
                    xticklabels=full_dataset.classes, yticklabels=full_dataset.classes, ax=axes[0])
        axes[0].set_title(f'{mode} — Standard (argmax)')
        axes[0].set_ylabel('True')
        axes[0].set_xlabel('Predicted')

        sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues',
                    xticklabels=full_dataset.classes, yticklabels=full_dataset.classes, ax=axes[1])
        axes[1].set_title(f'{mode} — MEL threshold = {best_threshold:.2f}')
        axes[1].set_ylabel('True')
        axes[1].set_xlabel('Predicted')

        plt.tight_layout()
        filename = f"cm_{mode.lower()}_threshold.png"
        plt.savefig(filename, dpi=150)
        plt.show()
        print(f"\nSaved: {filename}")


if __name__ == "__main__":
    main()