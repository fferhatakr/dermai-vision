import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig

sys.path.append(os.getcwd())
from src.engine.trainer_core import DermatologLightning
from src.dataloader.image_dataset import get_album_transform, get_tta_transforms, AlbumentationsDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_with_tta(model, val_indices, data_path, device, tta_transforms,batch_size, num_workers):
    all_probs = []
    all_true = []

    for t_idx, t in enumerate(tta_transforms):
        dataset = AlbumentationsDataset(data_path, album_transform=t)
        val_sub = Subset(dataset, val_indices)
        loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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


def evaluate_standard(model, val_indices, data_path, device, val_album, batch_size, num_workers):
    dataset = AlbumentationsDataset(data_path, album_transform=val_album)
    val_sub = Subset(dataset, val_indices)
    loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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


def find_optimal_threshold(all_probs, all_true, preds_standard, mel_class_idx, target_recall):
    mel_probs = all_probs[:, mel_class_idx]
    thresholds = np.arange(0.05, 0.50, 0.01)

    best_threshold = 0.25
    best_f1 = 0
    best_recall = 0
    target_recall = 0.85

    print(f"\n{'Threshold':>8} | {'Recall':>8} | {'Precision':>10} | {'F1':>8} | {'False Alarm':>12}")
    print("-" * 60)

    for thresh in thresholds:
        preds_thresh = preds_standard.copy()
        preds_thresh[mel_probs >= thresh] = mel_class_idx

        tp = np.sum((preds_thresh == mel_class_idx) & (all_true == mel_class_idx))
        fp = np.sum((preds_thresh == mel_class_idx) & (all_true != mel_class_idx))
        fn = np.sum((preds_thresh != mel_class_idx) & (all_true == mel_class_idx))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if recall >= target_recall:
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                best_recall = recall


    return best_threshold, best_recall, best_f1

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    csv_path = cfg.paths.csv_path
    data_path = cfg.paths.data_path
    model_path = cfg.train.fine_tune_checkpoint

    use_tta = cfg.inference.tta_enabled
    mel_class_idx = cfg.inference.mel_class_idx
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    image_size = cfg.model.image_size
    mean = cfg.model.mean
    std = cfg.model.std
    target_recall = cfg.inference.target_recall
    k_folds = cfg.train.k_folds
    random_seed = cfg.train.random_seed

    _, val_album = get_album_transform(image_size, mean, std)
    tta_transforms = get_tta_transforms(image_size, mean, std)

    df = pd.read_csv(csv_path)
   
    df_nv = df[df['targets'] == 1]
    df_others = df[df['targets'] != 1]
    if len(df_nv) > 3000:
        df_nv = df_nv.sample(n=3000, random_state=42)
    df = pd.concat([df_nv, df_others])
    df['image'] = df['image'].str.replace('_downsampled', '')
    df['lesion_id'] = df['lesion_id'].fillna(df['image'])
    df.set_index('image', inplace=True)

    full_dataset = datasets.ImageFolder(data_path)
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
        print(f"\nEvaluating Fold {fold+1}")

        real_val_indices = valid_imgfolder_indices[val_idx].tolist()

        print(f"Model loading: {model_path}")
        model = DermatologLightning.load_from_checkpoint(model_path, strict=False)
        model.to(DEVICE)
        model.eval()

        if use_tta:
            print(f"\nEvaluation with TTA ({len(tta_transforms)}).")
            all_probs, all_true = evaluate_with_tta(model, real_val_indices, data_path, DEVICE)
        else:
            print("\nStandard Evaluation")
            all_probs, all_true = evaluate_standard(model, real_val_indices, data_path, DEVICE)

        preds_standard = np.argmax(all_probs, axis=1)

        mode = "TTA" if use_tta else "STANDARD"
        print(f"{mode} EVALUATION")
   
        print(classification_report(all_true, preds_standard, target_names=full_dataset.classes))

        best_threshold, best_recall, best_f1 = find_optimal_threshold(all_probs, all_true, preds_standard)
        print(f"\nOptimal MEL threshold: {best_threshold:.2f}")
        print(f"At this threshold, recall: {best_recall:.1%}, F1: {best_f1:.3f}")

        mel_probs = all_probs[:, mel_class_idx]
        preds_optimized = preds_standard.copy()
        preds_optimized[mel_probs >= best_threshold] = mel_class_idx

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