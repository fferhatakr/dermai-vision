import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.utils.class_weight import compute_sample_weight

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):

    os.makedirs(cfg.meta.meta_model_dir, exist_ok=True)
    df = pd.read_csv(cfg.meta.meta_csv_path)

    if 'targets_x' in df.columns:
        df.rename(columns={'targets_x': 'targets'}, inplace=True)
        if 'targets_y' in df.columns:
            df.drop(columns=['targets_y'], inplace=True)

    cnn_features = cfg.meta.cnn_features
    feature_cols = cnn_features
    X = df[feature_cols].values
    DERM12345_TO_ISIC = {
        "mel": 0, "lm" : 0, "lmm": 0, "alm": 0, "anm": 0,
        "jb":   1, "ajb":  1, "cb":   1, "db":  1, "acb":  1,
        "bdb":  1, "ccb":  1, "cjb":  1, "mcb": 1,
        "jd":   1, "ajd":  1, "cd":   1, "acd": 1, "ccd":  1,
        "rd":   1, "srjd": 1,
        "bcc":2, "ak": 3, "sk": 4, "ls": 4, "sl": 4, "lk" : 4, "isl":4,
        "df":5, "ha":6, "la":6, "pg":6, "sa":6, "angk":6, "ks":6,
        "scc":7, "bd":7,
    }
    label_col = cfg.meta.oof_label_col
    clean_labels = df[label_col].astype(str).str.lower().str.strip()
    df['true_targets'] = clean_labels.map(DERM12345_TO_ISIC)
    
    if df['true_targets'].isnull().any():
        missing_labels = clean_labels[df['true_targets'].isnull()].unique()
        print(f"NOTE: The following tags, which could not be found in the dictionary, have been removed: {missing_labels}")
        df = df.dropna(subset=['true_targets'])
        X = df[feature_cols].values
        groups = df[cfg.meta.oof_group_col].values

    y = df['true_targets'].astype(int).values
    df['true_targets'] = df[label_col].map(DERM12345_TO_ISIC)
    groups = df[cfg.meta.oof_group_col].values


   
    xgb_params = OmegaConf.to_container(cfg.meta.xgb_params, resolve=True)

    sgkf = StratifiedGroupKFold(n_splits=cfg.meta.meta_folds, shuffle=True, random_state=cfg.train.random_seed)
    
    oof_preds = np.zeros((len(df), cfg.model.num_classes))
    fold_scores = []
    best_fold_score = -1
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"\n--- Training Fold {fold + 1} ---")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        weights = compute_sample_weight('balanced', y_train)
        mel_idx = cfg.inference.mel_class_idx
        weights[y_train == mel_idx] *= 2.0

        model = xgb.XGBClassifier(**xgb_params)
        
        model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_preds = model.predict_proba(X_val)
        oof_preds[val_idx] = val_preds

        val_preds_classes = np.argmax(val_preds, axis=1)

        mel_f1 = f1_score(y_val, val_preds_classes, labels=[cfg.inference.mel_class_idx], average='macro')
        fold_scores.append(mel_f1)
        print(f"Fold {fold + 1} Melanoma F1-Score: {mel_f1:.4f}")

        if  mel_f1 > best_fold_score:
            best_fold_score = mel_f1
            model.save_model(cfg.meta.meta_model_path)
            joblib.dump(feature_cols, cfg.meta.meta_feature_path)
            print(f"New best model saved at fold {fold+1} (MEL F1: {mel_f1:.4f})")

    print("META-LEARNER FINAL OOF PERFORMANCE")
    final_preds_classes = np.argmax(oof_preds, axis=1)
    
    class_names_clean = [name.split('_')[1].upper() for name in cnn_features]
    print(classification_report(y, final_preds_classes, target_names=class_names_clean))
    print(f"Average Melanoma F1-Score across all folds: {np.mean(fold_scores):.4f}")


    print("\nGenerating Confusion Matrix")
    cm = confusion_matrix(y, final_preds_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_clean, yticklabels=class_names_clean)
    
    plt.title('Meta-Learner (XGBoost) Final Confusion Matrix', fontsize=16)
    plt.ylabel('True Clinical Diagnosis', fontsize=12)
    plt.xlabel('Meta-Learner Prediction', fontsize=12)
    
    plt.tight_layout()
    cm_filename = cfg.paths.cm_plot_path
    plt.savefig(cm_filename, dpi=150)
    print(f"Saved Confusion Matrix visualization to: {cm_filename}")

    plt.show()

if __name__ == "__main__":
    main()