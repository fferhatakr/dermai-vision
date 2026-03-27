import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the newly generated Out-Of-Fold (OOF) metadata
CSV_PATH = "data/processed/oof_meta_dataset.csv"
K_FOLDS = 5

def main():
    print("Loading OOF Meta-Dataset...")
    df = pd.read_csv(CSV_PATH)

    # 1. Clean up duplicate columns from the merge process
    if 'targets_x' in df.columns:
        df.rename(columns={'targets_x': 'targets'}, inplace=True)
        if 'targets_y' in df.columns:
            df.drop(columns=['targets_y'], inplace=True)

    # 2. Encode categorical clinical data into numbers for XGBoost
    print("Encoding categorical clinical features...")
    le_sex = LabelEncoder()
    df['sex_encoded'] = le_sex.fit_transform(df['sex'].astype(str))

    le_site = LabelEncoder()
    df['site_encoded'] = le_site.fit_transform(df['anatom_site_general'].astype(str))

    # Save the encoders so the deployment API knows how to process new patient data
    joblib.dump(le_sex, "models/detector/le_sex.pkl")
    joblib.dump(le_site, "models/detector/le_site.pkl")

    # 3. Define the features (inputs) and the target (output)
    cnn_features = ['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']
    clinical_features = ['age_approx', 'sex_encoded', 'site_encoded']
    
    feature_cols = cnn_features + clinical_features
    X = df[feature_cols].values
    y = df['targets'].values
    groups = df['lesion_id'].values

    print(f"\nTraining Meta-Learner with {len(feature_cols)} features: {feature_cols}")

    # 4. Set up the XGBoost Model architecture
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 8,
        'eval_metric': 'mlogloss',
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 150,
        'random_state': 42
    }

    sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((len(df), 8))
    fold_scores = []

    # 5. Train the Meta-Learner using K-Fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"\n--- Training Fold {fold + 1} ---")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = xgb.XGBClassifier(**xgb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_preds = model.predict_proba(X_val)
        oof_preds[val_idx] = val_preds

        val_preds_classes = np.argmax(val_preds, axis=1)
        mel_f1 = f1_score(y_val, val_preds_classes, labels=[0], average='macro')
        fold_scores.append(mel_f1)
        print(f"Fold {fold + 1} Melanoma F1-Score: {mel_f1:.4f}")

        if fold == 0:
            model.save_model("models/kfold_models/xgb_meta_learner.json")
            joblib.dump(feature_cols, "models/kfold_models/xgb_features.pkl")
            print("Saved production Meta-Learner model (xgb_meta_learner.json).")

    # 6. Evaluate the final unified performance
    print("\n" + "="*50)
    print("META-LEARNER FINAL OOF PERFORMANCE")
    print("="*50)
    final_preds_classes = np.argmax(oof_preds, axis=1)
    
    class_names_clean = [name.split('_')[1].upper() for name in cnn_features]
    print(classification_report(y, final_preds_classes, target_names=class_names_clean))
    print(f"Average Melanoma F1-Score across all folds: {np.mean(fold_scores):.4f}")

    # 7. Visualize the Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y, final_preds_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_clean, yticklabels=class_names_clean)
    
    plt.title('Meta-Learner (XGBoost) Final Confusion Matrix', fontsize=16)
    plt.ylabel('True Clinical Diagnosis', fontsize=12)
    plt.xlabel('Meta-Learner Prediction', fontsize=12)
    
    plt.tight_layout()
    cm_filename = "models/kfold_models/meta_learner_cm.png"
    plt.savefig(cm_filename, dpi=150)
    print(f"Saved Confusion Matrix visualization to: {cm_filename}")
    
    # Display the plot directly on your screen
    plt.show()

if __name__ == "__main__":
    main()