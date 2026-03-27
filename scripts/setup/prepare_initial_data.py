"""
Combines ISIC 2019 Ground Truth and Metadata
and generates a CSV file in the oof_meta_dataset format for training.
"""

import pandas as pd
import os

GT_PATH = "data/raw/ISIC_2019_Training_GroundTruth.csv"
META_PATH = "data/raw/ISIC_2019_Training_Metadata.csv"
OUTPUT_PATH = "data/processed/full_metadata.csv"
IMAGE_DIR = "data/processed/full_dataset"

CLASS_COLS = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_MAP = {
    'MEL': 0, 'NV': 1, 'BCC': 2, 'AK': 3,
    'BKL': 4, 'DF': 5, 'VASC': 6, 'SCC': 7
}

def main():
    
    gt = pd.read_csv(GT_PATH)
    print(f"Ground Truth: {len(gt)} kayıt")

    
    def get_target(row):
        for col in CLASS_COLS:
            if row[col] == 1.0:
                return CLASS_MAP[col]
        return -1

    gt['targets'] = gt.apply(get_target, axis=1)
    gt = gt[gt['targets'] != -1]  
    print(f"After UNK is discarded: {len(gt)} records")
    
    meta = pd.read_csv(META_PATH)
    print(f"Metadata: {len(meta)}")

    # Merge
    df = pd.merge(gt[['image', 'targets']], meta, on='image', how='left')
    df['lesion_id'] = df['lesion_id'].fillna(df['image'])

    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())
    df['sex'] = df['sex'].fillna('unknown')
    df['anatom_site_general'] = df['anatom_site_general'].fillna('unknown')

    existing_images = set()
    for class_dir in os.listdir(IMAGE_DIR):
        class_path = os.path.join(IMAGE_DIR, class_dir)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                name = os.path.splitext(f)[0]
                existing_images.add(name)

    df = df[df['image'].isin(existing_images)]
    print(f"Not found in the folder: {len(df)}")

    for col, idx in sorted(CLASS_MAP.items(), key=lambda x: x[1]):
        count = len(df[df['targets'] == idx])
        print(f"  {idx}_{col.lower()}: {count}")

    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()