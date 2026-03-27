"""
Sorts the ISIC 2019 photos into class folders.
Skips the UNK class and distributes the remaining 8 classes into folders.
"""

import pandas as pd
import shutil
import os
from tqdm import tqdm

# Config 
CSV_PATH = "data/raw/ISIC_2019_Training_GroundTruth.csv"
IMAGE_DIR = "data/raw/all_data"
OUTPUT_DIR = "data/processed/full_dataset"
CLASS_MAP = {
    "MEL": "0_mel",
    "NV": "1_nv",
    "BCC": "2_bcc",
    "AK": "3_ak",
    "BKL": "4_bkl",
    "DF": "5_df",
    "VASC": "6_vasc",
    "SCC": "7_scc"
}

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"CSV: {len(df)} kayıt")

    
    for folder_name in CLASS_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, folder_name), exist_ok=True)

    counts = {v: 0 for v in CLASS_MAP.values()}
    skipped = 0
    not_found = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizing"):
        image_name = row["image"]

        assigned_class = None
        for col, folder in CLASS_MAP.items():
            if row[col] == 1.0:
                assigned_class = folder
                break

        if assigned_class is None:
            skipped += 1
            continue


        src = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            candidate = os.path.join(IMAGE_DIR, image_name + ext)
            if os.path.exists(candidate):
                src = candidate
                break

        if src is None:
            not_found += 1
            continue

        dst = os.path.join(OUTPUT_DIR, assigned_class, os.path.basename(src))
        shutil.copy2(src, dst)
        counts[assigned_class] += 1

    print("\n" + "=" * 40)
    print("=" * 40)
    total = 0
    for folder, count in sorted(counts.items()):
        print(f"  {folder}: {count}")
        total += count
    print(f"\n  Total: {total}")
    print(f"  Skipped: {skipped}")
    print(f"  Not found: {not_found}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()