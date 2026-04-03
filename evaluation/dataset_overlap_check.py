import pandas as pd

isic = pd.read_csv("data/processed/full_metadata.csv")
ham  = pd.read_csv("data/HAM10000_metadata.csv")

print(f"ISIC image count  : {len(isic)}")
print(f"HAM10000 image count: {len(ham)}")

overlap_images = set(isic["image"]).intersection(set(ham["image_id"]))
print(f"\nImage ID overlap  : {len(overlap_images)}")

overlap_lesions = set(isic["lesion_id"]).intersection(set(ham["lesion_id"]))
print(f"Lesion ID overlap : {len(overlap_lesions)}")

if len(overlap_images) == 0 and len(overlap_lesions) == 0:
    print("\nClean. No data leakage.")
elif len(overlap_images) == 0 and len(overlap_lesions) > 0:
    print(f"\nNo image overlap but {len(overlap_lesions)} shared lesion IDs.")
    print("This is expected due to dataset origin. No image-level leakage.")
else:
    print(f"\nWARNING: {len(overlap_images)} images exist in both datasets.")
    print("Remove them from test set.")