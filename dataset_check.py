import os
from PIL import Image

DATASET_PATH = "bell_paper_data/dataset"
SPLITS = ["train", "val"]

def check_images(folder):
    valid, invalid = 0, 0
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1
        except:
            invalid += 1
    return valid, invalid

for split in SPLITS:
    print(f"\n--- {split.upper()} ---")
    split_path = os.path.join(DATASET_PATH, split)

    for cls in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, cls)
        if not os.path.isdir(class_path):
            continue

        valid, invalid = check_images(class_path)
        total = valid + invalid

        print(f"{cls:20} | total: {total:4} | valid: {valid:4} | corrupt: {invalid}")
