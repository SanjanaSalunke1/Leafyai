import os
import shutil
import random

SRC_DIR = "Bell_paper_data/dataset_og"
DST_DIR = "Bell_paper_data/dataset"

TRAIN_DIR = os.path.join(DST_DIR, "train")
VAL_DIR = os.path.join(DST_DIR, "val")

SPLIT = 0.8
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".webp", ".WEBP")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for cls in os.listdir(SRC_DIR):
    src_cls = os.path.join(SRC_DIR, cls)
    if not os.path.isdir(src_cls):
        continue

    imgs = []
    for root, _, files in os.walk(src_cls):
        for f in files:
            if f.endswith(IMG_EXT):
                imgs.append(os.path.join(root, f))

    random.shuffle(imgs)
    split_idx = int(len(imgs) * SPLIT)

    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]

    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy2(img, os.path.join(TRAIN_DIR, cls, os.path.basename(img)))

    for img in val_imgs:
        shutil.copy2(img, os.path.join(VAL_DIR, cls, os.path.basename(img)))

    print(f"{cls}: {len(train_imgs)} train | {len(val_imgs)} val")

print(" SAFE dataset split completed")
