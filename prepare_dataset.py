import os
import random
import shutil
from pathlib import Path
import yaml

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = Path("dataset")
YOLO_DIR = BASE_DIR / "yolo"
CLASSIFIER_DIR = BASE_DIR / "classifier"

YOLO_SPLIT = [0.8, 0.1]   # train / val / test
CLASSIFIER_SPLIT = [0.8, 0.2]  # train / val

# STANDARDIZED FEATURE NAMES
FEATURES = [
    "emblem", 
    "watermark", 
    "denomination_number", 
    "denomination_text", 
    "rbi_seal", 
    "security_features"
]

LABELS = ["real", "fake"]  # For each feature
# ==============================


def make_dirs():
    """Ensure all necessary directories exist."""
    print("📂 Creating dataset folders...")
    
    # YOLO directories
    for sub in ["images", "labels"]:
        for split in ["train", "val", "test"]:
            (YOLO_DIR / sub / split).mkdir(parents=True, exist_ok=True)
    
    # Classifier directories
    for split in ["train", "val"]:
        for feature in FEATURES:
            for label in LABELS:
                (CLASSIFIER_DIR / split / feature / label).mkdir(parents=True, exist_ok=True)
    
    print("✅ All folders created.\n")


def create_data_yaml():
    """Create data.yaml for YOLO training."""
    print("📝 Creating data.yaml for YOLO...")
    
    data_config = {
        'train': str(YOLO_DIR / 'images' / 'train'),
        'val': str(YOLO_DIR / 'images' / 'val'),
        'test': str(YOLO_DIR / 'images' / 'test'),
        'nc': 7,  # number of classes
        'names': [
            'emblem',
            'watermark',
            'denomination_number',
            'denomination_text',
            'rbi_seal',
            'security_features',
            'currency_note'
        ]
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print("✅ data.yaml created.\n")


def split_yolo_dataset():
    """Split YOLO images and labels into train/val/test sets."""
    print("🔄 Splitting YOLO dataset...")
    
    images_all_dir = YOLO_DIR / "images" / "all"
    labels_all_dir = YOLO_DIR / "labels" / "all"
    
    if not images_all_dir.exists():
        print(f"⚠️ {images_all_dir} not found. Creating placeholder...")
        images_all_dir.mkdir(parents=True, exist_ok=True)
        print("Please add your images and labels to:")
        print(f"  - {images_all_dir}")
        print(f"  - {labels_all_dir}")
        return
    
    all_imgs = list(images_all_dir.glob("*.jpg")) + list(images_all_dir.glob("*.png"))
    
    if not all_imgs:
        print("⚠️ No images found in YOLO dataset.")
        return
    
    random.shuffle(all_imgs)
    n_total = len(all_imgs)
    n_train = int(n_total * YOLO_SPLIT[0])
    n_val = int(n_total * YOLO_SPLIT[1])

    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train + n_val],
        "test": all_imgs[n_train + n_val:]
    }

    for split, imgs in splits.items():
        for img_path in imgs:
            label_path = labels_all_dir / (img_path.stem + ".txt")
            img_out = YOLO_DIR / "images" / split / img_path.name
            lbl_out = YOLO_DIR / "labels" / split / label_path.name
            
            shutil.copy(img_path, img_out)
            if label_path.exists():
                shutil.copy(label_path, lbl_out)

    print(f"✅ YOLO dataset split: {len(splits['train'])} train, "
          f"{len(splits['val'])} val, {len(splits['test'])} test.\n")


def split_classifier_dataset():
    """Split classifier data by feature into train/val."""
    print("🔄 Splitting classifier dataset...")

    for feature in FEATURES:
        for label in LABELS:
            src_dir = CLASSIFIER_DIR / "all" / feature / label
            
            if not src_dir.exists():
                # Try to find old names and warn
                if feature == "security_features" and (CLASSIFIER_DIR / "all" / "special_features" / label).exists():
                     print(f"⚠️ PLEASE RENAME FOLDER: 'special_features' -> 'security_features'")
                
                print(f"⚠️ Creating {src_dir} (Empty)...")
                src_dir.mkdir(parents=True, exist_ok=True)
                continue

            imgs = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
            
            if not imgs:
                print(f"⚠️ No images in {feature}/{label}")
                continue
            
            random.shuffle(imgs)
            n_total = len(imgs)
            n_train = int(n_total * CLASSIFIER_SPLIT[0])

            train_dir = CLASSIFIER_DIR / "train" / feature / label
            val_dir = CLASSIFIER_DIR / "val" / feature / label

            for i, img_path in enumerate(imgs):
                dest = train_dir if i < n_train else val_dir
                shutil.copy(img_path, dest / img_path.name)
            
            print(f"  ✓ {feature}/{label}: {n_train} train, {n_total - n_train} val")

    print("✅ Classifier dataset split completed.\n")


def verify_dataset():
    """Verify dataset structure and print statistics."""
    print("🔍 Verifying dataset structure...\n")
    
    # YOLO stats
    print("YOLO Dataset:")
    for split in ["train", "val", "test"]:
        n_imgs = len(list((YOLO_DIR / "images" / split).glob("*")))
        n_lbls = len(list((YOLO_DIR / "labels" / split).glob("*.txt")))
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")
    
    print("\nClassifier Dataset:")
    for split in ["train", "val"]:
        print(f"  {split}:")
        for feature in FEATURES:
            for label in LABELS:
                path = CLASSIFIER_DIR / split / feature / label
                n = len(list(path.glob("*"))) if path.exists() else 0
                print(f"    {feature}/{label}: {n} images")
    
    print()


def main():
    print("🚀 Starting dataset setup...\n")
    make_dirs()
    create_data_yaml()
    split_yolo_dataset()
    split_classifier_dataset()
    verify_dataset()
    print("🎉 Dataset preparation completed!\n")

if __name__ == "__main__":
    main()