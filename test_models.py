"""
Model Testing Script with Label Correction
"""

import os
# Suppress TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU

from ultralytics import YOLO
import tensorflow as tf
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

# ==============================
# CONFIGURATION
# ==============================
YOLO_MODEL_PATH = Path("models/yolo/train/weights/best.pt")
CLASSIFIER_DIR = Path("models/classifier")

FEATURES = [
    "emblem", 
    "watermark", 
    "denomination_number", 
    "denomination_text", 
    "rbi_seal", 
    "security_features"
]

# --- CRITICAL FIX: LABEL CORRECTION MAP ---
# This maps the [Incorrect YOLO Label] -> [Correct Real Label]
LABEL_MAP = {
    "denomination_number": "emblem",           # Fixes Ashoka Pillar
    "denomination_text": "rbi_seal",           # Fixes RBI Seal
    "watermark": "denomination_text",          # Fixes Hindi Text
    "emblem": "denomination_number",           # Fixes Big Number '20'
    "rbi_seal": "security_features",           # Fixes Serial Number
    "security_features": "watermark",          # Fixes Gandhi Portrait
    
    # Legacy/Variations
    "special_features": "security_features",
    "rbi seal": "rbi_seal"
}

TEST_IMAGES_DIR = Path("dataset/yolo/images/test")
# ==============================

class CurrencyNoteDetector:
    def __init__(self):
        self.yolo_model = None
        self.classifiers = {}
        self.load_models()
    
    def load_models(self):
        print("\n📥 Loading models...")
        
        # 1. Load YOLO
        if YOLO_MODEL_PATH.exists():
            try:
                self.yolo_model = YOLO(str(YOLO_MODEL_PATH))
                print("  ✓ YOLO model loaded")
            except Exception as e:
                print(f"  ❌ YOLO Error: {e}")
        else:
            print(f"  ⚠️ YOLO model not found at {YOLO_MODEL_PATH}")
        
        # 2. Load Classifiers
        for feature in FEATURES:
            model_path = CLASSIFIER_DIR / feature / "best_model.h5"
            if model_path.exists():
                try:
                    self.classifiers[feature] = tf.keras.models.load_model(str(model_path))
                    print(f"  ✓ {feature} classifier loaded")
                except Exception as e:
                    print(f"  ❌ Failed to load {feature}: {e}")
            else:
                print(f"  ⚠️ {feature} classifier not found")
        print("-" * 50)
    
    def classify_feature(self, image_rgb, feature_name):
        """Classify a specific feature crop as real or fake."""
        if feature_name not in self.classifiers:
            return None, 0.0
        
        # Preprocess: Resize to 224x224, Normalize
        img_resized = cv2.resize(image_rgb, (224, 224))
        img_norm = img_resized.astype("float32") / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)
        
        # Predict
        prediction = self.classifiers[feature_name].predict(img_batch, verbose=0)[0][0]
        is_real = prediction > 0.5
        confidence = prediction if is_real else (1 - prediction)
        
        return is_real, confidence

    def normalize_label(self, raw_label):
        """Fixes the label name using the LABEL_MAP."""
        clean = raw_label.lower().strip()
        if clean in LABEL_MAP:
            return LABEL_MAP[clean]
        if clean in FEATURES:
            return clean
        return clean

    def process_image(self, image_path, visualize=True):
        path_obj = Path(image_path)
        if not path_obj.exists():
            print(f"❌ File not found: {image_path}")
            return

        print(f"\n🔍 Processing: {path_obj.name}")
        
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None: return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_bgr.shape[:2]

        if not self.yolo_model: return

        # Run YOLO with lower threshold to catch everything
        results = self.yolo_model(img_rgb, conf=0.25, verbose=False)[0]
        
        detections = []
        feature_report = {}

        if len(results.boxes) == 0:
            print("  ⚠️ No features detected.")
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            raw_label = results.names[cls_id]
            
            # --- APPLY LABEL CORRECTION HERE ---
            mapped_label = self.normalize_label(raw_label)
            
            # Debug Print
            # print(f"  Found '{raw_label}' -> Swapped to '{mapped_label}' ({conf:.2f})")

            # Store detection with CORRECTED label
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "label": mapped_label, 
                "conf": conf
            })

            # Run Classifier using CORRECTED label
            if mapped_label in self.classifiers:
                pad = 5
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(img_w, x2+pad), min(img_h, y2+pad)
                crop = img_rgb[cy1:cy2, cx1:cx2]
                
                if crop.size > 0:
                    is_real, auth_conf = self.classify_feature(crop, mapped_label)
                    status = "✅ REAL" if is_real else "❌ FAKE"
                    feature_report[mapped_label] = f"{status} {auth_conf:.0%}"
                    print(f"    -> {mapped_label:20} : {status} (Conf: {auth_conf:.2f})")
            
        if visualize:
            self.visualize(img_rgb, detections, feature_report)

    def visualize(self, img_rgb, detections, report):
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        ax = plt.gca()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["conf"]
            
            # Determine Color
            if label in report:
                color = 'lime' if "REAL" in report[label] else 'red'
                text_label = f"{label.upper()}\n{report[label]}"
            else:
                color = 'orange'
                text_label = f"{label} (?)"

            # Draw Box
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Draw Text
            ax.text(x1, y1-5, text_label, fontsize=9, color='white', 
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))

        plt.axis('off')
        plt.tight_layout()
        plt.show()

# ==============================
# MENU
# ==============================
def get_image_files(directory):
    return list(directory.glob("*.jpg")) + list(directory.glob("*.png"))

def main():
    detector = CurrencyNoteDetector()
    
    while True:
        print("\n" + "="*40)
        print("   TEST MENU (WITH LABEL FIX)")
        print("="*40)
        print("1. 🖼️  Test Random Image")
        print("2. 📂  Test ALL Images")
        print("3. 📍  Test Specific Path")
        print("q. 🚪  Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == '1':
            files = get_image_files(TEST_IMAGES_DIR)
            if files:
                detector.process_image(random.choice(files))
            else:
                print("No images found.")

        elif choice == '2':
            files = get_image_files(TEST_IMAGES_DIR)
            for img_path in files:
                detector.process_image(img_path)
                input("Press Enter for next...")

        elif choice == '3':
            path_str = input("Path: ").strip().replace('"', '')
            detector.process_image(path_str)

        elif choice == 'q':
            break

if __name__ == "__main__":
    main()