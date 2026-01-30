"""
YOLO Training Script for Currency Note Detection
Trains YOLOv8 model to detect currency notes and their features
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

# ==============================
# CONFIGURATION
# ==============================
MODEL_SIZE = 'n'  # n, s, m, l, x (nano, small, medium, large, xlarge)
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_NAME = 'models/yolo'
PATIENCE = 20  # Early stopping patience
# ==============================


def setup_training():
    """Setup training environment and verify data."""
    print("🔧 Setting up YOLO training environment...")
    
    # Check if data.yaml exists
    if not Path('data.yaml').exists():
        print("❌ data.yaml not found. Run prepare_dataset.py first!")
        return False
    
    # Load and verify data.yaml
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"✓ Dataset configuration loaded")
    print(f"  Train path: {data_config.get('train')}")
    print(f"  Val path: {data_config.get('val')}")
    print(f"  Classes: {data_config.get('nc')}")
    print(f"  Names: {data_config.get('names')}\n")
    
    return True


def train_yolo():
    """Train YOLO model for currency note detection."""
    print(f"🚀 Starting YOLO training on {DEVICE}...\n")
    
    # Initialize model
    model = YOLO(f'yolov8{MODEL_SIZE}.pt')  # Load pretrained model
    
    print(f"📊 Training Configuration:")
    print(f"  Model: YOLOv8{MODEL_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  Patience: {PATIENCE}\n")
    
    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name='train',
        exist_ok=True,
        patience=PATIENCE,
        
        # Augmentation settings
        hsv_h=0.015,      # HSV-Hue augmentation
        hsv_s=0.7,        # HSV-Saturation augmentation
        hsv_v=0.4,        # HSV-Value augmentation
        degrees=10.0,     # Rotation augmentation
        translate=0.1,    # Translation augmentation
        scale=0.5,        # Scale augmentation
        shear=0.0,        # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,       # Vertical flip probability
        fliplr=0.5,       # Horizontal flip probability
        mosaic=1.0,       # Mosaic augmentation probability
        mixup=0.1,        # Mixup augmentation probability
        
        # Training settings
        lr0=0.01,         # Initial learning rate
        lrf=0.01,         # Final learning rate
        momentum=0.937,   # SGD momentum
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=3.0,    # Warmup epochs
        warmup_momentum=0.8,  # Warmup momentum
        
        # Loss weights
        box=7.5,          # Box loss gain
        cls=0.5,          # Class loss gain
        dfl=1.5,          # Distribution focal loss gain
        
        # Other settings
        save=True,        # Save checkpoints
        save_period=10,   # Save every N epochs
        plots=True,       # Save training plots
        verbose=True,     # Verbose output
    )
    
    print("\n✅ Training completed!")
    print(f"📁 Best model saved to: {PROJECT_NAME}/train/weights/best.pt")
    
    return results


def validate_model():
    """Validate the trained model."""
    print("\n🔍 Validating model...")
    
    best_model_path = Path(PROJECT_NAME) / 'train' / 'weights' / 'best.pt'
    
    if not best_model_path.exists():
        print("❌ Best model not found!")
        return
    
    model = YOLO(str(best_model_path))
    
    # Validate
    metrics = model.val(data='data.yaml')
    
    print("\n📊 Validation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model():
    """Export model to ONNX format for web deployment."""
    print("\n📦 Exporting model to ONNX format...")
    
    best_model_path = Path(PROJECT_NAME) / 'train' / 'weights' / 'best.pt'
    
    if not best_model_path.exists():
        print("❌ Best model not found!")
        return
    
    model = YOLO(str(best_model_path))
    
    # Export to ONNX
    export_path = model.export(
        format='onnx',
        imgsz=IMG_SIZE,
        simplify=True,
        dynamic=False,
        opset=12
    )
    
    print(f"✅ Model exported to: {export_path}")
    
    # Copy to frontend public folder
    public_model_dir = Path('money/public/models/yolo')
    public_model_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    dest_path = public_model_dir / 'best.onnx'
    shutil.copy(export_path, dest_path)
    
    print(f"✅ Model copied to frontend: {dest_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🎯 YOLO Currency Note Detection Training")
    print("=" * 60)
    
    # Setup
    if not setup_training():
        return
    
    # Train
    train_yolo()
    
    # Validate
    validate_model()
    
    # Export
    export_model()
    
    print("\n" + "=" * 60)
    print("🎉 YOLO training pipeline completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Check training results in:", PROJECT_NAME)
    print("  2. Run: python train_classifier.py")
    print("  3. Test model: python test_yolo.py")


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import ultralytics
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'ultralytics'])
    
    main()