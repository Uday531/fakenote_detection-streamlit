import tensorflow as tf
from pathlib import Path
import json
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIGURATION
# ==============================
# STANDARDIZED LIST
FEATURES = [
    "emblem", 
    "watermark", 
    "denomination_number", 
    "denomination_text", 
    "rbi_seal", 
    "security_features"
]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
CLASSIFIER_DIR = Path("dataset/classifier")
MODELS_DIR = Path("models/classifier")
# ==============================

def create_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_data_generators(feature):
    train_dir = CLASSIFIER_DIR / "train" / feature
    val_dir = CLASSIFIER_DIR / "val" / feature
    
    if not train_dir.exists():
        print(f"❌ DATA MISSING: Could not find folder {train_dir}")
        return None, None

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator

def train_feature_classifier(feature):
    print(f"\n{'='*60}")
    print(f"🎯 Training: {feature.upper()}")
    
    train_gen, val_gen = create_data_generators(feature)
    if not train_gen: return None, None
    
    if train_gen.samples == 0:
        print(f"❌ No images found in {feature} folder.")
        return None, None
        
    model = create_model()
    
    feature_model_dir = MODELS_DIR / feature
    feature_model_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(feature_model_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    return model, history

def main():
    print("🎯 Currency Note Classifier Training")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for feature in FEATURES:
        try:
            model, history = train_feature_classifier(feature)
        except Exception as e:
            print(f"❌ Error training {feature}: {e}")

if __name__ == "__main__":
    main()