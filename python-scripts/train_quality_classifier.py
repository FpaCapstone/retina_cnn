"""
Train ML-based Image Quality Classifier
Learns to detect blur, glare, and crop issues from labeled data
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
QUALITY_DATASET_DIR = os.path.join(DATASET_DIR, "quality_labels")
MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "models")
ASSETS_MODEL_DIR = os.path.join(PROJECT_ROOT, "assets", "images", "models")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Quality classes
QUALITY_CLASSES = ["good", "blurry", "glare", "poor_crop", "multiple_issues"]

# ============================================
# DATA LOADING
# ============================================

def load_quality_dataset():
    """
    Load quality-labeled images.
    
    Expected structure:
    quality_labels/
    ├── good/
    │   ├── image1.jpg
    │   └── ...
    ├── blurry/
    │   ├── image2.jpg
    │   └── ...
    ├── glare/
    │   └── ...
    ├── poor_crop/
    │   └── ...
    └── multiple_issues/
        └── ...
    """
    print(f"📥 Loading quality dataset from: {QUALITY_DATASET_DIR}")
    
    images = []
    labels = []
    
    if not os.path.exists(QUALITY_DATASET_DIR):
        print(f"⚠️  Quality dataset directory not found: {QUALITY_DATASET_DIR}")
        print("\n📝 To create a quality dataset:")
        print("1. Create folders: good/, blurry/, glare/, poor_crop/, multiple_issues/")
        print("2. Manually label images or use auto-labeling script")
        print("3. Run this training script again")
        return None, None
    
    for quality_class in QUALITY_CLASSES:
        class_path = os.path.join(QUALITY_DATASET_DIR, quality_class)
        if not os.path.exists(class_path):
            print(f"⚠️  Missing folder: {quality_class}")
            continue
        
        class_images = []
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, IMG_SIZE)
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    class_images.append(img_normalized)
                    labels.append(quality_class)
                except Exception as e:
                    print(f"⚠️  Error loading {img_file}: {e}")
        
        images.extend(class_images)
        print(f"✅ Loaded {len(class_images)} images for class: {quality_class}")
    
    if len(images) == 0:
        print("❌ No images loaded!")
        return None, None
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\n✅ Total loaded: {len(images)} images")
    return images, labels

def auto_label_quality_from_main_dataset():
    """
    Automatically label images from main dataset using traditional CV methods.
    This creates a training dataset for the ML quality classifier.
    """
    print("\n🤖 Auto-labeling images using traditional CV methods...")
    print("This will create quality labels for training the ML classifier.")
    
    from enhanced_pipeline import assess_image_quality
    
    main_dataset_dir = os.path.join(DATASET_DIR, "outer_eye")
    if not os.path.exists(main_dataset_dir):
        print(f"⚠️  Main dataset not found: {main_dataset_dir}")
        return
    
    # Create quality labels directory
    os.makedirs(QUALITY_DATASET_DIR, exist_ok=True)
    for quality_class in QUALITY_CLASSES:
        os.makedirs(os.path.join(QUALITY_DATASET_DIR, quality_class), exist_ok=True)
    
    labeled_count = 0
    
    # Process each disease folder
    for disease_folder in os.listdir(main_dataset_dir):
        disease_path = os.path.join(main_dataset_dir, disease_folder)
        if not os.path.isdir(disease_path):
            continue
        
        print(f"\n📂 Processing {disease_folder}...")
        
        for img_file in os.listdir(disease_path):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            img_path = os.path.join(disease_path, img_file)
            
            try:
                quality_report = assess_image_quality(img_path)
                
                # Determine quality class
                issues = quality_report['issues']
                if len(issues) == 0:
                    quality_class = "good"
                elif len(issues) > 1:
                    quality_class = "multiple_issues"
                elif "blur" in issues:
                    quality_class = "blurry"
                elif "glare" in issues:
                    quality_class = "glare"
                elif "crop" in issues:
                    quality_class = "poor_crop"
                else:
                    quality_class = "good"
                
                # Copy image to quality class folder
                import shutil
                dest_path = os.path.join(QUALITY_DATASET_DIR, quality_class, img_file)
                shutil.copy2(img_path, dest_path)
                labeled_count += 1
                
                if labeled_count % 100 == 0:
                    print(f"  ✅ Labeled {labeled_count} images...")
                    
            except Exception as e:
                print(f"⚠️  Error processing {img_file}: {e}")
    
    print(f"\n✅ Auto-labeled {labeled_count} images")
    print(f"📁 Quality labels saved to: {QUALITY_DATASET_DIR}")

# ============================================
# MODEL ARCHITECTURE
# ============================================

def build_quality_classifier():
    """
    Build a lightweight CNN for quality classification.
    Uses MobileNetV2 as base for transfer learning.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(QUALITY_CLASSES), activation='softmax', name='quality_output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# TRAINING
# ============================================

def train_quality_classifier():
    """Train the quality classifier model."""
    print("\n" + "=" * 80)
    print("🚀 TRAINING IMAGE QUALITY CLASSIFIER")
    print("=" * 80)
    
    # Load dataset
    images, labels = load_quality_dataset()
    
    if images is None:
        print("\n💡 Tip: Run with --auto-label to create quality labels from main dataset")
        return
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    print(f"\n📊 Dataset distribution:")
    for i, quality_class in enumerate(QUALITY_CLASSES):
        count = np.sum(labels_encoded == i)
        print(f"  {quality_class}: {count} images")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_encoded,
        test_size=0.2,
        random_state=42,
        stratify=labels_encoded
    )
    
    print(f"\n📦 Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Build model
    print("\n🏗️  Building model...")
    model = build_quality_classifier()
    model.summary()
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train
    print("\n🎯 Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "quality_classifier.h5")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save label encoder
    encoder_path = os.path.join(MODEL_DIR, "quality_label_encoder.json")
    with open(encoder_path, 'w') as f:
        json.dump({
            'classes': label_encoder.classes_.tolist(),
            'mapping': {int(k): v for k, v in enumerate(label_encoder.classes_)}
        }, f, indent=2)
    print(f"💾 Label encoder saved to: {encoder_path}")
    
    # Also save to assets
    assets_model_path = os.path.join(ASSETS_MODEL_DIR, "quality_classifier.h5")
    os.makedirs(ASSETS_MODEL_DIR, exist_ok=True)
    model.save(assets_model_path)
    print(f"💾 Model also saved to: {assets_model_path}")
    
    return model, label_encoder

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--auto-label":
        auto_label_quality_from_main_dataset()
    else:
        train_quality_classifier()

