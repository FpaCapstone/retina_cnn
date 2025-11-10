"""
Train a 6-layer CNN for binary normal/abnormal classification.
This matches the documentation specification for Stage 3.

Architecture:
- 6 layers with ReLU activation and Max Pooling
- Binary output (normal/abnormal)
- Decision threshold: ≥0.85 confidence for "Normal"
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "../backend/models")
ASSETS_MODEL_DIR = os.path.join(BASE_DIR, "../assets/images/models")

IMG_SIZE = (224, 224)
MODEL_NAME = "normal_abnormal_classifier.h5"
TFLITE_MODEL_NAME = "normal_abnormal_classifier.tflite"

# Disease categories
DISEASES = ["Normal", "Uveitis", "Conjunctivitis", "Cataract", "Eyelid Drooping"]

def load_dataset():
    """Load dataset and convert to binary normal/abnormal labels."""
    print(f"📥 Loading dataset from: {DATASET_DIR}")
    images = []
    labels = []  # Binary: 1 for Normal, 0 for Abnormal
    
    for disease in DISEASES:
        disease_path = os.path.join(DATASET_DIR, disease)
        if not os.path.exists(disease_path):
            print(f"⚠️ Missing folder for {disease} ({disease_path})")
            continue
        
        # Normal = 1, all others = 0
        binary_label = 1 if disease == "Normal" else 0
        
        for img_file in tqdm(os.listdir(disease_path), desc=f"Loading {disease}"):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(disease_path, img_file)
                try:
                    img = image.load_img(img_path, target_size=IMG_SIZE)
                    img_array = image.img_to_array(img)
                    images.append(img_array)
                    labels.append(binary_label)
                except Exception as e:
                    print(f"⚠️ Could not read {img_file}: {e}")
    
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.float32)
    
    print(f"✅ Loaded {len(images)} images")
    print(f"   Normal: {np.sum(labels == 1)}")
    print(f"   Abnormal: {np.sum(labels == 0)}")
    
    return images, labels

def build_model():
    """
    Build 6-layer CNN with ReLU activation and Max Pooling.
    Architecture as per documentation:
    - Layer 1: Conv2D + ReLU + MaxPooling
    - Layer 2: Conv2D + ReLU + MaxPooling
    - Layer 3: Conv2D + ReLU + MaxPooling
    - Layer 4: Flatten
    - Layer 5: Dense + ReLU + Dropout
    - Layer 6: Output (binary sigmoid)
    """
    model = tf.keras.Sequential([
        # Layer 1: Conv2D + ReLU + MaxPooling
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Layer 2: Conv2D + ReLU + MaxPooling
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Layer 3: Conv2D + ReLU + MaxPooling
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Layer 4: Flatten
        tf.keras.layers.Flatten(),
        
        # Layer 5: Dense + ReLU + Dropout
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Layer 6: Output (binary: normal/abnormal)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Train the 6-layer CNN normal/abnormal classifier."""
    print("🔬 Training 6-Layer CNN Normal/Abnormal Classifier")
    print("=" * 60)
    
    # Load dataset
    images, labels = load_dataset()
    
    if len(images) == 0:
        raise ValueError("❌ No images loaded! Please check your dataset folder.")
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"🧪 Train: {len(X_train)} | Validation: {len(X_val)}")
    
    # Build model
    print("\n🏗️ Building 6-layer CNN model...")
    model = build_model()
    model.summary()
    
    # Train
    print("\n🚀 Starting training...")
    EPOCHS = 15
    BATCH_SIZE = 32
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Evaluate
    print("\n📈 Evaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"✅ Validation Accuracy: {val_accuracy:.4f}")
    print(f"✅ Validation Loss: {val_loss:.4f}")
    
    # Save H5 model to backend only (for server use)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(model_path)
    print(f"✅ Saved H5 model to backend: {model_path}")
    
    # Convert to TFLite for mobile deployment
    print("\n📱 Converting to TFLite format...")
    try:
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert to TFLite
        tflite_model = converter.convert()
        
        # Save TFLite model to backend (for server fallback)
        tflite_path = os.path.join(MODEL_DIR, TFLITE_MODEL_NAME)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ Saved TFLite model to backend: {tflite_path}")
        
        # Save TFLite to assets for mobile app (small, mobile-friendly)
        os.makedirs(ASSETS_MODEL_DIR, exist_ok=True)
        tflite_assets_path = os.path.join(ASSETS_MODEL_DIR, TFLITE_MODEL_NAME)
        with open(tflite_assets_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ Saved TFLite model to assets: {tflite_assets_path}")
        
        # Get model size
        tflite_size_mb = len(tflite_model) / (1024 * 1024)
        h5_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   H5 model size: {h5_size_mb:.2f} MB (backend only)")
        print(f"   TFLite model size: {tflite_size_mb:.2f} MB (backend + assets)")
        print(f"   ✅ TFLite is {h5_size_mb/tflite_size_mb:.1f}x smaller - perfect for mobile!")
    except Exception as e:
        print(f"⚠️  Could not convert to TFLite: {e}")
        print("   H5 model will still be available in backend")
    
    print("\n🎉 Training complete! 6-layer CNN ready for Stage 3.")
    print("   Decision threshold: ≥0.85 confidence for 'Normal' classification")
    print("   TFLite version available for mobile fallback")

if __name__ == "__main__":
    main()

