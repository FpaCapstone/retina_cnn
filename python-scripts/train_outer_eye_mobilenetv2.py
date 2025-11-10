import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from tqdm import tqdm

# -----------------------------
# 🧠 CONFIGURATION
# -----------------------------
# Always resolve dataset path relative to where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "../backend/models")

IMG_SIZE = (224, 224)
MODEL_NAME = "outer_eye_mobilenetv2.h5"
MODEL_TFLITE = "outer_eye_mobilenetv2.tflite"

# Disease categories — make sure these match your folder names exactly
DISEASES = ["Normal", "Uveitis", "Conjunctivitis", "Cataract", "Eyelid Drooping"]

# -----------------------------
# 📂 LOAD DATASET
# -----------------------------
print(f"📥 Loading dataset from: {DATASET_DIR}")
images = []
labels = []

for disease in DISEASES:
    disease_path = os.path.join(DATASET_DIR, disease)
    if not os.path.exists(disease_path):
        print(f"⚠️ Missing folder for {disease} ({disease_path})")
        continue

    for img_file in tqdm(os.listdir(disease_path), desc=f"Loading {disease}"):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(disease_path, img_file)
            try:
                img = image.load_img(img_path, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                images.append(img_array)
                labels.append(disease)
            except Exception as e:
                print(f"⚠️ Could not read {img_file}: {e}")

images = np.array(images, dtype=np.float32) / 255.0  # normalize to [0,1]
labels = np.array(labels)
print(f"✅ Loaded {len(images)} images")

# -----------------------------
# 🧬 ENCODE LABELS
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(labels)

if len(images) == 0:
    raise ValueError("❌ No images loaded! Please check your dataset folder names and paths.")

# Flatten for SMOTE
X_flat = images.reshape(len(images), -1)
y_flat = y_encoded

# -----------------------------
# ⚖️ APPLY SMOTE
# -----------------------------
print("⚖️ Applying SMOTE balancing...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_flat, y_flat)

# Reshape back to image tensors
X_resampled = X_resampled.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 3)
y_resampled_cat = to_categorical(y_resampled)

print(f"✅ After SMOTE: {len(X_resampled)} samples (balanced across {len(DISEASES)} classes)")

# -----------------------------
# ✂️ SPLIT TRAIN/VALIDATION
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled_cat, test_size=0.2, random_state=42
)
print(f"🧪 Train: {len(X_train)} | Validation: {len(X_val)}")

# -----------------------------
# 🧠 BUILD MODEL: MobileNetV2
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze pre-trained layers

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(DISEASES), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 🚀 TRAIN MODEL
# -----------------------------
EPOCHS = 15
BATCH_SIZE = 32

print("\n🚀 Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# -----------------------------
# 📊 VALIDATION RESULTS
# -----------------------------
print("\n📈 Evaluating model...")
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=DISEASES))

# -----------------------------
# 💾 SAVE MODEL
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
model.save(model_path)
print(f"✅ Saved Keras model: {model_path}")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = os.path.join(MODEL_DIR, MODEL_TFLITE)

with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"✅ Saved TFLite model: {tflite_path}")

print("\n🎉 Training complete — MobileNetV2 model ready for deployment!")
