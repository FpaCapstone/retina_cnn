import sys
import json
import numpy as np
import os
import warnings

# Suppress all warnings before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors only
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Disable CUDA warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only to avoid CUDA warnings

from tensorflow.keras.preprocessing import image

if len(sys.argv) < 3:
    print(json.dumps({"error": "Missing arguments"}), file=sys.stderr)
    print(json.dumps({"error": "Missing arguments"}))
    sys.exit(1)

image_path = sys.argv[1]
model_path = sys.argv[2]

# Define labels in same order as training
DISEASES = ["Normal", "Uveitis", "Conjunctivitis", "Cataract", "Eyelid Drooping"]

try:
    # Redirect stderr to suppress TensorFlow warnings during model loading
    import io
    import contextlib
    
    # Suppress stderr during model loading
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        finally:
            sys.stderr = old_stderr
    
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Suppress verbose output during prediction
    predictions = model.predict(img_array, verbose=0)[0]
    pred_index = np.argmax(predictions)
    confidence = float(predictions[pred_index])
    result = {
        "prediction": DISEASES[pred_index],
        "confidence": confidence,
    }
    print(json.dumps(result))
    sys.exit(0)  # Explicitly exit with success code

except Exception as e:
    error_msg = str(e)
    # Only print actual errors, not warnings
    if "CUDA" not in error_msg and "cuda" not in error_msg and "GPU" not in error_msg:
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        print(json.dumps({"error": error_msg}))
    sys.exit(1)
