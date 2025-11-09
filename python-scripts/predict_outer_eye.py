import sys
import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Suppress TensorFlow progress output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

if len(sys.argv) < 3:
    print(json.dumps({"error": "Missing arguments"}), file=sys.stderr)
    print(json.dumps({"error": "Missing arguments"}))
    sys.exit(1)

image_path = sys.argv[1]
model_path = sys.argv[2]

# Define labels in same order as training
DISEASES = ["Normal", "Uveitis", "Conjunctivitis", "Cataract", "Eyelid Drooping"]

try:
    model = tf.keras.models.load_model(model_path)
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

except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
