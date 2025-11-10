# TFLite Model Evaluation Guide

## Overview

This script (`evaluate_tflite_model.py`) evaluates the TensorFlow Lite (TFLite) model used for on-device inference in the mobile app. It provides the same comprehensive evaluation metrics as the H5 model evaluation, but specifically for the TFLite format.

## Purpose

- **Evaluate TFLite model accuracy**: Compare TFLite model performance against the H5 model
- **Verify model conversion**: Ensure the TFLite model maintains similar accuracy after conversion
- **Performance metrics**: Get detailed metrics (precision, recall, F1-score) for the TFLite model
- **Visualization**: Generate confusion matrices and per-class metrics plots

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `tensorflow` (for TFLite interpreter)
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `pillow`

## Usage

### Basic Usage (Auto-detect TFLite Model)

```bash
python evaluate_tflite_model.py
```

This will:
1. Automatically find the TFLite model in `backend/models/` or `assets/images/models/`
2. Load the test dataset
3. Run predictions using the TFLite interpreter
4. Generate evaluation metrics and visualizations

### Evaluate on Full Dataset

```bash
python evaluate_tflite_model.py --full-dataset
```

### Use Custom TFLite Model Path

```bash
python evaluate_tflite_model.py --model-path /path/to/your/model.tflite
```

## Model Search Strategy

The script searches for the TFLite model in this order:

1. **Primary**: `backend/models/outer_eye_mobilenetv2.tflite`
2. **Fallback**: `assets/images/models/outer_eye_mobilenetv2.tflite`

## Output Files

The script generates the following files in `evaluation_results/evaluation_tflite_YYYYMMDD_HHMMSS/`:

### Metrics Files
- **`classification_report.json`**: JSON format classification report
- **`detailed_metrics.txt`**: Detailed text report (used by `plot_metrics.py`)

### Visualization Files
- **`confusion_matrix.png`**: Default confusion matrix (normalized percent view)
- **`confusion_matrix_absolute.png`**: Confusion matrix with absolute counts
- **`confusion_matrix_normalized.png`**: Confusion matrix normalized (decimal)
- **`confusion_matrix_percent.png`**: Confusion matrix normalized (percent)
- **`per_class_metrics.png`**: Per-class precision, recall, and F1-score bar chart

## Key Differences from H5 Evaluation

### 1. Model Loading
- **H5**: Uses `tf.keras.models.load_model()`
- **TFLite**: Uses `tf.lite.Interpreter()` with tensor allocation

### 2. Inference
- **H5**: Can process batches (e.g., batch_size=32)
- **TFLite**: Processes one image at a time (batch_size=1)

### 3. Performance
- **H5**: Slower, but supports batching
- **TFLite**: Faster per-image inference, optimized for mobile

## Expected Results

### Accuracy Comparison
- **H5 Model**: Typically 94-95% accuracy
- **TFLite Model**: Should be very similar (within 0.1-0.5% difference)

### Why Small Differences?
- **Quantization**: TFLite models may use quantization (int8) which can cause minor accuracy loss
- **Optimization**: TFLite optimizations may affect numerical precision slightly

## Troubleshooting

### Error: "TFLite model not found"
**Solution**: Ensure the TFLite model exists in one of these locations:
- `backend/models/outer_eye_mobilenetv2.tflite`
- `assets/images/models/outer_eye_mobilenetv2.tflite`

### Error: "Failed to load TFLite model"
**Solution**: 
1. Verify the TFLite file is not corrupted
2. Check that TensorFlow version supports TFLite (TensorFlow 2.x)
3. Ensure the model was converted correctly from H5 to TFLite

### Error: "No images loaded"
**Solution**: 
1. Check that the dataset directory exists: `python-scripts/datasets/`
2. Verify folder names match disease classes exactly:
   - `Normal/`
   - `Uveitis/`
   - `Conjunctivitis/`
   - `Cataract/`
   - `Eyelid Drooping/`

## Integration with Plot Metrics

The generated `detailed_metrics.txt` file can be used with `plot_metrics.py` in JupyterLab:

```python
from plot_metrics import plot_from_evaluation_results

# Plot TFLite evaluation results
plot_from_evaluation_results('evaluation_results/evaluation_tflite_YYYYMMDD_HHMMSS')
```

## Comparison with H5 Model

To compare H5 and TFLite model performance:

1. **Evaluate H5 model**:
   ```bash
   python evaluate_model.py
   ```

2. **Evaluate TFLite model**:
   ```bash
   python evaluate_tflite_model.py
   ```

3. **Compare results**:
   - Check accuracy in `detailed_metrics.txt` files
   - Compare confusion matrices
   - Review per-class metrics

## Notes

- **TFLite is optimized for mobile**: The TFLite model is designed for on-device inference with lower memory usage and faster inference
- **Batch processing**: TFLite processes images one at a time, which is slower for evaluation but matches mobile app usage
- **Quantization**: If the TFLite model uses quantization, there may be minor accuracy differences from the H5 model
- **File size**: TFLite models are typically smaller than H5 models, making them ideal for mobile deployment

## Example Output

```
================================================================================
TFLITE MODEL EVALUATION SCRIPT
================================================================================
Evaluation Date: 2025-11-10 10:30:00

🔍 TFLite Model Search Strategy:
  1. Primary: /path/to/backend/models
  2. Fallback: /path/to/assets/images/models

✅ Backend TFLite model available: /path/to/backend/models/outer_eye_mobilenetv2.tflite

📥 Loading dataset from: python-scripts/datasets
✅ Loaded 1000 images

📥 Loading TFLite model from: /path/to/backend/models/outer_eye_mobilenetv2.tflite
✅ TFLite model loaded successfully
📋 Input shape: [1, 224, 224, 3]
📋 Output shape: [1, 5]

🔮 Running predictions on 200 images using TFLite...
✅ Predictions complete

📊 Overall Accuracy: 0.9450 (94.50%)

✅ TFLITE EVALUATION COMPLETE
📁 Results saved to: evaluation_results/evaluation_tflite_20251110_103000
```

## Next Steps

1. **Compare with H5**: Run both evaluations and compare results
2. **Visualize**: Use `plot_metrics.py` to create visualizations
3. **Optimize**: If accuracy is lower, consider retraining or adjusting quantization
4. **Deploy**: Use the TFLite model in the mobile app for on-device inference

