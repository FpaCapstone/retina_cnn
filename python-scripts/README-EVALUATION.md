# Model Evaluation Script

## Overview

The `evaluate_model.py` script provides comprehensive evaluation of your trained eye disease detection model. It loads your trained model, runs predictions on test data, and generates detailed metrics and visualizations.

## Features

✅ **Loads trained model** from `backend/models/outer_eye_mobilenetv2.h5`  
✅ **Loads test dataset** from `datasets/` directory  
✅ **Runs predictions** on test images  
✅ **Extracts true labels** for comparison  
✅ **Generates classification report** with precision, recall, F1-score  
✅ **Creates confusion matrices** (normalized and absolute)  
✅ **Saves all results** to timestamped directory  

## Requirements

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- imblearn

## Usage

### Basic Usage (Test Split)

Evaluate on a test split (20% of dataset):

```bash
cd python-scripts
python evaluate_model.py
```

### Evaluate on Full Dataset

Evaluate on the entire dataset:

```bash
python evaluate_model.py --full-dataset
```

### Custom Batch Size

Use a different batch size for predictions:

```bash
python evaluate_model.py --batch-size 64
```

### Combined Options

```bash
python evaluate_model.py --full-dataset --batch-size 64
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--full-dataset` | flag | False | Evaluate on full dataset instead of test split |
| `--batch-size` | int | 32 | Batch size for model predictions |

## Output Files

The script creates a timestamped directory in `evaluation_results/` containing:

### Text Reports

#### 1. `classification_report.json`
JSON file with detailed per-class metrics:
- Precision
- Recall
- F1-score
- Support (number of samples)
- Macro/micro averages

#### 2. `detailed_metrics.txt`
Human-readable text report with:
- Overall accuracy
- Per-class metrics table
- Full classification report
- Confusion matrix in text format

### Visualizations

#### 3. `per_class_metrics.png`
**Bar chart showing per-class metrics:**
- Precision (blue bars)
- Recall (green bars)
- F1-Score (red bars)
- Values displayed on each bar
- Easy comparison across disease classes

#### 4. `confusion_matrix_absolute.png`
**Confusion matrix with absolute counts:**
- Shows actual number of predictions
- Useful for understanding raw prediction counts
- Format: Integer values

#### 5. `confusion_matrix_normalized.png`
**Normalized confusion matrix (decimal):**
- Values normalized to 0-1 range
- Shows proportion of predictions per class
- Format: Decimal values (0.000 - 1.000)

#### 6. `confusion_matrix_percent.png`
**Normalized confusion matrix (percent view):**
- Same as normalized but displayed as percentages
- Easier to read (0% - 100%)
- Format: Percentage values with 1 decimal place
- **Most intuitive view for understanding model performance**

## Output Structure

```
evaluation_results/
└── evaluation_20240101_120000/
    ├── classification_report.json          # JSON metrics
    ├── detailed_metrics.txt                 # Text report
    ├── per_class_metrics.png                # Bar chart: Precision/Recall/F1
    ├── confusion_matrix_absolute.png        # Confusion matrix (counts)
    ├── confusion_matrix_normalized.png      # Confusion matrix (0-1)
    └── confusion_matrix_percent.png         # Confusion matrix (0-100%)
```

## Example Output

```
================================================================================
MODEL EVALUATION SCRIPT
================================================================================
Evaluation Date: 2024-01-01 12:00:00

📥 Loading dataset from: /path/to/datasets
✅ Loaded 2298 images

📊 Creating test split from dataset...
✅ Test set: 460 images
📊 Class distribution:
   Normal: 92
   Uveitis: 92
   Conjunctivitis: 92
   Cataract: 92
   Eyelid Drooping: 92

📥 Loading model from: /path/to/backend/models/outer_eye_mobilenetv2.h5
✅ Model loaded successfully

🔮 Running predictions on 460 images...
✅ Predictions complete

📊 Overall Accuracy: 0.8565 (85.65%)

📋 Classification Report:
              precision    recall  f1-score   support

      Normal       0.89      0.91      0.90        92
     Uveitis       0.82      0.78      0.80        92
Conjunctivitis       0.85      0.88      0.86        92
     Cataract       0.88      0.85      0.86        92
Eyelid Drooping       0.84      0.82      0.83        92

    accuracy                           0.86       460
   macro avg       0.86      0.85      0.85       460
weighted avg       0.86      0.86      0.85       460

💾 Saved classification report to: evaluation_results/evaluation_20240101_120000/classification_report.json
💾 Saved detailed metrics to: evaluation_results/evaluation_20240101_120000/detailed_metrics.txt
📊 Creating per-class metrics plot...
💾 Saved per-class metrics plot to: evaluation_results/evaluation_20240101_120000/per_class_metrics.png
📊 Creating confusion matrix (absolute counts)...
💾 Saved absolute confusion matrix to: evaluation_results/evaluation_20240101_120000/confusion_matrix_absolute.png
📊 Creating confusion matrix (normalized - decimal)...
💾 Saved normalized confusion matrix to: evaluation_results/evaluation_20240101_120000/confusion_matrix_normalized.png
📊 Creating confusion matrix (normalized - percent)...
💾 Saved percent confusion matrix to: evaluation_results/evaluation_20240101_120000/confusion_matrix_percent.png
✅ All visualizations created successfully!

================================================================================
✅ EVALUATION COMPLETE
================================================================================
📁 Results saved to: evaluation_results/evaluation_20240101_120000
📊 Overall Accuracy: 0.8565 (85.65%)
```

## Metrics Explained

### Accuracy
Overall percentage of correct predictions across all classes.

### Precision
Percentage of positive predictions that were actually correct.
- High precision = fewer false positives

### Recall
Percentage of actual positives that were correctly identified.
- High recall = fewer false negatives

### F1-Score
Harmonic mean of precision and recall.
- Balanced metric that considers both precision and recall

### Confusion Matrix
Shows where the model makes mistakes:
- Rows = True labels
- Columns = Predicted labels
- Diagonal = Correct predictions
- Off-diagonal = Misclassifications

## Model Locations & Fallback Strategy

The script automatically searches for the model with **automatic fallback**:

### Priority Order:
1. **Backend** (Primary): `backend/models/outer_eye_mobilenetv2.h5`
   - Used when available (server/backend deployment)
   
2. **Assets** (Fallback): `assets/images/models/outer_eye_mobilenetv2.h5`
   - Automatically used if backend model is missing
   - Useful for offline/mobile scenarios

### How It Works:
- ✅ **Backend model exists** → Uses backend model
- ⚠️ **Backend missing, Assets exists** → Automatically falls back to assets model
- ❌ **Both missing** → Shows detailed error with solution

The script will:
1. Check backend first
2. If backend is missing, automatically use assets (no manual intervention needed)
3. Show clear messages about which model is being used

**Note:** The script requires a `.h5` (Keras) model file. The `.tflite` model in assets is for mobile deployment and cannot be used for evaluation.

## Troubleshooting

### Model Not Found
```
❌ Model not found!
Checked locations:
  - backend/models/outer_eye_mobilenetv2.h5
  - assets/images/models/outer_eye_mobilenetv2.h5
```
**Solution:** 
- Train the model first using `train_outer_eye_mobilenetv2.py`
- Or specify a custom model path: `python evaluate_model.py --model-path /path/to/model.h5`

### Dataset Not Found
```
❌ No images loaded! Please check your dataset folder names and paths.
```
**Solution:** Ensure your dataset is in `python-scripts/datasets/` with folders:
- `Normal/`
- `Uveitis/`
- `Conjunctivitis/`
- `Cataract/`
- `Eyelid Drooping/`

### Memory Issues
If you run out of memory, reduce batch size:
```bash
python evaluate_model.py --batch-size 16
```

## Integration with Training

This evaluation script is designed to work with the model trained by `train_outer_eye_mobilenetv2.py`. The script expects:

1. **Model format:** Keras `.h5` file
2. **Model architecture:** MobileNetV2 with 5-class output
3. **Input size:** 224x224x3 RGB images
4. **Classes:** Normal, Uveitis, Conjunctivitis, Cataract, Eyelid Drooping

## Next Steps

After evaluation, you can:

1. **Analyze confusion matrix** to identify which classes are confused
2. **Review per-class metrics** to see which diseases need more training data
3. **Compare results** across different model versions
4. **Fine-tune model** based on weak areas identified

## Notes

- The script uses the same random seed (42) as training for consistent splits
- Test split is 20% of the dataset by default
- All images are normalized to [0, 1] range
- Results are saved with timestamps for easy comparison

