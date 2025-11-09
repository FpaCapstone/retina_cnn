# ML-Based Image Quality Classifier

## Overview

An optional ML-based image quality classifier that can learn from your data to detect blur, glare, and crop issues more accurately than traditional computer vision methods.

## Why ML for Quality Assessment?

### Traditional CV (Current Default)
- ✅ **Fast**: ~50ms inference
- ✅ **No training needed**: Works immediately
- ✅ **Interpretable**: Clear thresholds
- ❌ **Limited**: May miss edge cases
- ❌ **Manual tuning**: Thresholds need adjustment

### ML-Based (Optional Enhancement)
- ✅ **More accurate**: Learns from your data
- ✅ **Adaptive**: Improves with more training data
- ✅ **Complex patterns**: Can detect subtle issues
- ❌ **Requires training**: Needs labeled data
- ❌ **Slower**: ~100-200ms inference
- ❌ **Less interpretable**: Black box model

## Hybrid Approach (Recommended)

The enhanced pipeline uses a **hybrid approach**:
1. **Try ML classifier first** (if trained model exists)
2. **Fallback to CV methods** (if ML unavailable or fails)

This gives you the best of both worlds:
- ML accuracy when available
- CV reliability as fallback
- No breaking changes

## Training the ML Quality Classifier

### Step 1: Auto-Label Your Dataset

The easiest way to create training data is to auto-label your existing dataset:

```bash
cd python-scripts
python train_quality_classifier.py --auto-label
```

This will:
1. Process all images in your main dataset
2. Use traditional CV methods to assess quality
3. Create labeled folders: `good/`, `blurry/`, `glare/`, `poor_crop/`, `multiple_issues/`
4. Copy images to appropriate quality folders

### Step 2: Review and Refine Labels

Manually review the auto-labeled images and move any mislabeled ones:

```
datasets/quality_labels/
├── good/              # High quality images
├── blurry/            # Blurry images
├── glare/             # Overexposed images
├── poor_crop/         # Poorly cropped images
└── multiple_issues/   # Images with multiple problems
```

### Step 3: Train the Model

```bash
python train_quality_classifier.py
```

This will:
- Load quality-labeled images
- Train a MobileNetV2-based classifier
- Save model to `backend/models/quality_classifier.h5`
- Also save to `assets/images/models/` for mobile use

### Step 4: Use ML Quality Assessment

Once trained, the enhanced pipeline will automatically use the ML classifier:

```python
# In enhanced_pipeline.py, it automatically tries ML first
quality_report = assess_image_quality(image_path, use_ml=True)
```

## Model Architecture

```
MobileNetV2 (pre-trained on ImageNet)
  ↓
GlobalAveragePooling2D
  ↓
Dense(128, relu) + Dropout(0.3)
  ↓
Dense(64, relu) + Dropout(0.2)
  ↓
Dense(5, softmax)  # 5 quality classes
```

## Quality Classes

1. **good** - High quality, ready for analysis
2. **blurry** - Motion blur or focus issues
3. **glare** - Overexposure or bright reflections
4. **poor_crop** - Eye not properly centered/visible
5. **multiple_issues** - Combination of problems

## Usage

### In Enhanced Pipeline

The enhanced pipeline automatically uses ML if available:

```python
# ML will be used if model exists
results = run_enhanced_pipeline(image_path)
```

### Disable ML (Use CV Only)

```python
# Force CV methods only
results = run_enhanced_pipeline(image_path, enable_stages={
    'quality_check': True,
    'use_ml_quality': False  # Use CV instead
})
```

### Standalone Quality Assessment

```python
from enhanced_pipeline import assess_image_quality

# Try ML first, fallback to CV
quality = assess_image_quality("image.jpg", use_ml=True)

# Force CV only
quality = assess_image_quality("image.jpg", use_ml=False)
```

## Expected Performance

### Training Data Requirements

| Images per Class | Expected Accuracy |
|-----------------|-------------------|
| 50+ | 75-80% |
| 100+ | 80-85% |
| 200+ | 85-90% |
| 500+ | 90-95% |

### Inference Speed

- **ML Classifier**: ~100-200ms (on CPU)
- **CV Methods**: ~50ms
- **Difference**: ~50-150ms (negligible for user experience)

## Comparison: ML vs CV

### When ML is Better

- ✅ **Complex patterns**: Subtle blur, mixed issues
- ✅ **Domain-specific**: Trained on your eye images
- ✅ **Adaptive**: Improves with more data
- ✅ **Edge cases**: Handles unusual scenarios

### When CV is Better

- ✅ **Speed**: Faster inference
- ✅ **Interpretability**: Clear thresholds
- ✅ **No training**: Works immediately
- ✅ **Reliability**: Predictable behavior

## Recommendations

### Start with CV (Current)

The traditional CV methods work well for most cases. Use them as your baseline.

### Add ML When You Have Data

Once you have:
- 200+ quality-labeled images
- Time to review/refine labels
- Need for higher accuracy

Then train the ML classifier for incremental improvements.

### Hybrid is Best

The current implementation uses both:
- ML when available (better accuracy)
- CV as fallback (reliability)

This gives you the best of both worlds without complexity.

## Troubleshooting

### "Quality dataset directory not found"

Run auto-labeling first:
```bash
python train_quality_classifier.py --auto-label
```

### "Model not found"

Train the model:
```bash
python train_quality_classifier.py
```

### "Low accuracy"

- Review and refine labels
- Add more training images
- Ensure balanced classes
- Check for mislabeled images

### "ML assessment failed"

The pipeline automatically falls back to CV methods. Check:
- Model file exists and is valid
- Image format is supported
- Sufficient memory available

## Future Enhancements

- [ ] Multi-task learning (quality + disease classification)
- [ ] Active learning (model suggests which images to label)
- [ ] Transfer learning from other quality datasets
- [ ] Real-time quality feedback during capture
- [ ] Quality score calibration

