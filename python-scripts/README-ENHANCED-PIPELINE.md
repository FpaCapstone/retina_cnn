# Enhanced 5-Stage Pipeline for Eye Disease Detection

## Overview

The Enhanced Pipeline improves accuracy for camera-captured images by implementing a 5-stage analysis process that addresses common issues with smartphone photography.

## Pipeline Stages

### 1️⃣ Image Quality AI
**Purpose:** Detects image quality issues before analysis

**Checks:**
- **Blur Detection** - Uses Laplacian variance to detect motion blur or focus issues
- **Glare Detection** - Identifies overexposure or bright reflections
- **Crop Issues** - Verifies eye is properly centered and visible

**Output:**
- Quality score (0-1)
- List of detected issues
- Recommendation: `proceed` or `retake`

**Thresholds:**
- Blur: Laplacian variance < 100.0
- Glare: Average brightness > 0.3
- Crop: Eye coverage < 70%

### 2️⃣ Preprocessing Enhancer
**Purpose:** Automatically fixes detected quality issues

**Enhancements:**
- **Contrast Enhancement** - CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Sharpness Enhancement** - Unsharp masking for blur correction
- **Glare Reduction** - Gamma correction to reduce overexposure

**Applied When:**
- Blur detected → Sharpness enhancement
- Glare detected → Glare reduction
- Always → Contrast enhancement (subtle)

### 3️⃣ Normal-vs-Abnormal AI
**Purpose:** Quick binary filter to identify healthy eyes

**How It Works:**
- Uses the existing model's "Normal" class probability
- If Normal probability > 70% → Skip disease classification
- If Normal probability ≤ 70% → Proceed to disease classification

**Benefits:**
- Faster processing for healthy eyes
- Reduces false positives
- Better user experience

### 4️⃣ Disease Classifier
**Purpose:** Multi-class disease classification (existing Retina CNN)

**Classes:**
- Normal
- Uveitis
- Conjunctivitis
- Cataract
- Eyelid Drooping

**Uses:** Your existing trained MobileNetV2 model

### 5️⃣ Confidence/Consistency Validator
**Purpose:** Validates prediction reliability before final output

**Checks:**
- **Quality-Adjusted Confidence** - Adjusts confidence based on image quality
- **Consistency Score** - Checks if top predictions are close (uncertainty)
- **Reliability Threshold** - Final confidence > 60%, quality > 50%, consistency > 50%

**Output:**
- Final confidence score
- Reliability flag
- Recommendation: `accept` or `retake`

## Usage

### Python Script

```bash
# Run full pipeline
python enhanced_pipeline.py path/to/image.jpg

# Disable specific stages
python enhanced_pipeline.py path/to/image.jpg --disable-stage preprocessing
python enhanced_pipeline.py path/to/image.jpg --disable-stage normal_filter
```

### Backend API (tRPC)

```typescript
// Use enhanced pipeline
const result = await trpcClient.detection.analyzeEnhanced.mutate({
  imageUri: imageUri,
  enableStages: {
    quality_check: true,
    preprocessing: true,
    normal_filter: true,
    disease_classification: true,
    validation: true,
  }
});
```

### Frontend Integration

```typescript
import { analyzeEyeImageEnhanced } from '@/utils/ml-analysis-enhanced';

// Use enhanced analysis
const result = await analyzeEyeImageEnhanced(imageUri, {
  enableQualityCheck: true,
  enablePreprocessing: true,
  enableNormalFilter: true,
  enableDiseaseClassification: true,
  enableValidation: true,
});
```

## Output Format

```json
{
  "stages": {
    "quality": {
      "quality_score": 0.85,
      "blur_score": 150.5,
      "is_blurry": false,
      "glare_score": 0.25,
      "has_glare": false,
      "coverage_ratio": 0.92,
      "has_crop_issue": false,
      "issues": [],
      "needs_preprocessing": false,
      "recommendation": "proceed"
    },
    "preprocessing": {
      "preprocessing_applied": []
    },
    "normal_filter": {
      "result": "abnormal",
      "confidence": 0.75,
      "skip_disease_classification": false
    },
    "disease_classification": {
      "predicted_disease": "Cataract",
      "confidence": 0.88,
      "probabilities": {
        "Normal": 0.05,
        "Uveitis": 0.02,
        "Conjunctivitis": 0.03,
        "Cataract": 0.88,
        "Eyelid Drooping": 0.02
      }
    },
    "validation": {
      "original_confidence": 0.88,
      "quality_adjusted_confidence": 0.75,
      "final_confidence": 0.82,
      "consistency_score": 0.93,
      "is_reliable": true,
      "recommendation": "accept"
    }
  },
  "final_prediction": {
    "predicted_disease": "Cataract",
    "confidence": 0.88,
    "probabilities": {...}
  },
  "recommendation": "accept",
  "final_confidence": 0.82
}
```

## Benefits

### Improved Accuracy
- ✅ Quality checks prevent analysis of poor images
- ✅ Preprocessing fixes common camera issues
- ✅ Normal filter reduces false positives
- ✅ Validation ensures reliable predictions

### Better User Experience
- ✅ Clear feedback on image quality
- ✅ Automatic image enhancement
- ✅ Recommendations to retake if needed
- ✅ Faster processing for normal eyes

### Modular Design
- ✅ Each stage can be enabled/disabled
- ✅ Easy to tune thresholds
- ✅ Can add new stages easily

## Configuration

### Quality Thresholds

Edit in `enhanced_pipeline.py`:

```python
BLUR_THRESHOLD = 100.0      # Laplacian variance
GLARE_THRESHOLD = 0.3       # Brightness (0-1)
MIN_CROP_RATIO = 0.7        # Eye coverage ratio
```

### Normal Filter Threshold

```python
is_normal = normal_prob > 0.7  # In create_normal_vs_abnormal_classifier()
```

### Validation Thresholds

```python
is_reliable = (
    final_confidence > 0.6 and
    quality_score > 0.5 and
    consistency_score > 0.5
)
```

## Integration with Existing Code

The enhanced pipeline is **backward compatible**. You can:

1. **Use enhanced pipeline** - Call `analyzeEyeImageEnhanced()`
2. **Use standard pipeline** - Call `analyzeEyeImage()` (existing)
3. **Hybrid approach** - Try enhanced first, fallback to standard

## Performance

- **Stage 1 (Quality):** ~50ms
- **Stage 2 (Preprocessing):** ~100ms (if needed)
- **Stage 3 (Normal Filter):** ~200ms
- **Stage 4 (Disease Classification):** ~500ms
- **Stage 5 (Validation):** ~10ms

**Total:** ~860ms (vs ~500ms for standard)

**Note:** Stage 3 can skip Stage 4 for normal eyes, saving ~500ms.

## Troubleshooting

### "Model not found"
- Ensure model exists in `backend/models/` or `assets/images/models/`
- Check model name matches `outer_eye_mobilenetv2.h5`

### "Image quality too low"
- Check blur threshold (may be too strict)
- Verify image is properly focused
- Ensure good lighting

### "Low confidence"
- Image may need retaking
- Check quality score in results
- Review preprocessing applied

## Future Enhancements

- [ ] Adaptive thresholds based on device camera quality
- [ ] Eye region detection and cropping
- [ ] Multi-angle consistency checking
- [ ] Temporal consistency (video analysis)
- [ ] User feedback loop for quality improvements

