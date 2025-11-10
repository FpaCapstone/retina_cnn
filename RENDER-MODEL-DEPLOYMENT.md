# Render Model Deployment Guide

## Overview

Since model files exceed GitHub's 100 MB limit, they need to be deployed to Render separately from the Git repository.

## Models Needed for Enhanced Pipeline

1. **`normal_abnormal_classifier.h5`** (127.87 MB) - 6-layer CNN for Stage 3
2. **`outer_eye_mobilenetv2.h5`** (13 MB) - Main disease classifier for Stage 4
3. **`outer_eye_mobilenetv2.tflite`** (9.7 MB) - Optional, for mobile fallback

## Deployment Options

### Option 1: Render File System (Current)

**Models are stored directly on Render's file system**

1. **After Git deployment**:
   - Render clones your Git repo (without models)
   - Models need to be uploaded separately

2. **Upload models to Render**:
   ```bash
   # Using Render CLI or web interface
   # Or SSH into Render instance and upload files
   ```

3. **Verify models exist**:
   - Check `backend/models/` directory on Render
   - Verify files: `ls backend/models/`

### Option 2: Cloud Storage (Recommended for Production)

**Store models in cloud storage (S3, Google Cloud Storage, etc.) and download on startup**

1. **Upload models to cloud storage**:
   ```bash
   # Example with AWS S3
   aws s3 cp backend/models/normal_abnormal_classifier.h5 s3://your-bucket/models/
   aws s3 cp backend/models/outer_eye_mobilenetv2.h5 s3://your-bucket/models/
   ```

2. **Download on Render startup**:
   Create a startup script `download_models.py`:
   ```python
   import boto3
   import os
   
   s3 = boto3.client('s3')
   bucket = 'your-bucket'
   models_dir = 'backend/models'
   
   os.makedirs(models_dir, exist_ok=True)
   
   # Download models
   s3.download_file(bucket, 'models/normal_abnormal_classifier.h5', 
                   f'{models_dir}/normal_abnormal_classifier.h5')
   s3.download_file(bucket, 'models/outer_eye_mobilenetv2.h5',
                   f'{models_dir}/outer_eye_mobilenetv2.h5')
   ```

3. **Update Render start command**:
   ```bash
   python download_models.py && npx tsx server.ts
   ```

### Option 3: Render Persistent Disk (Paid Plans)

**Use Render's persistent disk feature**

1. **Enable persistent disk** on Render
2. **Store models on persistent disk**
3. **Models persist across deployments**

## Current Setup (Render File System)

### Verify Models on Render

1. **Check Render logs** for model loading:
   ```
   [Backend] ✅ Loading model from backend: /opt/render/project/src/backend/models/outer_eye_mobilenetv2.h5
   ```

2. **SSH into Render** (if available):
   ```bash
   # Check if models exist
   ls backend/models/
   ```

3. **Test enhanced pipeline**:
   ```bash
   curl -X POST https://retina-cnn.onrender.com/trpc/detection.analyzeEnhanced \
     -H "Content-Type: application/json" \
     -d '{"imageUri": "base64..."}'
   ```

## Enhanced Pipeline Fallback

The enhanced pipeline works even without the 6-layer CNN:

1. **With 6-layer CNN**: Best accuracy for Stage 3 (normal/abnormal filter)
2. **Without 6-layer CNN**: Falls back to main model's Normal probability
3. **Still functional**: Pipeline completes successfully in both cases

### Fallback Behavior

```python
# Enhanced pipeline tries to load 6-layer CNN
normal_classifier_model = load_normal_vs_abnormal_classifier()

if normal_classifier_model is not None:
    # Use dedicated 6-layer CNN (better accuracy)
    return classify_normal_abnormal_cnn(image_array)
else:
    # Fallback: Use main model's Normal probability
    return classify_normal_abnormal_fallback(image_array)
```

## Deployment Checklist

- [ ] Models trained locally
- [ ] Models uploaded to Render (or cloud storage)
- [ ] Models verified on Render file system
- [ ] Enhanced pipeline tested on Render
- [ ] Fallback behavior verified (if models missing)
- [ ] Render logs show model loading success

## Troubleshooting

### Models Not Found on Render

**Error**: `FileNotFoundError: Model not found in backend/models/`

**Solution**:
1. Upload models to Render manually
2. Or use cloud storage and download on startup
3. Or verify model paths in enhanced_pipeline.py

### Enhanced Pipeline Falls Back to Main Model

**Expected behavior**: If 6-layer CNN not available, pipeline uses main model

**To enable 6-layer CNN**:
1. Ensure `normal_abnormal_classifier.h5` is in `backend/models/` on Render
2. Verify file permissions on Render
3. Check Render logs for loading errors

## Summary

- ✅ **Models excluded from Git** (exceeds 100 MB limit)
- ✅ **Models deployed to Render** (separate from Git)
- ✅ **Enhanced pipeline works** (with or without 6-layer CNN)
- ✅ **Fallback mechanism** (uses main model if 6-layer CNN unavailable)
- ✅ **Best accuracy** (when 6-layer CNN is available on Render)

The enhanced pipeline is designed to work flexibly - it provides the best accuracy when all models are available, but still functions correctly with fallbacks.

