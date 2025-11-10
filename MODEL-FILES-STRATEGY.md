# Model Files Strategy for Enhanced Pipeline

## Problem
- `normal_abnormal_classifier.h5` is 127.87 MB (exceeds GitHub's 100 MB limit)
- Model is needed for the enhanced 5-stage pipeline
- Backend is deployed on Render

## Solution: Store Models on Render, Not in Git

### ✅ Recommended Approach

**Models stay on Render backend, not in Git repository**

1. **Render Backend** (Primary):
   - Models are stored in `backend/models/` on Render
   - Enhanced pipeline uses models from Render's file system
   - Models are deployed directly to Render (not via Git)

2. **Git Repository**:
   - Model files are excluded from Git (see `.gitignore`)
   - Only code and configuration are in Git
   - Models are deployed separately to Render

3. **Local Development**:
   - Models remain on your local machine
   - Developers can train/download models locally
   - Models sync to Render via deployment

## Deployment Strategy

### Option 1: Manual Upload to Render (Recommended)

1. **Train models locally**:
   ```bash
   python python-scripts/train_normal_abnormal_classifier.py
   python python-scripts/train_outer_eye_mobilenetv2.py
   ```

2. **Upload to Render**:
   - Use Render's file system or storage
   - Or use S3/cloud storage and download on Render startup
   - Or use Render's persistent disk (paid plans)

### Option 2: Git LFS (If You Must Track in Git)

If you really need models in Git:

1. **Install Git LFS**:
   ```bash
   brew install git-lfs  # macOS
   git lfs install
   ```

2. **Track model files**:
   ```bash
   git lfs track "*.h5"
   git lfs track "*.tflite"
   git add .gitattributes
   ```

3. **Add models**:
   ```bash
   git add assets/images/models/normal_abnormal_classifier.h5
   git add backend/models/normal_abnormal_classifier.h5
   git commit -m "Add models with Git LFS"
   ```

**Note**: Git LFS requires GitHub LFS bandwidth (1 GB free/month)

### Option 3: Cloud Storage (Best for Production)

1. **Upload models to cloud storage** (S3, Google Cloud Storage, etc.)
2. **Download on Render startup**:
   ```python
   # In server.ts or startup script
   import boto3  # or gsutil, etc.
   # Download models from S3 to backend/models/
   ```

3. **Benefits**:
   - Models not in Git
   - Easy to update without redeploying code
   - Scalable storage

## Current Setup

### Enhanced Pipeline Fallback

The enhanced pipeline already has a fallback mechanism:

1. **Tries to load 6-layer CNN** (`normal_abnormal_classifier.h5`)
2. **Falls back to main model** if 6-layer CNN not available
3. **Uses main model's Normal probability** for normal/abnormal classification

This means:
- ✅ Enhanced pipeline works with or without the 6-layer CNN
- ✅ Better accuracy with 6-layer CNN (when available)
- ✅ Still functional without it (uses main model)

## Render Deployment

### Ensure Models are on Render

1. **Check Render file system**:
   - Models should be in `backend/models/` on Render
   - Verify after deployment: `ls backend/models/`

2. **Deploy models separately**:
   - Use Render's file upload feature
   - Or use a deployment script that copies models
   - Or use environment variables to download from cloud storage

3. **Verify models are loaded**:
   - Check Render logs for model loading messages
   - Test enhanced pipeline endpoint

## Next Steps

### For Immediate Fix (GitHub Push)

1. **Keep models out of Git** (current setup):
   ```bash
   # Models are already in .gitignore
   # Just push code without models
   git push origin main
   ```

2. **Ensure models on Render**:
   - Upload models to Render manually
   - Or use deployment script
   - Or use cloud storage

### For Long-term Solution

1. **Use cloud storage** (S3, etc.) for models
2. **Download models on Render startup**
3. **Keep Git repo clean** (no large files)
4. **Easy model updates** (update cloud storage, restart Render)

## Summary

- ✅ **Models needed for enhanced pipeline**: Yes
- ✅ **Models in Git**: No (exceeds 100 MB limit)
- ✅ **Models on Render**: Yes (deployed separately)
- ✅ **Pipeline works without 6-layer CNN**: Yes (has fallback)
- ✅ **Best accuracy**: With 6-layer CNN on Render

The enhanced pipeline will work best when models are available on Render backend, but it has a fallback to use the main model if the 6-layer CNN isn't available.

