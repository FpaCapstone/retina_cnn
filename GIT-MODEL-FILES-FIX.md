# Fix: Large Model Files in Git

## Problem
GitHub rejected the push because `normal_abnormal_classifier.h5` (127.87 MB) exceeds GitHub's 100 MB file size limit.

## Solution
Remove large model files from Git tracking since they're now stored on Render backend.

## Steps to Fix

### 1. Remove Model Files from Git Tracking

Run these commands to remove the files from Git (but keep them locally):

```bash
# Remove H5 model files from Git tracking
git rm --cached assets/images/models/normal_abnormal_classifier.h5
git rm --cached backend/models/normal_abnormal_classifier.h5
git rm --cached backend/models/outer_eye_mobilenetv2.h5

# Remove TFLite file from Git tracking (optional - it's small but consistent)
git rm --cached assets/images/models/outer_eye_mobilenetv2.tflite

# Commit the removal
git commit -m "Remove large model files from Git (stored on Render backend)"
```

### 2. Verify .gitignore is Updated

The `.gitignore` file has been updated to exclude:
- `*.h5` files
- `*.tflite` files
- `backend/models/` directory
- `assets/images/models/` directory

### 3. Push to GitHub

```bash
git push origin main
```

## Why This is Safe

1. **Backend on Render**: Models are stored on Render backend server
2. **APK Optimization**: Models are excluded from APK build (see `.easignore`)
3. **Local Development**: Models remain on your local machine
4. **Training Scripts**: You can always retrain and regenerate models

## Alternative: Git LFS (If You Need to Track Models)

If you really need to track model files in Git, use Git LFS:

```bash
# Install Git LFS
brew install git-lfs  # macOS
# or: apt install git-lfs  # Linux

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.h5"
git lfs track "*.tflite"

# Add .gitattributes
git add .gitattributes

# Remove files from Git and re-add with LFS
git rm --cached assets/images/models/*.h5
git rm --cached backend/models/*.h5
git add assets/images/models/*.h5
git add backend/models/*.h5

# Commit
git commit -m "Track model files with Git LFS"
```

**Note**: Git LFS requires a GitHub account with LFS bandwidth. For this project, it's better to exclude models from Git since they're on Render.

## Verification

After removing files from Git:

```bash
# Check that files are no longer tracked
git ls-files | grep -E "\.h5$|\.tflite$"

# Should return empty (no files tracked)
```

## Files That Will Be Ignored

- `backend/models/*.h5` (127.87 MB + 13 MB)
- `assets/images/models/*.h5` (127.87 MB)
- `assets/images/models/*.tflite` (9.7 MB - optional, can keep if needed)

## Next Steps

1. ✅ Run the removal commands above
2. ✅ Commit the changes
3. ✅ Push to GitHub
4. ✅ Verify models are on Render backend
5. ✅ Test that backend works without models in Git

