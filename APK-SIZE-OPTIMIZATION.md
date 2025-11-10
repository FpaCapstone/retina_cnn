# APK Size Optimization Guide

## Problem
The APK file was ~800MB due to bundling large backend-only files.

## Solution
Since the backend is now hosted on Render (https://retina-cnn.onrender.com), we can exclude backend-only files from the mobile app build.

## Files Excluded from APK

### ✅ Backend Model Files (NOT in APK - 141MB saved)
- `backend/models/*.h5` - Backend Python models
  - `outer_eye_mobilenetv2.h5` (13MB)
  - `normal_abnormal_classifier.h5` (128MB)
- `assets/images/models/*.h5` - H5 models in assets (backend-only)

### ✅ Backend Server Files (NOT in APK)
- `backend/storage/` - Backend storage directory
- `backend/hono.ts` - Backend server setup
- `backend/create-context.ts` - Backend context
- `server.ts` - Backend server entry point
- `render.yaml` - Render deployment config

### ⚠️ Backend Type Definitions (KEPT for TypeScript)
- `backend/trpc/app-router.ts` - Type definitions (needed for TypeScript)
- `backend/trpc/routes/` - Route type definitions
- **Note**: These files are kept for TypeScript compilation but **won't be bundled** in the runtime because:
  1. They use server-side APIs (`fs`, `path`, `spawn`) that don't exist in React Native
  2. They're only imported as types (`import type`) in the app code
  3. Metro bundler excludes them automatically due to incompatible imports

### ✅ Python Scripts (NOT in APK)
- `python-scripts/` - All Python training/evaluation scripts
- `requirements.txt` - Python dependencies
- `venv/` - Python virtual environment
- Evaluation results and plots

### ✅ Documentation (NOT in APK)
- All `.md` files except `README.md`

## Files Kept in APK

### ✅ Essential App Files
- `app/` - App screens and routes
- `components/` - React Native components
- `utils/` - TypeScript utilities
- `assets/images/retina_logo.png` - App logo
- `assets/images/models/outer_eye_mobilenetv2.tflite` - **Small TFLite model (~10MB) for offline fallback**

## Expected APK Size Reduction

### Before Optimization
- **Total**: ~800MB
- Backend models: 141MB
- Python scripts: ~50MB
- Evaluation results: ~100MB+
- Documentation: ~10MB
- Other: ~500MB (dependencies, assets)

### After Optimization
- **Expected**: ~50-100MB
- TFLite model: 10MB (for offline fallback)
- App code: ~20-30MB
- Dependencies: ~40-60MB (React Native, TensorFlow.js, etc.)

## How It Works

### Online Mode (Primary)
1. App connects to Render backend: `https://retina-cnn.onrender.com`
2. Backend uses H5 models for inference
3. No models needed in mobile app

### Offline Mode (Fallback)
1. App uses on-device TFLite model (`outer_eye_mobilenetv2.tflite`)
2. Model is small (~10MB) and bundled in APK
3. Provides basic offline functionality

### Fallback Chain
```
1. Backend API (Render) → Uses H5 models
   ↓ (if unavailable)
2. On-Device TFLite → Uses bundled .tflite model
   ↓ (if unavailable)
3. Offline Rule-Based → Uses heuristics
```

## Benefits

1. **Smaller APK**: ~90% size reduction (800MB → 50-100MB)
2. **Faster Downloads**: Users download much smaller app
3. **Better Updates**: Model updates happen on backend, not app updates
4. **Offline Support**: Still works offline with TFLite fallback
5. **Centralized Models**: All model improvements happen on backend

## Verification

After building, verify:
1. APK size is reduced to ~50-100MB
2. App can connect to Render backend
3. Offline fallback still works with TFLite
4. All app functionality works as expected

## Notes

- The backend on Render should have all H5 models in `backend/models/`
- The mobile app only needs the small TFLite model for offline fallback
- Model updates can be deployed to Render without app updates
- Users get latest models automatically when online

