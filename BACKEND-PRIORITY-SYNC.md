# Backend Priority & Auto-Sync Implementation

## ✅ Implementation Complete

The app now ensures that:
1. **Backend models run first** (before offline fallback)
2. **Auto-sync happens when online**

## Analysis Priority Chain

The app tries models in this order for maximum accuracy:

### 1. 🚀 Enhanced Backend Pipeline (Priority 1)
- **5-stage pipeline** with quality checks, preprocessing, normal filter, disease classification, and validation
- **Best accuracy** for camera-captured images
- **Timeout**: 10 seconds
- **Location**: Render backend (`https://retina-cnn.onrender.com`)

### 2. 🔄 Standard Backend Analysis (Priority 2)
- **Fallback** if enhanced pipeline unavailable
- **Standard multi-class classification**
- **Timeout**: 8 seconds
- **Location**: Render backend

### 3. 📱 TFLite On-Device Model (Priority 3)
- **Offline fallback** when backend unavailable
- **Small model** (~10MB) bundled in APK
- **Basic accuracy** but works offline

### 4. ⚙️ Offline Rule-Based (Priority 4)
- **Final fallback** if all models fail
- **Heuristic-based** analysis
- **Lowest accuracy** but always works

## Auto-Sync Implementation

### When Online
- ✅ **Automatic sync** when device comes online
- ✅ **Immediate sync** when saving detections/training images
- ✅ **Retry logic** for failed syncs
- ✅ **Network detection** using NetInfo (mobile) and navigator.onLine (web)

### Sync Flow
1. **Save locally first** (AsyncStorage)
2. **Try to sync immediately** if online
3. **Mark as synced** if successful
4. **Keep as pending** if offline or fails
5. **Auto-sync later** when online

### Network Detection
- **Web**: Uses `navigator.onLine` and `online`/`offline` events
- **Mobile**: Uses `@react-native-community/netinfo` for accurate detection
- **Checks**: Both connection status AND internet reachability

## Code Changes

### `utils/ml-analysis.ts`
- ✅ Enhanced pipeline tried FIRST
- ✅ Standard backend tried SECOND
- ✅ TFLite fallback tried THIRD
- ✅ Improved backend availability check
- ✅ Better error handling and logging

### `contexts/offline-context.tsx`
- ✅ NetInfo integration for mobile network detection
- ✅ Auto-sync when coming online
- ✅ Immediate sync when saving (if online)
- ✅ Retry logic for failed syncs
- ✅ Better pending item tracking

### `utils/image-utils.ts`
- ✅ Image conversion to base64 for backend upload
- ✅ Handles file URIs, base64, and web URLs

## Benefits

1. **Best Accuracy First**: Enhanced pipeline provides best results
2. **Graceful Degradation**: Falls back through multiple options
3. **Offline Support**: Works even without internet
4. **Automatic Sync**: No manual intervention needed
5. **Data Integrity**: All data synced to backend when online
6. **Better UX**: Seamless experience regardless of connectivity

## Testing

To verify the implementation:

1. **Online Test**:
   - Connect to internet
   - Take a photo and analyze
   - Should use enhanced backend pipeline
   - Check logs: `[ML Analysis] 🚀 Attempting enhanced backend pipeline`

2. **Offline Test**:
   - Turn off internet
   - Take a photo and analyze
   - Should use TFLite fallback
   - Check logs: `[ML Analysis] 📱 Using TFLite model from assets`

3. **Sync Test**:
   - Go offline
   - Take a photo (saves locally)
   - Go online
   - Should auto-sync automatically
   - Check logs: `[Offline] 🔄 Auto-syncing X pending items...`

## Backend Requirements

Ensure your Render backend has:
- ✅ Enhanced pipeline script: `python-scripts/enhanced_pipeline.py`
- ✅ Standard analysis script: `python-scripts/predict_outer_eye.py`
- ✅ Model files: `backend/models/outer_eye_mobilenetv2.h5`
- ✅ Health endpoint: Returns `{"status":"ok"}` at `/`
- ✅ Environment variable: `EXPO_PUBLIC_API_BASE_URL=https://retina-cnn.onrender.com`

## Notes

- Backend timeout is set to 8-10 seconds to avoid long waits
- Auto-sync has a 1-2 second delay after coming online to ensure network is ready
- Failed syncs are retried automatically when online
- All images are converted to base64 before uploading to backend
- Sync status is tracked and displayed in the UI

