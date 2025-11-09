# Detection Flow Explanation

## Why It Shows "Offline" Even When Connected to WiFi

The app uses a **fallback chain** for detection, and the badge shows which method was **actually used**, not your connection status.

## Detection Flow (Priority Order)

### 1️⃣ **Enhanced Pipeline (Online)** 🌐
- **When**: Backend server is running and accessible
- **Shows**: "🌐 Backend AI Model (Online)" or "Enhanced 5-Stage Pipeline"
- **Requires**: 
  - WiFi/Internet connection
  - Backend server running (`python-scripts/enhanced_pipeline.py` or backend API)
- **Best accuracy**: Uses the 5-stage enhanced pipeline

### 2️⃣ **Standard Backend (Online)** 🌐
- **When**: Backend server is running but enhanced pipeline unavailable
- **Shows**: "🌐 Backend AI Model (Online)"
- **Requires**: 
  - WiFi/Internet connection
  - Backend server running
- **Good accuracy**: Uses standard MobileNetV2 model

### 3️⃣ **TFLite Model (Offline)** 📱
- **When**: Backend unavailable, but TFLite model exists in assets
- **Shows**: "📱 On-Device AI Model (Offline)"
- **Requires**: 
  - TFLite model file in `assets/images/models/`
  - No internet needed
- **Decent accuracy**: On-device TensorFlow Lite model

### 4️⃣ **Offline Fallback (Offline)** ⚙️
- **When**: All above methods fail
- **Shows**: "⚙️ Offline Analysis"
- **Requires**: Nothing (always works)
- **Basic accuracy**: Rule-based analysis using image features

## Why You Might See "Offline" When Connected to WiFi

### Common Reasons:

1. **Backend Server Not Running**
   - The backend Python server needs to be started separately
   - Even with WiFi, if the backend isn't running, it falls back to offline methods

2. **Backend Timeout**
   - If the backend takes >5 seconds to respond, it times out
   - Falls back to TFLite or offline methods

3. **Backend Connection Failed**
   - Network issues, firewall, or CORS problems
   - Falls back to offline methods

4. **TFLite Model Available**
   - If TFLite model exists, it uses that instead of trying backend
   - This is actually faster and works offline!

## How to Ensure Online Detection

### Option 1: Start Backend Server

```bash
# In your backend directory
cd backend
# Start your backend server (depends on your setup)
# For example, if using Hono/Node.js:
npm run dev

# Or if using Python backend:
python python-scripts/enhanced_pipeline.py
```

### Option 2: Check Backend Configuration

Make sure your `lib/trpc.ts` points to the correct backend URL:

```typescript
// Should point to your backend server
const trpcClient = createTRPCClient<AppRouter>({
  links: [
    httpBatchLink({
      url: 'http://localhost:3000/trpc', // or your backend URL
    }),
  ],
});
```

### Option 3: Use Enhanced Pipeline

The enhanced pipeline tries the backend first. Make sure:
- Backend server is running
- Enhanced pipeline route is accessible
- Network connection is stable

## Understanding the Badge

The badge shows **which method was actually used**, not your connection status:

- **🌐 Backend AI Model (Online)** = Used backend server (requires WiFi + server)
- **📱 On-Device AI Model (Offline)** = Used TFLite model (works without WiFi)
- **⚙️ Offline Analysis** = Used rule-based fallback (always works)

## Benefits of This Approach

✅ **Always Works**: Even without WiFi, you get results
✅ **Best Available**: Uses the best method available
✅ **Fast Fallback**: Quickly falls back if backend is slow
✅ **Privacy**: On-device methods don't send data anywhere

## Summary

**"Offline" doesn't mean you're not connected to WiFi** - it means the app used an offline-capable method (TFLite or rule-based) instead of the backend server. This could be because:
- Backend server isn't running
- Backend timed out
- TFLite model was faster/available
- Network issues prevented backend connection

The app prioritizes **getting you results** over requiring a specific connection type!

