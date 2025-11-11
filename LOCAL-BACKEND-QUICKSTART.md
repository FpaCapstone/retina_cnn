# Quick Start: Local Backend + Render Fallback

## Overview
The app now supports automatic fallback between local backend (for syncing from your phone) and Render backend (production). 

## How It Works

### Backend Priority Order:
1. **Explicit URL** (`EXPO_PUBLIC_API_BASE_URL`) - If set, only this URL is used
2. **Local Backend** (`EXPO_PUBLIC_LOCAL_BACKEND_URL`) - For development/syncing
3. **Render Backend** (`https://retina-cnn.onrender.com`) - Production fallback

### Automatic Behavior:
- If `EXPO_PUBLIC_API_BASE_URL` is **not set**, the app will:
  - Try local backend first (if `EXPO_PUBLIC_LOCAL_BACKEND_URL` is configured)
  - Automatically fall back to Render backend if local is unavailable
  - Use TFLite/offline fallback if both backends are unavailable

## Setup for Local Backend (Phone Syncing)

### Step 1: Find Your Local IP Address

**macOS/Linux:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```
Look for something like: `inet 192.168.1.100`

**Windows:**
```cmd
ipconfig
```
Look for `IPv4 Address`, something like: `192.168.1.100`

### Step 2: Start Local Backend

```bash
cd retina_cnn
npm run backend
# or
bun run backend
```

Backend will start on `http://localhost:3000`

### Step 3: Configure Mobile App

Create or update `.env` file in project root:

```env
# Local backend (for syncing from phone)
EXPO_PUBLIC_LOCAL_BACKEND_URL=http://YOUR_LOCAL_IP:3000
# Example: EXPO_PUBLIC_LOCAL_BACKEND_URL=http://192.168.1.100:3000

# Leave EXPO_PUBLIC_API_BASE_URL unset to enable automatic fallback
```

### Step 4: Ensure Same Network

- Phone and computer must be on the **same WiFi network**
- Firewall should allow connections on port 3000
- Test connectivity: From your phone browser, visit `http://YOUR_LOCAL_IP:3000`

### Step 5: Restart App

```bash
npm start -- --clear
# or
bun start -- --clear
```

## Configuration Options

### Option 1: Automatic Fallback (Recommended)
```env
# Local backend (tried first)
EXPO_PUBLIC_LOCAL_BACKEND_URL=http://192.168.1.100:3000

# Render backend will be used automatically if local fails
# (No EXPO_PUBLIC_API_BASE_URL set)
```

### Option 2: Explicit Local Backend Only
```env
# Only use local backend (no fallback to Render)
EXPO_PUBLIC_API_BASE_URL=http://192.168.1.100:3000
```

### Option 3: Explicit Render Backend Only
```env
# Only use Render backend (no fallback to local)
EXPO_PUBLIC_API_BASE_URL=https://retina-cnn.onrender.com
```

### Option 4: Render Backend Only (Production)
```env
# For production builds, don't set local backend
# Render backend will be used automatically
# (No EXPO_PUBLIC_LOCAL_BACKEND_URL or EXPO_PUBLIC_API_BASE_URL)
```

## Testing

### Test Local Backend
```bash
# From computer
curl http://localhost:3000/

# From phone (replace with your IP)
curl http://192.168.1.100:3000/
```

### Test from App
1. Open app on phone
2. Check sync indicator - should show "Online" if connected
3. Upload an image - should sync to local backend
4. Check backend logs - should see incoming requests

## Troubleshooting

### Phone Can't Connect
1. **Check Network**: Both devices on same WiFi?
2. **Check Firewall**: Port 3000 allowed?
3. **Check IP**: Correct local IP in `.env`?
4. **Check Backend**: Backend running on `localhost:3000`?

### Backend Not Starting
```bash
# Check if port is in use
lsof -i :3000

# Kill process if needed
kill -9 <PID>

# Restart backend
npm run backend
```

### Render Backend Not Working
1. Check Render dashboard: https://dashboard.render.com
2. Verify backend is running
3. Check backend logs for errors
4. Test: `curl https://retina-cnn.onrender.com/`

## Benefits

✅ **Local Development**: Test backend changes locally
✅ **File Syncing**: Sync detection and training data from phone to local backend
✅ **Automatic Fallback**: Falls back to Render when local is unavailable
✅ **Production Ready**: Works with Render backend in production
✅ **Offline Support**: Falls back to TFLite if both backends are unavailable

## Production Deployment

For production (APK/iOS build):

**Don't set** `EXPO_PUBLIC_LOCAL_BACKEND_URL` (mobile won't have access to your local IP)

**Option 1: Use Render automatically**
```env
# Leave both unset - Render will be used automatically
```

**Option 2: Explicit Render URL**
```env
EXPO_PUBLIC_API_BASE_URL=https://retina-cnn.onrender.com
```

## Summary

- **Local Backend**: Set `EXPO_PUBLIC_LOCAL_BACKEND_URL` for phone syncing
- **Render Backend**: Automatically used as fallback (or set `EXPO_PUBLIC_API_BASE_URL` for explicit)
- **Automatic Fallback**: Works seamlessly between local and Render
- **Offline Support**: Falls back to TFLite if both backends unavailable

