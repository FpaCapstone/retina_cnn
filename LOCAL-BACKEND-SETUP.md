# Local Backend Setup for Mobile App Sync

## Overview
This guide explains how to run the backend locally so your phone can sync files with it, while also maintaining Render backend as a fallback when online.

## Setup Steps

### 1. Start Local Backend

```bash
# In the project root
cd retina_cnn
npm run backend
# or
bun run backend
```

The backend will start on `http://localhost:3000`

### 2. Find Your Local IP Address

#### macOS/Linux:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
# Look for something like: inet 192.168.1.100
```

#### Windows:
```cmd
ipconfig
# Look for IPv4 Address, something like: 192.168.1.100
```

### 3. Configure Mobile App

#### Option A: Use Environment Variable (Recommended)

Create a `.env` file in the project root:

```env
# Local backend (for development/syncing)
EXPO_PUBLIC_LOCAL_BACKEND_URL=http://YOUR_LOCAL_IP:3000
# Example: EXPO_PUBLIC_LOCAL_BACKEND_URL=http://192.168.1.100:3000

# Render backend (production fallback)
# Leave EXPO_PUBLIC_API_BASE_URL unset to enable automatic fallback
# Or set it explicitly: EXPO_PUBLIC_API_BASE_URL=https://retina-cnn.onrender.com
```

#### Option B: Use Explicit Backend URL

If you want to force a specific backend:

```env
# Use local backend only
EXPO_PUBLIC_API_BASE_URL=http://192.168.1.100:3000

# Or use Render backend only
EXPO_PUBLIC_API_BASE_URL=https://retina-cnn.onrender.com
```

### 4. Ensure Phone and Computer are on Same Network

- Both devices must be on the same WiFi network
- Firewall should allow connections on port 3000
- Local IP should be accessible from your phone

### 5. Restart Expo App

After configuring, restart your Expo app:

```bash
# Clear cache and restart
npm start -- --clear
# or
bun start -- --clear
```

## Backend Priority Order

The app tries backends in this order:

1. **Explicit URL** (`EXPO_PUBLIC_API_BASE_URL`) - If set, only this URL is used
2. **Local Backend** (`EXPO_PUBLIC_LOCAL_BACKEND_URL` or auto-detected) - For development/syncing
3. **Render Backend** (`https://retina-cnn.onrender.com`) - Production fallback

## Automatic Fallback

When `EXPO_PUBLIC_API_BASE_URL` is **not set**, the app will:

1. Try local backend first (if configured)
2. Automatically fall back to Render backend if local is unavailable
3. Use TFLite/offline fallback if both backends are unavailable

## Testing

### Test Local Backend

```bash
# Check if backend is running
curl http://localhost:3000/

# From your phone (replace with your local IP)
curl http://192.168.1.100:3000/
```

### Test from Mobile App

1. Open the app on your phone
2. Go to Detection or Training screen
3. Check the sync indicator - it should show "Online" if connected
4. Try uploading an image - it should sync to local backend

## Troubleshooting

### Phone Can't Connect to Local Backend

1. **Check Network**: Ensure phone and computer are on same WiFi
2. **Check Firewall**: Allow port 3000 on your computer
3. **Check IP Address**: Verify local IP is correct in `.env`
4. **Check Backend**: Ensure backend is running on `localhost:3000`

### Backend Not Starting

```bash
# Check if port 3000 is in use
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
4. Test Render URL: `curl https://retina-cnn.onrender.com/`

## Production Deployment

For production (APK/iOS build):

1. **Don't set** `EXPO_PUBLIC_LOCAL_BACKEND_URL` (mobile won't have access to your local IP)
2. **Set** `EXPO_PUBLIC_API_BASE_URL=https://retina-cnn.onrender.com` (for EAS build)
3. Or leave unset to use Render backend automatically

## Environment Variables

### Development (Local Sync)
```env
EXPO_PUBLIC_LOCAL_BACKEND_URL=http://192.168.1.100:3000
# EXPO_PUBLIC_API_BASE_URL is not set (enables auto-fallback)
```

### Production (Render Only)
```env
EXPO_PUBLIC_API_BASE_URL=https://retina-cnn.onrender.com
# EXPO_PUBLIC_LOCAL_BACKEND_URL is not set
```

### Both (Explicit Priority)
```env
EXPO_PUBLIC_API_BASE_URL=http://192.168.1.100:3000
# Only this URL will be used (no fallback)
```

## Benefits

✅ **Local Development**: Test backend changes locally
✅ **File Syncing**: Sync detection and training data from phone to local backend
✅ **Automatic Fallback**: Falls back to Render when local is unavailable
✅ **Production Ready**: Works with Render backend in production
✅ **Offline Support**: Falls back to TFLite if both backends are unavailable

