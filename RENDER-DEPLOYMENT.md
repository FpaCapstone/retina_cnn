# Render Deployment Guide

## Overview

Your backend is now configured to deploy on Render. This guide will help you set it up.

## Render Setup Steps

### Step 1: Connect GitHub Repository

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select the `retina_cnn` repository

### Step 2: Configure Service

**Service Settings:**

- **Name**: `retina-backend` (or your preferred name)
- **Environment**: `Node`
- **Build Command**: `bun install`
- **Start Command**: `bun run backend`
- **Plan**: Free tier is fine for testing

### Step 3: Environment Variables

Add these in Render Dashboard → Environment:

```
NODE_ENV=production
PORT=10000
```

**Optional** (if you need Python backend):
```
PYTHON_VERSION=3.13
```

### Step 4: Deploy

Click **"Create Web Service"** and Render will:
1. Clone your repository
2. Run `bun install`
3. Start the server with `bun run backend`
4. Give you a URL like: `https://retina-backend.onrender.com`

### Step 5: Update Frontend Configuration

Once deployed, you'll get a URL like: `https://retina-backend.onrender.com`

**Option A: Environment Variable (Recommended)**

Create a `.env` file in your project root:

```env
EXPO_PUBLIC_API_BASE_URL=https://retina-backend.onrender.com
```

**Option B: Update in Render Dashboard**

In your Render service → Environment, add:

```
EXPO_PUBLIC_API_BASE_URL=https://retina-backend.onrender.com
```

Then update your frontend's `.env` or `app.json` to use this.

## Frontend Configuration

### For Web Deployment

The frontend will automatically detect if it's on the same domain as the backend.

### For Mobile/Expo Go

Update `lib/trpc.ts` or use environment variable:

```typescript
// In your .env file
EXPO_PUBLIC_API_BASE_URL=https://retina-backend.onrender.com
```

### For Production Build

When building for production, make sure to set the environment variable:

```bash
EXPO_PUBLIC_API_BASE_URL=https://retina-backend.onrender.com eas build --platform all
```

## Testing Your Render Backend

### 1. Health Check

Visit: `https://your-service.onrender.com/`

Should return:
```json
{
  "status": "ok",
  "message": "API is running ✅",
  "timestamp": "..."
}
```

### 2. Model Status

Visit: `https://your-service.onrender.com/api/model/status`

Should return model availability status.

### 3. Test from Frontend

Once the frontend is configured with the Render URL, run your app and check:
- Console logs should show: `[ML Analysis] Backend analysis successful`
- Results should show: `🌐 Backend AI Model (Online)`

## Important Notes

### Render Free Tier Limitations

- **Spins down after 15 minutes** of inactivity
- **Takes ~30 seconds** to spin back up
- **First request** after spin-down will be slow

**Solutions:**
1. Use Render's paid tier for always-on
2. Set up a cron job to ping your service every 10 minutes
3. Accept the cold start delay

### Python Scripts on Render

Your backend calls Python scripts. Render supports this, but you need to:

1. **Install Python dependencies** in build command:
   ```bash
   bun install && cd python-scripts && pip3 install -r requirements.txt
   ```

2. **Or use a separate Python service** on Render:
   - Create a second service for Python
   - Update backend to call that service's URL

### Model Files

Make sure your model files are in the repository or uploaded to Render:

- `backend/models/outer_eye_mobilenetv2.h5` should be committed to Git
- Or use Render's persistent disk to store models

## Troubleshooting

### Backend Not Starting

**Check Render Logs:**
1. Go to Render Dashboard → Your Service → Logs
2. Look for errors in build or start commands

**Common Issues:**
- Missing dependencies → Add to `package.json`
- Port issues → Render uses `PORT` env var automatically
- Python not found → Add Python buildpack or use Docker

### Frontend Can't Connect

**Check:**
1. Render service is running (green status)
2. CORS is enabled (already done in `backend/hono.ts`)
3. Frontend URL is correct in environment variables
4. Network tab shows requests to Render URL

### Slow First Request

This is normal on Render free tier (cold start). Subsequent requests will be fast.

## Production Checklist

- [ ] Backend deployed on Render
- [ ] Health check endpoint works
- [ ] Environment variables set
- [ ] Frontend configured with Render URL
- [ ] Model files accessible
- [ ] CORS enabled
- [ ] Tested analysis endpoint
- [ ] Monitoring set up (optional)

## Next Steps

1. **Deploy backend** to Render
2. **Get your Render URL** (e.g., `https://retina-backend.onrender.com`)
3. **Update frontend** with the URL
4. **Test the connection**
5. **Enjoy always-on backend!** 🎉

Your backend will now show "🌐 Backend AI Model (Online)" when connected to Render!

