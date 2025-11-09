# Render Deployment - Quick Start

## 🚀 Your Backend is Ready for Render!

Since you've connected your GitHub to Render, here's what you need to do:

## Step 1: Render Dashboard Setup

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select the `retina_cnn` repository

## Step 2: Configure Service

**Settings:**
- **Name**: `retina-backend` (or your choice)
- **Environment**: `Node`
- **Build Command**: `bun install && pip3 install -r python-scripts/requirements.txt`
- **Start Command**: `bun run backend`
- **Plan**: Free tier works for testing

**Environment Variables** (add in Render Dashboard):
```
NODE_ENV=production
PORT=10000
```

## Step 3: Get Your Render URL

After deployment, Render will give you a URL like:
```
https://retina-backend.onrender.com
```

## Step 4: Update Frontend

### Option A: Environment Variable (Recommended)

Create a `.env` file in your project root:

```env
EXPO_PUBLIC_API_BASE_URL=https://retina-backend.onrender.com
```

### Option B: Update in Code

If you can't use `.env`, update `lib/trpc.ts`:

```typescript
return "https://retina-backend.onrender.com"; // Your Render URL
```

## Step 5: Test

1. **Health Check**: Visit `https://your-service.onrender.com/`
2. **Run your app**: The frontend should connect automatically
3. **Check badge**: Should show "🌐 Backend AI Model (Online)"

## Important Notes

### Python on Render

Render supports Python, but you may need to:
- Add Python buildpack in Render settings
- Or use a Dockerfile (more advanced)

### Free Tier Limitations

- **Spins down** after 15 min inactivity
- **Cold start** takes ~30 seconds
- First request after spin-down will be slow

**Solution**: Use a cron job service to ping your backend every 10 minutes to keep it awake.

### Model Files

Make sure `backend/models/outer_eye_mobilenetv2.h5` is committed to Git, or Render won't have access to it.

## Troubleshooting

**Backend not starting?**
- Check Render logs for errors
- Make sure `bun` is available (Render should auto-detect)
- Verify Python is installed (may need buildpack)

**Frontend can't connect?**
- Check Render service is running (green status)
- Verify URL in environment variable
- Check CORS is enabled (already done)

**Python scripts failing?**
- Make sure `python-scripts/requirements.txt` dependencies are installed
- Check Python path in backend routes (currently uses `python3`)

## That's It! 🎉

Once deployed, your backend will be always-on (or mostly-on with free tier) and your app will show "🌐 Backend AI Model (Online)" when connected!

