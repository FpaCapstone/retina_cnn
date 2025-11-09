# Setup Render Backend URL

## Your Render Backend

**URL**: https://retina-cnn.onrender.com

## Step 1: Create .env File

Create a `.env` file in your project root (`retina_cnn/.env`):

```env
EXPO_PUBLIC_API_BASE_URL=https://retina-cnn.onrender.com
```

## Step 2: Restart Frontend

After creating the `.env` file, restart your frontend:

```bash
# Stop current server (Ctrl+C)
# Then restart
bun run start-web
```

## Step 3: Test Connection

1. **Test Backend Health**: Visit https://retina-cnn.onrender.com/
   - Should return: `{"status":"ok","message":"API is running ✅"}`

2. **Test from App**: 
   - Open your app
   - Try analyzing an image
   - Check the results badge - should show: **"🌐 Backend AI Model (Online)"**

## Quick Setup Command

```bash
# Create .env file
echo "EXPO_PUBLIC_API_BASE_URL=https://retina-cnn.onrender.com" > .env

# Restart frontend
bun run start-web
```

## That's It! 🎉

Your frontend will now connect to your Render backend automatically!

