# Render Build Command Guide

## Build Command Options

### Option 1: Simple (Node.js only)
```
bun install
```

**Use this if:**
- You only need Node.js dependencies
- Python scripts will be handled separately
- You're testing the basic setup first

### Option 2: With Python (Recommended)
```
bun install && pip3 install -r python-scripts/requirements.txt
```

**Use this if:**
- You need Python scripts to work (enhanced pipeline, predictions)
- Render has Python available in your environment
- You want everything to work out of the box

### Option 3: With Error Handling
```
bun install && (pip3 install -r python-scripts/requirements.txt || echo "Python deps skipped")
```

**Use this if:**
- You want the build to succeed even if Python isn't available
- You're unsure about Python availability on Render
- You want to test Node.js first, add Python later

## Recommended: Start with Option 1

**For your first deployment, use:**
```
bun install
```

**Why?**
- Simplest to get started
- Verifies Node.js backend works
- You can add Python later if needed

**Then test:**
1. Deploy with just `bun install`
2. Check if backend starts successfully
3. Test health endpoint
4. If Python scripts are needed, add Python to build command

## If Python Scripts Don't Work

If you get errors about Python scripts not found, you have two options:

### Option A: Add Python to Build Command
```
bun install && pip3 install -r python-scripts/requirements.txt
```

### Option B: Use Docker (Advanced)
Create a Dockerfile that includes both Node.js and Python.

## Current Recommendation

**Start with:**
```
bun install
```

This will:
- ✅ Install all Node.js dependencies
- ✅ Build your TypeScript server
- ✅ Start the backend
- ✅ Allow you to test the basic setup

**Then, if you need Python scripts:**
- Add Python buildpack in Render settings, OR
- Update build command to include Python installation

