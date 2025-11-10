# Remove Large Model Files from Git History

## Problem
Large model files (127.87 MB) are in Git history, causing GitHub to reject pushes even after removing them from tracking.

## Solution
Remove files from entire Git history using `git filter-branch`.

## ⚠️ Important Warnings

1. **Rewrites Git history**: This will change commit hashes
2. **Force push required**: You'll need to force push to GitHub
3. **Team coordination**: All team members need to re-clone or reset their repos
4. **Backup first**: Make sure you have a backup

## Quick Fix (Automated Script)

```bash
# Run the removal script
./remove-models-from-history.sh
```

## Manual Steps

### Step 1: Check if Files are in History

```bash
# Check for H5 files in history
git log --all --full-history --oneline -- "*.h5" | head -5

# Check for TFLite files in history
git log --all --full-history --oneline -- "*.tflite" | head -5
```

### Step 2: Remove from History

```bash
# Remove .h5 files from entire history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch *.h5 backend/models/*.h5 assets/images/models/*.h5" \
  --prune-empty --tag-name-filter cat -- --all

# Remove .tflite files from entire history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch *.tflite backend/models/*.tflite assets/images/models/*.tflite" \
  --prune-empty --tag-name-filter cat -- --all
```

### Step 3: Clean Up Repository

```bash
# Remove backup refs
rm -rf .git/refs/original/

# Expire reflog
git reflog expire --expire=now --all

# Garbage collection (removes unreachable objects)
git gc --prune=now --aggressive
```

### Step 4: Verify Files are Removed

```bash
# Check that files are no longer in history
git log --all --full-history -- "*.h5" | head -5
# Should return nothing or very old commits

# Check repository size
du -sh .git
```

### Step 5: Force Push to GitHub

```bash
# ⚠️ WARNING: This rewrites remote history
git push --force origin main
```

## Alternative: Use BFG Repo-Cleaner (Faster)

BFG is faster and easier than `git filter-branch`:

```bash
# Install BFG
brew install bfg

# Remove large files from history
bfg --delete-files "*.h5"
bfg --delete-files "*.tflite"

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push
git push --force origin main
```

## After Removing from History

### 1. Verify GitHub Push Works

```bash
git push origin main
# Should now work without file size errors
```

### 2. Ensure Models on Render

Models should be on Render backend:
- Upload manually via Render dashboard
- Or use deployment script
- Or use cloud storage (S3) and download on startup

### 3. Team Members

If you have team members, they need to:

```bash
# Option 1: Re-clone (easiest)
git clone https://github.com/FpaCapstone/retina_cnn.git

# Option 2: Reset local repo
git fetch origin
git reset --hard origin/main
```

## Verification Checklist

- [ ] Files removed from Git history
- [ ] Repository size reduced
- [ ] Can push to GitHub without errors
- [ ] Models available on Render backend
- [ ] Enhanced pipeline works on Render
- [ ] Team members notified (if applicable)

## Troubleshooting

### Error: "remote: error: GH001: Large files detected"

**Solution**: Files are still in history. Run the removal script again or use BFG.

### Error: "Updates were rejected because the tip of your current branch is behind"

**Solution**: This happens if someone else pushed. You need to force push:
```bash
git push --force origin main
```

### Error: "Permission denied"

**Solution**: Make sure you have push access to the repository.

## Summary

✅ **Remove files from Git history** (using filter-branch or BFG)
✅ **Clean up repository** (gc, reflog expire)
✅ **Force push to GitHub** (rewrites remote history)
✅ **Deploy models to Render** (separate from Git)
✅ **Enhanced pipeline works** (with fallback if models missing)

After this, your Git repository will be clean and GitHub will accept your pushes!

