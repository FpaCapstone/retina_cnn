#!/bin/bash
# Remove large model files from Git history
# This script uses git filter-branch to remove .h5 and .tflite files from entire Git history

set -e

echo "⚠️  WARNING: This will rewrite Git history!"
echo "⚠️  Make sure you have a backup of your repository!"
echo ""
read -p "Do you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Aborted"
    exit 1
fi

echo ""
echo "🚀 Removing large model files from Git history..."
echo ""

# Check if files exist in history
echo "📊 Checking for large files in Git history..."
H5_FILES=$(git log --all --full-history --name-only --pretty=format: -- "*.h5" | sort -u)
TFLITE_FILES=$(git log --all --full-history --name-only --pretty=format: -- "*.tflite" | sort -u)

if [ -z "$H5_FILES" ] && [ -z "$TFLITE_FILES" ]; then
    echo "✅ No model files found in Git history"
    echo "✅ Files are already removed. You can push now."
    exit 0
fi

echo "Found files in history:"
echo "$H5_FILES" | grep -v "^$" || true
echo "$TFLITE_FILES" | grep -v "^$" || true
echo ""

# Method 1: Try git filter-branch (built-in, no external dependencies)
echo "🔧 Using git filter-branch to remove files from history..."
echo ""

# Remove .h5 files
if [ ! -z "$H5_FILES" ]; then
    echo "Removing .h5 files from history..."
    git filter-branch --force --index-filter \
        "git rm --cached --ignore-unmatch *.h5 backend/models/*.h5 assets/images/models/*.h5" \
        --prune-empty --tag-name-filter cat -- --all
fi

# Remove .tflite files
if [ ! -z "$TFLITE_FILES" ]; then
    echo "Removing .tflite files from history..."
    git filter-branch --force --index-filter \
        "git rm --cached --ignore-unmatch *.tflite backend/models/*.tflite assets/images/models/*.tflite" \
        --prune-empty --tag-name-filter cat -- --all
fi

echo ""
echo "🧹 Cleaning up Git repository..."
echo ""

# Remove backup refs
rm -rf .git/refs/original/

# Expire reflog
git reflog expire --expire=now --all

# Garbage collection
git gc --prune=now --aggressive

echo ""
echo "✅ Git history cleaned!"
echo ""
echo "📝 Next steps:"
echo "1. Verify files are removed:"
echo "   git log --all --full-history -- '*.h5' | head -5"
echo ""
echo "2. Force push to GitHub (WARNING: Rewrites remote history):"
echo "   git push --force origin main"
echo ""
echo "⚠️  WARNING: Force push will rewrite remote history!"
echo "⚠️  Make sure all team members are aware and coordinate with them!"
echo ""
echo "3. Verify models are on Render backend"
echo "4. Test enhanced pipeline on Render"

