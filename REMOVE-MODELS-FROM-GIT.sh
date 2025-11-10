#!/bin/bash
# Script to remove large model files from Git tracking
# Run this script to fix GitHub push errors for large files

echo "🚀 Removing large model files from Git tracking..."
echo ""

# Remove model files from Git (but keep locally)
git rm --cached assets/images/models/normal_abnormal_classifier.h5 2>/dev/null
git rm --cached backend/models/normal_abnormal_classifier.h5 2>/dev/null
git rm --cached backend/models/outer_eye_mobilenetv2.h5 2>/dev/null
git rm --cached assets/images/models/outer_eye_mobilenetv2.tflite 2>/dev/null

echo "✅ Model files removed from Git tracking"
echo ""
echo "📝 Next steps:"
echo "1. Commit the changes:"
echo "   git commit -m 'Remove large model files from Git (stored on Render backend)'"
echo ""
echo "2. Push to GitHub:"
echo "   git push origin main"
echo ""
echo "✅ Files will remain on your local machine and on Render backend"
echo "✅ Future commits will ignore these files (see .gitignore)"

