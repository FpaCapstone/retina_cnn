# Quick Build Commands for APK and iOS Preview

## 🚀 Ready to Build!

Your EAS configuration is set up. Here are the commands:

### Android APK Preview

```bash
cd retina_cnn
eas build --platform android --profile preview
```

**What you'll get:**
- APK file you can install directly on Android devices
- No Google Play Store needed
- Perfect for testing and sharing

### iOS Preview (Device)

```bash
cd retina_cnn
eas build --platform ios --profile preview
```

**What you'll get:**
- iOS app build for physical devices
- Requires Apple Developer account ($99/year)
- Can be installed via TestFlight or direct install

### Build Both at Once

```bash
cd retina_cnn
eas build --platform all --profile preview
```

## 📋 Before Building

1. **Make sure you're logged in:**
   ```bash
   eas login
   ```

2. **Check your project:**
   ```bash
   eas whoami
   ```

3. **Verify configuration:**
   - ✅ EAS project ID: `2be631b1-fd63-4b11-bb61-fbbd0b5ad138`
   - ✅ Android package: `com.kazuyasensei.retinaeyediseasetest`
   - ✅ iOS bundle ID: `app.retinaeyediseasetest`

## 🔧 Build Process

1. **EAS will:**
   - Upload your code to Expo servers
   - Build the app in the cloud
   - Generate download links

2. **You'll receive:**
   - Email notification when build completes
   - Download link in terminal
   - Build status in EAS dashboard

## ⏱️ Build Time

- **Android APK**: ~10-15 minutes
- **iOS**: ~15-20 minutes (first build may be longer)

## 📱 After Build

### Android
1. Download APK from EAS dashboard
2. Transfer to Android device
3. Enable "Install from unknown sources" in settings
4. Install and test!

### iOS
1. Download from EAS dashboard
2. Install via TestFlight (recommended) or direct install
3. Test on your iPhone/iPad

## 🐛 Troubleshooting

### If build fails:
- Check EAS dashboard for error logs
- Verify all assets are included
- Make sure environment variables are set (if needed)

### iOS-specific:
- Need Apple Developer account
- Certificates are auto-managed by EAS
- First build may take longer

## 💡 Pro Tips

1. **Set environment variables** (if your app needs them):
   ```bash
   eas secret:create --scope project --name EXPO_PUBLIC_API_BASE_URL --value https://retina-cnn.onrender.com
   ```

2. **Check build status:**
   ```bash
   eas build:list
   ```

3. **View build details:**
   ```bash
   eas build:view [BUILD_ID]
   ```

## 🎯 Next Steps

1. Run the build command for your platform
2. Wait for build to complete
3. Download and test on device
4. Share with testers if needed

Good luck! 🚀

