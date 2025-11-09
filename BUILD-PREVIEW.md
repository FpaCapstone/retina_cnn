# Building APK and iOS Preview Builds

## Prerequisites

1. **EAS CLI installed** ✅ (Already installed)
2. **Expo account** - Make sure you're logged in: `eas login`
3. **EAS project configured** ✅ (Project ID: `2be631b1-fd63-4b11-bb61-fbbd0b5ad138`)

## Build Commands

### Android APK Preview

```bash
# Build APK for Android preview
eas build --platform android --profile preview
```

This will:
- Build an APK file (not AAB)
- Allow installation without Google Play Store
- Perfect for testing and sharing

### iOS Preview

```bash
# Build iOS preview (requires Apple Developer account)
eas build --platform ios --profile preview
```

**Note**: iOS builds require:
- Apple Developer account ($99/year)
- Proper certificates and provisioning profiles
- EAS will handle this automatically if configured

## Build Both Platforms

```bash
# Build both Android and iOS previews
eas build --platform all --profile preview
```

## Check Build Status

```bash
# View all your builds
eas build:list

# View specific build details
eas build:view [BUILD_ID]
```

## Download Builds

After builds complete, you'll get:
- **Android**: Direct download link for APK
- **iOS**: Download link (requires TestFlight or direct install)

## EAS Build Profiles

Your `eas.json` should have a `preview` profile configured. If not, EAS will use defaults:
- **Android**: APK format, no signing (for testing)
- **iOS**: Development build (requires Apple Developer account)

## Quick Start

1. **Login to EAS** (if not already):
   ```bash
   eas login
   ```

2. **Build Android APK**:
   ```bash
   cd retina_cnn
   eas build --platform android --profile preview
   ```

3. **Build iOS** (if you have Apple Developer account):
   ```bash
   eas build --platform ios --profile preview
   ```

## Troubleshooting

### Android Build Issues
- Make sure `package` in `app.json` is correct: `com.kazuyasensei.retinaeyediseasetest`
- Check that all assets are properly referenced

### iOS Build Issues
- Ensure you have an Apple Developer account
- Check bundle identifier: `app.retinaeyediseasetest`
- Verify certificates are set up in EAS

## Build Time

- **Android APK**: ~10-15 minutes
- **iOS**: ~15-20 minutes (first build may take longer)

## After Build

1. Download the APK/iOS build from the EAS dashboard
2. **Android**: Install APK directly on device
3. **iOS**: Install via TestFlight or direct install (if configured)

## Environment Variables

If your app needs environment variables (like API URLs), set them in EAS:

```bash
# Set environment variable for builds
eas secret:create --scope project --name EXPO_PUBLIC_API_BASE_URL --value https://retina-cnn.onrender.com
```

## Next Steps

After building:
1. Test the APK on Android device
2. Test iOS build on iPhone/iPad
3. Share with testers if needed
4. Fix any issues and rebuild

Good luck with your builds! 🚀

