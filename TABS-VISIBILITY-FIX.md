# Tabs Visibility Fix

## Issue
The Detection and Training tabs on the home screen may not be visible due to styling issues.

## Solution
Enhanced the tab visibility with:
1. Increased background opacity (0.1 → 0.15)
2. Thicker borders (2 → 2.5px)
3. More visible border color (0.2 → 0.4 opacity)
4. Added shadows for depth
5. Larger, bolder text
6. Better active state styling
7. Larger icons with varying opacity

## Navigation Flow

### Dashboard → Home Screen (with Tabs)
- Dashboard has a "Model Information" button that goes to `/home`
- Home screen (`/home`) has three tabs: Detection, Training, Info
- Tabs switch between different views within the home screen

### Dashboard → Direct Routes
- "Detect Eye Diseases" button goes directly to `/detect` route
- "Training Data" button goes directly to `/training` route
- These routes are separate screens WITHOUT tabs

## Tabs Location
The tabs (Detection, Training, Info) are ONLY on the `/home` route, not on `/detect` or `/training` routes.

## To See Tabs
1. Go to Dashboard
2. Click "Model Information" button (or navigate to `/home`)
3. You will see the three tabs: Detection, Training, Info

## Enhanced Styling
- Tab container: More visible background and borders
- Active tab: Highlighted with border and background
- Icons: Larger and more visible
- Text: Bolder and larger font size

