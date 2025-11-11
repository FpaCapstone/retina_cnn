/**
 * Backend Configuration Utility
 * Handles automatic detection and fallback between local and Render backends
 */

import { Platform } from 'react-native';

// Backend URLs
const RENDER_BACKEND_URL = 'https://retina-cnn.onrender.com';
const LOCAL_BACKEND_URL = 'http://localhost:3000';

// For mobile devices, we need to detect the local network IP
// This will be set via environment variable or detected automatically
const getLocalNetworkIP = (): string | null => {
  // Priority 1: Explicit environment variable (recommended for mobile)
  const explicit = process.env.EXPO_PUBLIC_LOCAL_BACKEND_URL;
  if (explicit && explicit.length > 0) {
    console.log('[Backend Config] Using explicit local backend URL:', explicit);
    return explicit;
  }

  // Priority 2: For web, use localhost
  if (Platform.OS === 'web') {
    console.log('[Backend Config] Web platform: using localhost');
    return LOCAL_BACKEND_URL;
  }

  // Priority 3: For mobile, check if we're in development mode
  // In development, try localhost first (works if using Expo Go with tunnel)
  // @ts-ignore - __DEV__ is a global in React Native
  if (typeof __DEV__ !== 'undefined' && __DEV__) {
    console.log('[Backend Config] Development mode: using localhost (set EXPO_PUBLIC_LOCAL_BACKEND_URL for local network IP)');
    return LOCAL_BACKEND_URL;
  }

  // For mobile production, return null if not explicitly set
  // User should configure their local IP in .env
  console.log('[Backend Config] Mobile production: no local backend configured');
  return null;
};

/**
 * Get backend URLs in priority order
 * Returns array of URLs to try, in order of preference
 */
export function getBackendUrls(): string[] {
  const urls: string[] = [];

  // Priority 1: Explicit environment variable (highest priority)
  const explicit = process.env.EXPO_PUBLIC_API_BASE_URL;
  if (explicit && explicit.length > 0) {
    urls.push(explicit);
    // If explicit is set, only use that (no fallback)
    return urls;
  }

  // Priority 2: Local backend (for development/syncing)
  const localUrl = getLocalNetworkIP();
  if (localUrl) {
    urls.push(localUrl);
  }

  // Priority 3: Render backend (production, always available when online)
  urls.push(RENDER_BACKEND_URL);

  return urls;
}

/**
 * Check if a backend URL is available
 */
export async function checkBackendHealth(url: string, timeout: number = 3000): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const response = await fetch(`${url}/`, {
      method: 'GET',
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
      },
    });

    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    console.log(`[Backend Config] Health check failed for ${url}:`, error);
    return false;
  }
}

/**
 * Find the first available backend URL
 * Tries URLs in priority order and returns the first one that responds
 */
export async function findAvailableBackend(): Promise<string | null> {
  const urls = getBackendUrls();
  console.log(`[Backend Config] Trying ${urls.length} backend URL(s):`, urls);

  for (const url of urls) {
    try {
      const isAvailable = await checkBackendHealth(url, 3000);
      if (isAvailable) {
        console.log(`[Backend Config] ✅ Backend available: ${url}`);
        return url;
      }
    } catch (error) {
      console.log(`[Backend Config] ❌ Backend not available: ${url}`, error);
      continue;
    }
  }

  console.log(`[Backend Config] ⚠️ No backend available. All URLs failed.`);
  return null;
}

/**
 * Get the primary backend URL (first in priority list)
 * Use this for TRPC client initialization
 */
export function getPrimaryBackendUrl(): string {
  const urls = getBackendUrls();
  return urls[0] || RENDER_BACKEND_URL;
}

/**
 * Get backend URL with automatic fallback
 * For TRPC client that supports retries and error handling
 */
export function getBackendUrlWithFallback(): string {
  // Use primary URL, TRPC will handle retries
  // The actual fallback happens in the API calls
  return getPrimaryBackendUrl();
}

