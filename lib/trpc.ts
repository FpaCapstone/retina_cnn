import { createTRPCReact } from "@trpc/react-query";
import { httpLink } from "@trpc/client";
import type { AppRouter } from "@/backend/trpc/app-router";
import superjson from "superjson";
import { Platform } from "react-native";
import { getPrimaryBackendUrl } from "@/utils/backend-config";

export const trpc = createTRPCReact<AppRouter>();

/**
 * Get base URL for TRPC client
 * Supports automatic fallback between local and Render backends
 * 
 * Priority order:
 * 1. EXPO_PUBLIC_API_BASE_URL (explicit, no fallback)
 * 2. Local backend (EXPO_PUBLIC_LOCAL_BACKEND_URL or auto-detected)
 * 3. Render backend (https://retina-cnn.onrender.com)
 */
const getBaseUrl = () => {
  // Priority 1: Explicit environment variable (highest priority, no fallback)
  const explicit = process.env.EXPO_PUBLIC_API_BASE_URL;
  if (explicit && explicit.length > 0) {
    console.log('[TRPC] Using explicit backend URL:', explicit);
    return explicit;
  }

  // Priority 2: For web, try to use same origin (if backend is on same domain)
  if (Platform.OS === "web" && typeof window !== "undefined") {
    const hostname = window.location.hostname;
    // If on Render domain, use same origin
    if (hostname.includes('render.com') || hostname.includes('onrender.com')) {
      console.log('[TRPC] Using same origin for Render deployment:', window.location.origin);
      return window.location.origin;
    }
  }

  // Priority 3: Use backend config (supports local + Render fallback)
  // This will return local backend if configured, otherwise Render backend
  const primaryUrl = getPrimaryBackendUrl();
  console.log('[TRPC] Using primary backend URL:', primaryUrl);
  return primaryUrl;
};

/**
 * Create TRPC client
 * Note: Fallback between backends is handled at the application level
 * (in ml-analysis.ts and ml-analysis-enhanced.ts) rather than in TRPC client
 */
export const trpcClient = trpc.createClient({
  links: [
    httpLink({
      url: `${getBaseUrl()}/trpc`,
      transformer: superjson,
      fetch: async (url, options) => {
        // Add timeout and better error handling
        try {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 8000); // 8 second timeout

          const response = await fetch(url, {
            ...options,
            signal: controller.signal,
          });

          clearTimeout(timeoutId);
          return response;
        } catch (error) {
          console.log('[TRPC] Request failed:', error);
          throw error;
        }
      },
    }),
  ],
});
