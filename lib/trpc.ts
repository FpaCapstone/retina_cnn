import { createTRPCReact } from "@trpc/react-query";
import { httpLink } from "@trpc/client";
import type { AppRouter } from "@/backend/trpc/app-router";
import superjson from "superjson";
import { Platform } from "react-native";

export const trpc = createTRPCReact<AppRouter>();

const getBaseUrl = () => {
  // Priority 1: Explicit environment variable (for Render/production)
  const explicit = process.env.EXPO_PUBLIC_API_BASE_URL;
  if (explicit && explicit.length > 0) return explicit;

  // Priority 2: For web, try to use same origin (if backend is on same domain)
  if (Platform.OS === "web" && typeof window !== "undefined") {
    // Check if we're in production (Render deployment)
    const hostname = window.location.hostname;
    // If on Render domain, use same origin
    if (hostname.includes('render.com') || hostname.includes('onrender.com')) {
      return window.location.origin;
    }
  }

  // Priority 3: Default to localhost for development
  return "http://localhost:3000";
};

export const trpcClient = trpc.createClient({
  links: [
    httpLink({
      url: `${getBaseUrl()}/trpc`,
      transformer: superjson,
    }),
  ],
});
