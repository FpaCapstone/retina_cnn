// template
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Stack } from "expo-router";
import * as SplashScreen from "expo-splash-screen";
import React, { useEffect } from "react";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { trpc, trpcClient } from "@/lib/trpc";
import { OfflineProvider } from "@/contexts/offline-context";


// Prevent the splash screen from auto-hiding before asset loading is complete.
SplashScreen.preventAutoHideAsync();

const queryClient = new QueryClient();

function RootLayoutNav() {
  console.log('[Navigation] RootLayoutNav mounted');
  return (
    <Stack screenOptions={{ headerBackTitle: "Back" }}>
      <Stack.Screen 
        name="index" 
        options={{ 
          headerShown: false,
          title: 'Dashboard'
        }} 
      />
      <Stack.Screen 
        name="home" 
        options={{ 
          headerShown: false,
          title: 'Home'
        }} 
      />
      <Stack.Screen 
        name="detect" 
        options={{ 
          headerShown: true
        }} 
      />
      <Stack.Screen 
        name="training" 
        options={{ 
          headerShown: true,
          title: 'Training Data'
        }} 
      />
    </Stack>
  );
}

export default function RootLayout() {
  useEffect(() => {
    SplashScreen.hideAsync();
  }, []);

  return (
    <trpc.Provider client={trpcClient} queryClient={queryClient}>
      <QueryClientProvider client={queryClient}>
        <OfflineProvider>
          <GestureHandlerRootView>
            <RootLayoutNav />
          </GestureHandlerRootView>
        </OfflineProvider>
      </QueryClientProvider>
    </trpc.Provider>
  );
}
