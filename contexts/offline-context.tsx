import createContextHook from '@nkzw/create-context-hook';
import { useState, useEffect, useCallback, useMemo } from 'react';
import { Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import { DetectionRecord, TrainingImageRecord } from '@/types/database';
import { useMutation } from '@tanstack/react-query';
import { trpcClient } from '@/lib/trpc';
import { convertDiseaseTypeToBackend } from '@/utils/dataset-utils';
import { convertImageToBase64 } from '@/utils/image-utils';

const DETECTIONS_KEY = 'offline_detections';
const TRAINING_IMAGES_KEY = 'offline_training_images';

export const [OfflineProvider, useOffline] = createContextHook(() => {
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [detections, setDetections] = useState<DetectionRecord[]>([]);
  const [trainingImages, setTrainingImages] = useState<TrainingImageRecord[]>([]);

  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSyncTime, setLastSyncTime] = useState<Date | null>(null);

  const syncPendingData = useCallback(async () => {
    if (!isOnline) {
      console.log('[Offline] Cannot sync - device is offline');
      return;
    }

    if (isSyncing) {
      console.log('[Offline] Sync already in progress');
      return;
    }

    console.log('[Offline] Starting sync...');
    setIsSyncing(true);
    
    try {
      const detectionsData = await AsyncStorage.getItem(DETECTIONS_KEY);
      const trainingData = await AsyncStorage.getItem(TRAINING_IMAGES_KEY);
      
      const currentDetections: DetectionRecord[] = detectionsData ? JSON.parse(detectionsData) : [];
      const currentTraining: TrainingImageRecord[] = trainingData ? JSON.parse(trainingData) : [];
      
      const unsyncedDetections = currentDetections.filter(d => !d.synced);
      const unsyncedTraining = currentTraining.filter(t => !t.synced);

      console.log(`[Offline] Found ${unsyncedDetections.length} unsynced detections and ${unsyncedTraining.length} unsynced training images`);

      // Sync detections with images
      for (const detection of unsyncedDetections) {
        try {
          // Convert image to base64 if needed
          const imageBase64 = await convertImageToBase64(detection.imageUri);
          
          // Convert DiseaseType to backend format
          const backendDetection = {
            ...detection,
            imageUri: imageBase64, // Use converted base64 image
            primaryDisease: convertDiseaseTypeToBackend(detection.primaryDisease),
            detections: detection.detections.map(d => ({
              ...d,
              disease: convertDiseaseTypeToBackend(d.disease),
            })),
          };
          await trpcClient.detection.save.mutate(backendDetection);
          detection.synced = true;
          console.log('[Offline] ✅ Synced detection:', detection.id);
        } catch (error) {
          console.error('[Offline] ❌ Failed to sync detection:', detection.id, error);
          // Keep synced=false so it will retry next time
        }
      }

      // Sync training images with images
      for (const training of unsyncedTraining) {
        try {
          // Convert image to base64 if needed
          const imageBase64 = await convertImageToBase64(training.imageUri);
          
          // Convert DiseaseType to backend format
          const backendTraining = {
            ...training,
            imageUri: imageBase64, // Use converted base64 image
            disease: convertDiseaseTypeToBackend(training.disease),
          };
          await trpcClient.training.save.mutate(backendTraining);
          training.synced = true;
          console.log('[Offline] ✅ Synced training image:', training.id);
        } catch (error) {
          console.error('[Offline] ❌ Failed to sync training image:', training.id, error);
          // Keep synced=false so it will retry next time
        }
      }

      // Update storage with synced status
      await Promise.all([
        AsyncStorage.setItem(DETECTIONS_KEY, JSON.stringify(currentDetections)),
        AsyncStorage.setItem(TRAINING_IMAGES_KEY, JSON.stringify(currentTraining)),
      ]);
      
      setDetections(currentDetections);
      setTrainingImages(currentTraining);
      setLastSyncTime(new Date());

      const syncedCount = currentDetections.filter(d => d.synced).length + currentTraining.filter(t => t.synced).length;
      console.log(`[Offline] ✅ Sync completed. Total synced: ${syncedCount}`);
    } catch (error) {
      console.error('[Offline] ❌ Sync failed:', error);
    } finally {
      setIsSyncing(false);
    }
  }, [isOnline, isSyncing]);

  useEffect(() => {
    let unsubscribe: (() => void) | undefined;

    if (Platform.OS === 'web') {
      const handleOnline = async () => {
        console.log('[Offline] Web network status changed: online');
        setIsOnline(true);
        // Auto-sync when coming online
        await loadCachedData();
        const pendingCount = detections.filter(d => !d.synced).length + trainingImages.filter(t => !t.synced).length;
        if (pendingCount > 0 && !isSyncing) {
          console.log(`[Offline] 🔄 Auto-syncing ${pendingCount} pending items...`);
          setTimeout(() => syncPendingData(), 1000); // Small delay to ensure network is ready
        }
      };
      const handleOffline = () => {
        console.log('[Offline] Web network status changed: offline');
        setIsOnline(false);
      };
      window.addEventListener('online', handleOnline);
      window.addEventListener('offline', handleOffline);
      setIsOnline(typeof navigator !== 'undefined' ? navigator.onLine : true);
      unsubscribe = () => {
        window.removeEventListener('online', handleOnline);
        window.removeEventListener('offline', handleOffline);
      };
    } else {
      // For mobile, use NetInfo for better network detection
      const unsubscribeNetInfo = NetInfo.addEventListener(state => {
        const wasOffline = !isOnline;
        const isNowOnline = state.isConnected && state.isInternetReachable === true;
        setIsOnline(isNowOnline);
        
        console.log('[Offline] Network status:', {
          isConnected: state.isConnected,
          isInternetReachable: state.isInternetReachable,
          wasOffline,
          isNowOnline,
        });
        
        // Auto-sync when coming online
        if (wasOffline && isNowOnline && !isSyncing) {
          loadCachedData().then(() => {
            // Reload data to get latest pending count
            AsyncStorage.getItem(DETECTIONS_KEY).then(detectionsData => {
              AsyncStorage.getItem(TRAINING_IMAGES_KEY).then(trainingData => {
                const currentDetections: DetectionRecord[] = detectionsData ? JSON.parse(detectionsData) : [];
                const currentTraining: TrainingImageRecord[] = trainingData ? JSON.parse(trainingData) : [];
                const pendingCount = currentDetections.filter(d => !d.synced).length + currentTraining.filter(t => !t.synced).length;
                
                if (pendingCount > 0) {
                  console.log(`[Offline] 🔄 Auto-syncing ${pendingCount} pending items...`);
                  setTimeout(() => syncPendingData(), 1000); // Small delay to ensure network is ready
                }
              });
            });
          });
        }
      });
      
      // Initial state
      NetInfo.fetch().then(state => {
        setIsOnline(state.isConnected && state.isInternetReachable === true);
      });
      
      unsubscribe = () => unsubscribeNetInfo();
    }

    loadCachedData();

    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, []);

  // Auto-sync when coming online (additional check)
  useEffect(() => {
    if (isOnline && !isSyncing) {
      // Check for pending items and sync
      const pendingDetections = detections.filter(d => !d.synced).length;
      const pendingTraining = trainingImages.filter(t => !t.synced).length;
      const pendingCount = pendingDetections + pendingTraining;
      
      if (pendingCount > 0) {
        console.log(`[Offline] 🔄 Auto-sync trigger: ${pendingCount} pending items`);
        // Debounce to avoid multiple syncs
        const timeoutId = setTimeout(() => {
          syncPendingData();
        }, 2000);
        
        return () => clearTimeout(timeoutId);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOnline]);

  const loadCachedData = async () => {
    try {
      const [detectionsData, trainingData] = await Promise.all([
        AsyncStorage.getItem(DETECTIONS_KEY),
        AsyncStorage.getItem(TRAINING_IMAGES_KEY),
      ]);

      if (detectionsData) {
        setDetections(JSON.parse(detectionsData));
      }
      if (trainingData) {
        setTrainingImages(JSON.parse(trainingData));
      }

      console.log('[Offline] Cached data loaded');
    } catch (error) {
      console.error('[Offline] Error loading cached data:', error);
    }
  };



  const saveDetectionMutation = useMutation({
    mutationFn: async (detection: DetectionRecord) => {
      const updatedDetections = [...detections, detection];
      setDetections(updatedDetections);
      await AsyncStorage.setItem(DETECTIONS_KEY, JSON.stringify(updatedDetections));

      if (isOnline) {
        try {
          // Convert image to base64 if needed
          const imageBase64 = await convertImageToBase64(detection.imageUri);
          
          // Convert DiseaseType to backend format
          const backendDetection = {
            ...detection,
            imageUri: imageBase64, // Use converted base64 image
            primaryDisease: convertDiseaseTypeToBackend(detection.primaryDisease),
            detections: detection.detections.map(d => ({
              ...d,
              disease: convertDiseaseTypeToBackend(d.disease),
            })),
          };
          await trpcClient.detection.save.mutate(backendDetection);
          detection.synced = true;
          
          // Update storage with synced status
          const updated = updatedDetections.map(d => 
            d.id === detection.id ? { ...d, synced: true } : d
          );
          setDetections(updated);
          await AsyncStorage.setItem(DETECTIONS_KEY, JSON.stringify(updated));
          
          console.log('[Offline] ✅ Detection synced immediately:', detection.id);
        } catch (error) {
          console.error('[Offline] Failed to sync detection immediately:', error);
          detection.synced = false;
          // Will be retried in next sync
        }
      } else {
        detection.synced = false;
      }

      return detection;
    },
  });

  const saveTrainingImageMutation = useMutation({
    mutationFn: async (trainingImage: TrainingImageRecord) => {
      const updatedTraining = [...trainingImages, trainingImage];
      setTrainingImages(updatedTraining);
      await AsyncStorage.setItem(TRAINING_IMAGES_KEY, JSON.stringify(updatedTraining));

      if (isOnline) {
        try {
          // Convert image to base64 if needed
          const imageBase64 = await convertImageToBase64(trainingImage.imageUri);
          
          // Convert DiseaseType to backend format
          const backendTraining = {
            ...trainingImage,
            imageUri: imageBase64, // Use converted base64 image
            disease: convertDiseaseTypeToBackend(trainingImage.disease),
          };
          await trpcClient.training.save.mutate(backendTraining);
          trainingImage.synced = true;
          
          // Update storage with synced status
          const updated = updatedTraining.map(t => 
            t.id === trainingImage.id ? { ...t, synced: true } : t
          );
          setTrainingImages(updated);
          await AsyncStorage.setItem(TRAINING_IMAGES_KEY, JSON.stringify(updated));
          
          console.log('[Offline] ✅ Training image synced immediately:', trainingImage.id);
        } catch (error) {
          console.error('[Offline] Failed to sync training image immediately:', error);
          trainingImage.synced = false;
          // Will be retried in next sync
        }
      } else {
        trainingImage.synced = false;
      }

      return trainingImage;
    },
  });

  const getDetectionHistory = useCallback((diseaseType?: string) => {
    if (diseaseType) {
      return detections.filter(d => d.primaryDisease === diseaseType);
    }
    return detections;
  }, [detections]);

  const getTrainingImageCount = useCallback((diseaseType: string) => {
    return trainingImages.filter(t => t.disease === diseaseType).length;
  }, [trainingImages]);

  const getPendingSyncCount = useCallback(() => {
    const pendingDetections = detections.filter(d => !d.synced).length;
    const pendingTraining = trainingImages.filter(t => !t.synced).length;
    return pendingDetections + pendingTraining;
  }, [detections, trainingImages]);

  return useMemo(() => ({
    isOnline,
    detections,
    trainingImages,
    saveDetection: saveDetectionMutation.mutate,
    saveTrainingImage: saveTrainingImageMutation.mutate,
    getDetectionHistory,
    getTrainingImageCount,
    getPendingSyncCount,
    syncPendingData,
    isSaving: saveDetectionMutation.isPending || saveTrainingImageMutation.isPending,
    isSyncing,
    lastSyncTime,
  }), [isOnline, detections, trainingImages, saveDetectionMutation.mutate, saveDetectionMutation.isPending, saveTrainingImageMutation.mutate, saveTrainingImageMutation.isPending, getDetectionHistory, getTrainingImageCount, getPendingSyncCount, syncPendingData, isSyncing, lastSyncTime]);
});
