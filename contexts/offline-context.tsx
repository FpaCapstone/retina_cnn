import createContextHook from '@nkzw/create-context-hook';
import { useState, useEffect, useCallback, useMemo } from 'react';
import { Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { DetectionRecord, TrainingImageRecord } from '@/types/database';
import { useMutation } from '@tanstack/react-query';
import { trpcClient } from '@/lib/trpc';
import { convertDiseaseTypeToBackend } from '@/utils/dataset-utils';

const DETECTIONS_KEY = 'offline_detections';
const TRAINING_IMAGES_KEY = 'offline_training_images';

export const [OfflineProvider, useOffline] = createContextHook(() => {
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [detections, setDetections] = useState<DetectionRecord[]>([]);
  const [trainingImages, setTrainingImages] = useState<TrainingImageRecord[]>([]);

  const syncPendingData = useCallback(async () => {
    console.log('[Offline] Starting sync...');
    
    try {
      const detectionsData = await AsyncStorage.getItem(DETECTIONS_KEY);
      const trainingData = await AsyncStorage.getItem(TRAINING_IMAGES_KEY);
      
      const currentDetections: DetectionRecord[] = detectionsData ? JSON.parse(detectionsData) : [];
      const currentTraining: TrainingImageRecord[] = trainingData ? JSON.parse(trainingData) : [];
      
      const unsyncedDetections = currentDetections.filter(d => !d.synced);
      const unsyncedTraining = currentTraining.filter(t => !t.synced);

      for (const detection of unsyncedDetections) {
        try {
          // Convert DiseaseType to backend format
          const backendDetection = {
            ...detection,
            primaryDisease: convertDiseaseTypeToBackend(detection.primaryDisease),
            detections: detection.detections.map(d => ({
              ...d,
              disease: convertDiseaseTypeToBackend(d.disease),
            })),
          };
          await trpcClient.detection.save.mutate(backendDetection);
          detection.synced = true;
          console.log('[Offline] Synced detection:', detection.id);
        } catch (error) {
          console.error('[Offline] Failed to sync detection:', error);
        }
      }

      for (const training of unsyncedTraining) {
        try {
          // Convert DiseaseType to backend format
          const backendTraining = {
            ...training,
            disease: convertDiseaseTypeToBackend(training.disease),
          };
          await trpcClient.training.save.mutate(backendTraining);
          training.synced = true;
          console.log('[Offline] Synced training image:', training.id);
        } catch (error) {
          console.error('[Offline] Failed to sync training image:', error);
        }
      }

      await Promise.all([
        AsyncStorage.setItem(DETECTIONS_KEY, JSON.stringify(currentDetections)),
        AsyncStorage.setItem(TRAINING_IMAGES_KEY, JSON.stringify(currentTraining)),
      ]);
      
      setDetections(currentDetections);
      setTrainingImages(currentTraining);

      console.log('[Offline] Sync completed');
    } catch (error) {
      console.error('[Offline] Sync failed:', error);
    }
  }, []);

  useEffect(() => {
    let unsubscribe: (() => void) | undefined;

    if (Platform.OS === 'web') {
      const handleOnline = () => {
        console.log('[Offline] Web network status changed: online');
        setIsOnline(true);
        syncPendingData();
      };
      const handleOffline = () => {
        console.log('[Offline] Web network status changed: offline');
        setIsOnline(false);
      };
      window.addEventListener('online', handleOnline);
      window.addEventListener('offline', handleOffline);
      setIsOnline(typeof navigator !== 'undefined' ? navigator.onLine : true);
      if (typeof navigator !== 'undefined' && navigator.onLine) {
        syncPendingData();
      }
      unsubscribe = () => {
        window.removeEventListener('online', handleOnline);
        window.removeEventListener('offline', handleOffline);
      };
    } else {
      setIsOnline(true);
      syncPendingData();
    }

    loadCachedData();

    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, [syncPendingData]);

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
          // Convert DiseaseType to backend format
          const backendDetection = {
            ...detection,
            primaryDisease: convertDiseaseTypeToBackend(detection.primaryDisease),
            detections: detection.detections.map(d => ({
              ...d,
              disease: convertDiseaseTypeToBackend(d.disease),
            })),
          };
          await trpcClient.detection.save.mutate(backendDetection);
          detection.synced = true;
        } catch (error) {
          console.error('[Offline] Failed to sync detection immediately:', error);
          detection.synced = false;
        }
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
          // Convert DiseaseType to backend format
          const backendTraining = {
            ...trainingImage,
            disease: convertDiseaseTypeToBackend(trainingImage.disease),
          };
          await trpcClient.training.save.mutate(backendTraining);
          trainingImage.synced = true;
        } catch (error) {
          console.error('[Offline] Failed to sync training image immediately:', error);
          trainingImage.synced = false;
        }
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
  }), [isOnline, detections, trainingImages, saveDetectionMutation.mutate, saveDetectionMutation.isPending, saveTrainingImageMutation.mutate, saveTrainingImageMutation.isPending, getDetectionHistory, getTrainingImageCount, getPendingSyncCount, syncPendingData]);
});
