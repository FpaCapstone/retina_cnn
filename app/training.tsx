import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Image, Alert, ActivityIndicator, Platform } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Database, Upload, CheckCircle, XCircle, ArrowLeft, Info, Home } from 'lucide-react-native';
import { useRouter, Stack } from 'expo-router';

import { useState, useEffect } from 'react';
import * as ImagePicker from 'expo-image-picker';
import Colors from '@/constants/colors';
import { DISEASES } from '@/constants/diseases';
import { DiseaseType } from '@/types/disease';
import { TrainingImageRecord } from '@/types/database';
import { validateTrainingImage } from '@/utils/ml-analysis';
import { useOffline } from '@/contexts/offline-context';

export default function TrainingScreen() {
  const router = useRouter();
  const { saveTrainingImage, trainingImages, getTrainingImageCount } = useOffline();
  const [selectedDisease, setSelectedDisease] = useState<DiseaseType | null>(null);
  const [uploadedImages, setUploadedImages] = useState<TrainingImageRecord[]>([]);
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<{ valid: boolean; reason: string } | null>(null);

  // Load training images from offline context
  useEffect(() => {
    setUploadedImages(trainingImages);
  }, [trainingImages]);

  const handleSelectDisease = (diseaseId: DiseaseType) => {
    setSelectedDisease(diseaseId);
    setValidationResult(null);
  };

  const handleUploadImage = async () => {
    if (!selectedDisease) {
      Alert.alert('Select Disease', 'Please select a disease type first');
      return;
    }

    if (Platform.OS === 'web') {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      
      input.onchange = async (e: any) => {
        const file = e.target?.files?.[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = async (event) => {
            const imageUri = event.target?.result as string;
            
            setIsValidating(true);
            setValidationResult(null);

            try {
              const validation = await validateTrainingImage(imageUri, selectedDisease!);
              setValidationResult(validation);

              if (validation.valid) {
                const newImage: TrainingImageRecord = {
                  id: Date.now().toString(),
                  imageUri: imageUri,
                  disease: selectedDisease,
                  uploadedAt: new Date().toISOString(),
                  validated: true,
                  validationReason: validation.reason,
                  synced: false,
                };

                // Save to offline context (will sync automatically if online)
                saveTrainingImage(newImage);
                
                Alert.alert(
                  'Success!',
                  'Training image validated and added successfully',
                  [{ text: 'OK' }]
                );
              } else {
                Alert.alert(
                  'Validation Failed',
                  validation.reason,
                  [{ text: 'Try Another Image' }]
                );
              }
            } catch (error) {
              console.error('Validation error:', error);
              Alert.alert('Error', 'Failed to validate image. Please try again.');
            } finally {
              setIsValidating(false);
            }
          };
          reader.readAsDataURL(file);
        }
      };
      
      input.click();
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
      allowsMultipleSelection: false,
    });

    if (!result.canceled && result.assets[0]) {
      const imageUri = result.assets[0].uri;
      
      setIsValidating(true);
      setValidationResult(null);

      try {
        const validation = await validateTrainingImage(imageUri, selectedDisease!);
        setValidationResult(validation);

        if (validation.valid) {
          const newImage: TrainingImageRecord = {
            id: Date.now().toString(),
            imageUri: imageUri,
            disease: selectedDisease,
            uploadedAt: new Date().toISOString(),
            validated: true,
            validationReason: validation.reason,
            synced: false,
          };

          // Save to offline context (will sync automatically if online)
          saveTrainingImage(newImage);
          
          Alert.alert(
            'Success!',
            'Training image validated and added successfully',
            [{ text: 'OK' }]
          );
        } else {
          Alert.alert(
            'Validation Failed',
            validation.reason,
            [{ text: 'Try Another Image' }]
          );
        }
      } catch (error) {
        console.error('Validation error:', error);
        Alert.alert('Error', 'Failed to validate image. Please try again.');
      } finally {
        setIsValidating(false);
      }
    }
  };

  const handleGoBack = () => {
    router.back();
  };

  const getImageCountByDisease = (diseaseId: DiseaseType) => {
    return getTrainingImageCount(diseaseId);
  };

  return (
    <View style={styles.container}>
      <Stack.Screen 
        options={{
          title: 'Training Data',
          headerStyle: {
            backgroundColor: Colors.primary.purple,
          },
          headerTintColor: Colors.text.primary,
          headerLeft: () => (
            <View style={{ flexDirection: 'row', gap: 12, marginLeft: 8 }}>
              <TouchableOpacity onPress={() => router.push('/')} style={{ flexDirection: 'row', alignItems: 'center', gap: 6, paddingVertical: 6, paddingHorizontal: 12, backgroundColor: 'rgba(255, 255, 255, 0.15)', borderRadius: 12 }}>
                <Home size={18} color={Colors.text.primary} />
                <Text style={{ color: Colors.text.primary, fontSize: 14, fontWeight: '600' }}>Dashboard</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={handleGoBack} style={{ paddingVertical: 6 }}>
                <ArrowLeft size={24} color={Colors.text.primary} />
              </TouchableOpacity>
            </View>
          ),
        }}
      />
      
      <LinearGradient
        colors={[Colors.primary.purple, Colors.primary.teal, Colors.primary.green]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={StyleSheet.absoluteFillObject}
      />
      
      <View style={styles.safeArea}>
        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.header}>
            <View style={styles.iconCircle}>
              <Database size={48} color={Colors.text.primary} strokeWidth={2} />
            </View>
            <Text style={styles.title}>Training Data</Text>
            <Text style={styles.subtitle}>Upload patient images to improve AI accuracy</Text>
          </View>

          <View style={styles.infoCard}>
            <Info size={24} color={Colors.status.info} strokeWidth={2} />
            <View style={styles.infoContent}>
              <Text style={styles.infoTitle}>How Training Works</Text>
              <Text style={styles.infoText}>
                Upload verified patient images for each disease type. The AI will validate and learn from these images to improve detection accuracy over time.
              </Text>
            </View>
          </View>

          <View style={styles.statsCard}>
            <Text style={styles.statsTitle}>Training Dataset</Text>
            <View style={styles.statsGrid}>
              {DISEASES.map((disease) => {
                const count = getImageCountByDisease(disease.id);
                return (
                  <View key={disease.id} style={styles.statItem}>
                    <View style={[styles.statIcon, { backgroundColor: disease.color }]}>
                      <Text style={styles.statCount}>{count}</Text>
                    </View>
                    <Text style={styles.statLabel}>{disease.name}</Text>
                  </View>
                );
              })}
            </View>
          </View>

          <Text style={styles.sectionTitle}>Select Disease Type</Text>
          <View style={styles.diseaseSelector}>
            {DISEASES.map((disease) => {
              const isSelected = selectedDisease === disease.id;
              return (
                <TouchableOpacity
                  key={disease.id}
                  style={[
                    styles.diseaseOption,
                    {
                      backgroundColor: isSelected ? disease.color : 'rgba(255, 255, 255, 0.1)',
                      borderColor: isSelected ? disease.color : 'rgba(255, 255, 255, 0.3)',
                    }
                  ]}
                  onPress={() => handleSelectDisease(disease.id)}
                  activeOpacity={0.7}
                >
                  <Text style={[
                    styles.diseaseOptionText,
                    { opacity: isSelected ? 1 : 0.8 }
                  ]}>
                    {disease.name}
                  </Text>
                  {isSelected && (
                    <CheckCircle size={20} color={Colors.text.primary} strokeWidth={2.5} />
                  )}
                </TouchableOpacity>
              );
            })}
          </View>

          {isValidating && (
            <View style={styles.validationCard}>
              <ActivityIndicator size="large" color={Colors.text.primary} />
              <Text style={styles.validationText}>Validating image...</Text>
            </View>
          )}

          {validationResult && (
            <View style={[
              styles.validationCard,
              { 
                backgroundColor: validationResult.valid 
                  ? 'rgba(16, 185, 129, 0.2)' 
                  : 'rgba(239, 68, 68, 0.2)',
                borderColor: validationResult.valid 
                  ? Colors.status.success 
                  : Colors.status.error,
              }
            ]}>
              {validationResult.valid ? (
                <CheckCircle size={32} color={Colors.status.success} strokeWidth={2} />
              ) : (
                <XCircle size={32} color={Colors.status.error} strokeWidth={2} />
              )}
              <Text style={styles.validationTitle}>
                {validationResult.valid ? 'Valid Training Image' : 'Invalid Image'}
              </Text>
              <Text style={styles.validationReason}>{validationResult.reason}</Text>
            </View>
          )}

          <TouchableOpacity
            style={[
              styles.uploadButton,
              { opacity: selectedDisease ? 1 : 0.5 }
            ]}
            onPress={handleUploadImage}
            disabled={!selectedDisease || isValidating}
            activeOpacity={0.8}
          >
            <Upload size={24} color={Colors.text.primary} strokeWidth={2} />
            <Text style={styles.uploadButtonText}>
              {isValidating ? 'Validating...' : 'Upload Training Image'}
            </Text>
          </TouchableOpacity>

          {uploadedImages.length > 0 && (
            <View style={styles.recentSection}>
              <Text style={styles.sectionTitle}>Recent Uploads</Text>
              <View style={styles.imageGrid}>
                {uploadedImages.slice(-6).reverse().map((img) => {
                  const disease = DISEASES.find(d => d.id === img.disease);
                  return (
                    <View key={img.id} style={styles.imageItem}>
                      <Image source={{ uri: img.imageUri }} style={styles.thumbnailImage} />
                      <View style={[styles.imageBadge, { backgroundColor: disease?.color }]}>
                        <Text style={styles.imageBadgeText}>
                          {disease?.name.substring(0, 3).toUpperCase()}
                        </Text>
                      </View>
                    </View>
                  );
                })}
              </View>
            </View>
          )}

          <View style={styles.footer}>
            <Text style={styles.footerText}>Total Images: {uploadedImages.length}</Text>
            <Text style={styles.footerSubtext}>
              More training data = Higher accuracy
            </Text>
          </View>
        </ScrollView>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  iconCircle: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
    borderWidth: 3,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  title: {
    fontSize: 32,
    fontWeight: '800' as const,
    color: Colors.text.primary,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 15,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    textAlign: 'center',
  },
  infoCard: {
    flexDirection: 'row',
    backgroundColor: 'rgba(59, 130, 246, 0.15)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 24,
    borderWidth: 2,
    borderColor: 'rgba(59, 130, 246, 0.3)',
  },
  infoContent: {
    flex: 1,
    marginLeft: 12,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 6,
  },
  infoText: {
    fontSize: 13,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    lineHeight: 18,
  },
  statsCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 16,
    textAlign: 'center',
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  statCount: {
    fontSize: 24,
    fontWeight: '800' as const,
    color: Colors.text.primary,
  },
  statLabel: {
    fontSize: 12,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.9,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 16,
  },
  diseaseSelector: {
    gap: 12,
    marginBottom: 24,
  },
  diseaseOption: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
  },
  diseaseOptionText: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  validationCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  validationText: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    marginTop: 12,
  },
  validationTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginTop: 12,
    marginBottom: 8,
  },
  validationReason: {
    fontSize: 14,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    textAlign: 'center',
    lineHeight: 20,
  },
  uploadButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.primary.purple,
    borderRadius: 16,
    padding: 18,
    marginBottom: 32,
    gap: 12,
  },
  uploadButtonText: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  recentSection: {
    marginBottom: 24,
  },
  imageGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  imageItem: {
    width: '31%',
    aspectRatio: 1,
    position: 'relative',
  },
  thumbnailImage: {
    width: '100%',
    height: '100%',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  imageBadge: {
    position: 'absolute',
    bottom: 4,
    right: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 6,
  },
  imageBadgeText: {
    fontSize: 10,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  footer: {
    alignItems: 'center',
    paddingTop: 20,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.2)',
  },
  footerText: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  footerSubtext: {
    fontSize: 13,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.8,
    marginTop: 4,
  },
});
