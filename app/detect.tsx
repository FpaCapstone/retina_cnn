import { View, Text, StyleSheet, TouchableOpacity, ScrollView, ActivityIndicator, Image, Animated, Platform } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Camera, ArrowLeft, Info, ScanEye } from 'lucide-react-native';
import { useRouter, Stack } from 'expo-router';
import { useState, useEffect, useRef } from 'react';
import * as ImagePicker from 'expo-image-picker';
import Colors from '@/constants/colors';
import { DISEASES } from '@/constants/diseases';
import { analyzeEyeImage } from '@/utils/ml-analysis';
import { AnalysisResult, DiseaseDetection } from '@/types/disease';
import { useOffline } from '@/contexts/offline-context';

type DetectionStep = 'instructions' | 'capture' | 'analyzing' | 'results';

export default function DetectScreen() {
  const router = useRouter();
  const { saveDetection } = useOffline();

  const [step, setStep] = useState<DetectionStep>('instructions');
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.9)).current;

  useEffect(() => {
    fadeAnim.setValue(0);
    scaleAnim.setValue(0.9);
    
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }),
    ]).start();
  }, [step]);

  const handleCaptureImage = async (useCamera: boolean) => {
    if (Platform.OS === 'web') {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      if (useCamera) input.capture = 'environment' as any;
      
      input.onchange = async (e: any) => {
        const file = e.target?.files?.[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = async (event) => {
            const uri = event.target?.result as string;
            setImageUri(uri);
            setStep('analyzing');
            await performAnalysis(uri);
          };
          reader.readAsDataURL(file);
        }
      };
      input.click();
      return;
    }

    if (useCamera) {
      const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
      if (!permissionResult.granted) {
        alert('Camera permission is required to take photos');
        return;
      }
    }

    const result = useCamera
      ? await ImagePicker.launchCameraAsync({
          mediaTypes: ['images'],
          allowsEditing: true,
          aspect: [1, 1],
          quality: 1,
        })
      : await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ['images'],
          allowsEditing: true,
          aspect: [1, 1],
          quality: 1,
        });

    if (!result.canceled && result.assets[0]) {
      setImageUri(result.assets[0].uri);
      setStep('analyzing');
      await performAnalysis(result.assets[0].uri);
    }
  };

  const performAnalysis = async (uri: string) => {
    try {
      const analysisResult = await analyzeEyeImage(uri);

      // ✅ Save detection locally for offline sync
      const detectionRecord = {
        id: Date.now().toString(),
        imageUri: uri,
        primaryDisease: analysisResult.primaryDisease,
        detections: analysisResult.detections,
        timestamp: new Date().toISOString(),
        details: analysisResult.details,
        uploadedAt: new Date().toISOString(),
        synced: false,
      };
      saveDetection(detectionRecord);

      setResult(analysisResult);
      setStep('results');
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Failed to analyze image. Please try again.');
      setStep('capture');
    }
  };

  const handleRetry = () => {
    setStep('instructions');
    setImageUri(null);
    setResult(null);
  };

  const handleGoHome = () => {
    router.back();
  };

  const renderInstructions = () => (
    <Animated.View style={[styles.contentContainer, { opacity: fadeAnim, transform: [{ scale: scaleAnim }] }]}>
      <View style={[styles.iconCircle, { backgroundColor: Colors.primary.purple }]}>
        <Info size={48} color={Colors.text.primary} strokeWidth={2} />
      </View>
      
      <Text style={styles.stepTitle}>Instructions</Text>
      <Text style={styles.stepSubtitle}>How to capture the eye image</Text>

      <View style={styles.instructionsList}>
        <View style={styles.instructionItem}>
          <View style={styles.instructionNumber}>
            <Text style={styles.instructionNumberText}>1</Text>
          </View>
          <Text style={styles.instructionText}>Ensure good lighting conditions</Text>
        </View>

        <View style={styles.instructionItem}>
          <View style={styles.instructionNumber}>
            <Text style={styles.instructionNumberText}>2</Text>
          </View>
          <Text style={styles.instructionText}>Keep the camera steady and focused</Text>
        </View>

        <View style={styles.instructionItem}>
          <View style={styles.instructionNumber}>
            <Text style={styles.instructionNumberText}>3</Text>
          </View>
          <Text style={styles.instructionText}>Capture a clear image of the eye</Text>
        </View>

        <View style={styles.instructionItem}>
          <View style={styles.instructionNumber}>
            <Text style={styles.instructionNumberText}>4</Text>
          </View>
          <Text style={styles.instructionText}>Avoid reflections and shadows</Text>
        </View>
      </View>

      <TouchableOpacity
        style={[styles.primaryButton, { backgroundColor: Colors.primary.purple }]}
        onPress={() => setStep('capture')}
        activeOpacity={0.8}
      >
        <Text style={styles.primaryButtonText}>Continue</Text>
      </TouchableOpacity>
    </Animated.View>
  );

  const renderCapture = () => (
    <Animated.View style={[styles.contentContainer, { opacity: fadeAnim, transform: [{ scale: scaleAnim }] }]}>
      <View style={[styles.iconCircle, { backgroundColor: Colors.primary.purple }]}>
        <Camera size={48} color={Colors.text.primary} strokeWidth={2} />
      </View>
      
      <Text style={styles.stepTitle}>Capture Image</Text>
      <Text style={styles.stepSubtitle}>Take or upload an eye image</Text>

      <View style={styles.captureOptions}>
        <TouchableOpacity
          style={[styles.captureButton, { borderColor: Colors.primary.purple }]}
          onPress={() => handleCaptureImage(true)}
          activeOpacity={0.7}
        >
          <Camera size={40} color={Colors.text.primary} strokeWidth={2} />
          <Text style={styles.captureButtonText}>Take Photo</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.captureButton, { borderColor: Colors.primary.purple }]}
          onPress={() => handleCaptureImage(false)}
          activeOpacity={0.7}
        >
          <Image 
            source={{ uri: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMjEgMTVWMTlBMiAyIDAgMCAxIDE5IDIxSDVBMiAyIDAgMCAxIDMgMTlWMTVNMTcgOEwxMiAzTTEyIDNMNyA4TTEyIDNWMTUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PC9zdmc+' }}
            style={{ width: 40, height: 40 }}
          />
          <Text style={styles.captureButtonText}>Upload Photo</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity
        style={styles.secondaryButton}
        onPress={() => setStep('instructions')}
        activeOpacity={0.8}
      >
        <Text style={styles.secondaryButtonText}>Back to Instructions</Text>
      </TouchableOpacity>
    </Animated.View>
  );

  const renderAnalyzing = () => (
    <Animated.View style={[styles.contentContainer, { opacity: fadeAnim }]}>
      <ActivityIndicator size="large" color={Colors.primary.purple} />
      <Text style={[styles.stepTitle, { marginTop: 24 }]}>Analyzing Image</Text>
      <Text style={styles.stepSubtitle}>AI is processing your image...</Text>
      
      {imageUri && (
        <Image source={{ uri: imageUri }} style={styles.previewImage} />
      )}

      <View style={styles.analyzingSteps}>
        <Text style={styles.analyzingStep}>✓ Image uploaded</Text>
        <Text style={styles.analyzingStep}>⟳ Running neural analysis</Text>
        <Text style={styles.analyzingStep}>⟳ Detecting patterns</Text>
        <Text style={styles.analyzingStep}>⟳ Calculating probabilities</Text>
      </View>
    </Animated.View>
  );

  const renderResults = () => {
    if (!result) return null;
    const primaryDisease = DISEASES.find(d => d.id === result.primaryDisease);
    const topDetections = result.detections.slice(0, 3);

    return (
      <Animated.View style={[styles.contentContainer, { opacity: fadeAnim, transform: [{ scale: scaleAnim }] }]}>
        <View style={[styles.iconCircle, { backgroundColor: primaryDisease?.color || Colors.primary.purple }]}>
          <ScanEye size={48} color="#FFFFFF" strokeWidth={2} />
        </View>
        
        <Text style={styles.stepTitle}>Analysis Complete</Text>
        <Text style={styles.stepSubtitle}>Multi-disease detection results</Text>

        {imageUri && <Image source={{ uri: imageUri }} style={styles.resultImage} />}

        <View style={styles.resultCard}>
          <Text style={styles.resultCardTitle}>Primary Finding</Text>
          <View style={[styles.primaryDiseaseCard, { borderColor: primaryDisease?.color }]}>
            <Text style={styles.primaryDiseaseName}>{primaryDisease?.name}</Text>
            <Text style={styles.primaryDiseasePercentage}>
              {result.detections[0].percentage.toFixed(1)}%
            </Text>
          </View>
          <Text style={styles.detailsText}>{result.details}</Text>
        </View>

        <View style={styles.resultCard}>
          <Text style={styles.resultCardTitle}>Top 3 Detections</Text>
          <Text style={styles.resultCardSubtitle}>Highest independent confidence scores</Text>
          
          {topDetections.map((detection: DiseaseDetection) => {
            const disease = DISEASES.find(d => d.id === detection.disease);
            return (
              <View key={detection.disease} style={styles.detectionRow}>
                <View style={styles.detectionLeft}>
                  <View style={[styles.diseaseIndicator, { backgroundColor: disease?.color }]} />
                  <Text style={styles.diseaseName}>{disease?.name}</Text>
                </View>
                <View style={styles.detectionRight}>
                  <View style={styles.progressBarContainer}>
                    <View 
                      style={[
                        styles.progressBarFill,
                        { width: `${detection.percentage}%`, backgroundColor: disease?.color },
                      ]}
                    />
                  </View>
                  <Text style={styles.percentageText}>{detection.percentage.toFixed(1)}%</Text>
                </View>
              </View>
            );
          })}
        </View>

        <View style={styles.modelBadge}>
          <Text style={styles.modelBadgeText}>
            {result.usedModel === 'backend' || result.usedModel === 'enhanced' 
              ? '🌐 Backend AI Model (Online)' 
              : result.usedModel === 'tflite'
              ? '📱 On-Device AI Model (Offline)'
              : '⚙️ Offline Analysis'}
          </Text>
          {result.modelInfo && (
            <Text style={styles.modelBadgeSubtext}>{result.modelInfo}</Text>
          )}
        </View>

        <TouchableOpacity
          style={[styles.primaryButton, { backgroundColor: Colors.primary.purple }]}
          onPress={handleRetry}
          activeOpacity={0.8}
        >
          <Text style={styles.primaryButtonText}>Analyze Another Image</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={handleGoHome}
          activeOpacity={0.8}
        >
          <Text style={styles.secondaryButtonText}>Back to Home</Text>
        </TouchableOpacity>
      </Animated.View>
    );
  };

  return (
    <View style={styles.container}>
      <Stack.Screen 
        options={{
          title: 'Eye Disease Detection',
          headerStyle: { backgroundColor: Colors.primary.purple },
          headerTintColor: Colors.text.primary,
          headerLeft: () => (
            <TouchableOpacity onPress={handleGoHome} style={{ marginLeft: 8 }}>
              <ArrowLeft size={24} color={Colors.text.primary} />
            </TouchableOpacity>
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
        <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
          {step === 'instructions' && renderInstructions()}
          {step === 'capture' && renderCapture()}
          {step === 'analyzing' && renderAnalyzing()}
          {step === 'results' && renderResults()}
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
    flexGrow: 1,
    padding: 20,
  },
  contentContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    marginVertical: 20,
  },
  iconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  stepTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#000000',
    marginBottom: 8,
  },
  stepSubtitle: {
    fontSize: 16,
    color: '#333333',
    marginBottom: 24,
    textAlign: 'center',
  },
  primaryButton: {
    width: '100%',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 24,
  },
  primaryButtonText: {
    color: Colors.text.primary,
    fontSize: 16,
    fontWeight: 'bold',
  },
  secondaryButton: {
    padding: 16,
    marginTop: 12,
  },
  secondaryButtonText: {
    color: Colors.text.primary,
    fontSize: 16,
  },
  instructionsList: {
    width: '100%',
    marginVertical: 24,
  },
  instructionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  instructionNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.primary.purple,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  instructionNumberText: {
    color: Colors.text.primary,
    fontSize: 16,
    fontWeight: 'bold',
  },
  instructionText: {
    color: '#000000',
    fontSize: 16,
    flex: 1,
  },
  captureOptions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginVertical: 24,
  },
  captureButton: {
    width: '48%',
    aspectRatio: 1,
    borderRadius: 16,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  captureButtonText: {
    color: Colors.text.primary,
    marginTop: 12,
    fontSize: 14,
  },
  previewImage: {
    width: 200,
    height: 200,
    borderRadius: 16,
    marginVertical: 24,
  },
  analyzingSteps: {
    width: '100%',
    marginTop: 24,
  },
  analyzingStep: {
    color: Colors.text.primary,
    fontSize: 14,
    marginBottom: 8,
  },
  resultImage: {
    width: '100%',
    height: 200,
    borderRadius: 16,
    marginVertical: 24,
  },
  resultCard: {
    width: '100%',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: 'rgba(0, 0, 0, 0.1)',
  },
  resultCardTitle: {
    color: '#000000',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  resultCardSubtitle: {
    color: '#333333',
    fontSize: 14,
    marginBottom: 16,
  },
  primaryDiseaseCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    borderRadius: 8,
    borderWidth: 2,
    marginVertical: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
  },
  primaryDiseaseName: {
    color: '#000000',
    fontSize: 16,
    fontWeight: 'bold',
  },
  primaryDiseasePercentage: {
    color: '#000000',
    fontSize: 16,
    fontWeight: 'bold',
  },
  detailsText: {
    color: '#333333',
    fontSize: 14,
    marginTop: 12,
  },
  detectionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  detectionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  diseaseIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  diseaseName: {
    color: '#000000',
    fontSize: 14,
  },
  detectionRight: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
  },
  progressBarContainer: {
    flex: 1,
    height: 6,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 3,
    marginRight: 8,
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 3,
  },
  percentageText: {
    color: '#000000',
    fontSize: 14,
    width: 50,
    textAlign: 'right',
  },
  modelBadge: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 16,
    marginTop: 24,
    borderWidth: 1,
    borderColor: 'rgba(0, 0, 0, 0.2)',
  },
  modelBadgeText: {
    color: '#000000',
    fontSize: 12,
    fontWeight: '600' as const,
    marginBottom: 2,
  },
  modelBadgeSubtext: {
    color: '#666666',
    fontSize: 10,
  },
});
