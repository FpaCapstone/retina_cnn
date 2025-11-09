import { View, Text, StyleSheet, TouchableOpacity, ScrollView, ActivityIndicator, Image, Animated, Platform } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Camera, CheckCircle, AlertCircle, ArrowLeft, Info } from 'lucide-react-native';
import { useRouter, useLocalSearchParams, Stack } from 'expo-router';

import { useState, useEffect, useRef } from 'react';
import * as ImagePicker from 'expo-image-picker';
import Colors from '@/constants/colors';
import { DISEASES } from '@/constants/diseases';
import { DiseaseType } from '@/types/disease';
import { analyzeEyeImageHybrid, HybridAnalysisResult } from '@/utils/ml-analysis-hybrid';
import { analyzeEyeImageEnhanced, EnhancedAnalysisResult } from '@/utils/ml-analysis-enhanced';

type DetectionStep = 'instructions' | 'capture' | 'analyzing' | 'results';

export default function DetectScreen() {
  const router = useRouter();
  const { disease } = useLocalSearchParams<{ disease: DiseaseType }>();
  const [step, setStep] = useState<DetectionStep>('instructions');
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<HybridAnalysisResult | null>(null);
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.9)).current;

  const diseaseInfo = DISEASES.find(d => d.id === disease);

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
  }, [step, fadeAnim, scaleAnim]);

  if (!diseaseInfo) {
    return null;
  }

  const handleCaptureImage = async (useCamera: boolean) => {
    if (Platform.OS === 'web') {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      if (useCamera) {
        input.capture = 'environment' as any;
      }
      
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
      // Try enhanced pipeline first (better accuracy for camera images)
      try {
        console.log('[Detection] Attempting enhanced pipeline...');
        const enhancedResult = await analyzeEyeImageEnhanced(uri, {
          enableQualityCheck: true,
          enablePreprocessing: true,
          enableNormalFilter: true,
          enableDiseaseClassification: true,
          enableValidation: true,
        });
        
        // Convert enhanced result to hybrid format for display
        const hybridResult: HybridAnalysisResult = {
          disease: disease,
          confidence: enhancedResult.detections[0]?.confidence || 0.5,
          detected: enhancedResult.primaryDisease !== 'normal',
          timestamp: enhancedResult.timestamp,
          imageUri: enhancedResult.imageUri,
          details: enhancedResult.details,
          usedModel: 'Enhanced Pipeline',
          modelInfo: enhancedResult.qualityScore 
            ? `Quality Score: ${(enhancedResult.qualityScore * 100).toFixed(1)}% | ${enhancedResult.recommendation === 'retake' ? '⚠️ Retake recommended' : '✅ Quality acceptable'}`
            : 'Enhanced 5-stage pipeline',
          githubRepo: '',
        };
        
        setResult(hybridResult);
        setStep('results');
        return;
      } catch (enhancedError) {
        console.log('[Detection] Enhanced pipeline not available, using hybrid fallback:', enhancedError);
      }
      
      // Fallback to hybrid analysis
      const analysisResult = await analyzeEyeImageHybrid(uri, disease);
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
      <View style={[styles.iconCircle, { backgroundColor: diseaseInfo.color }]}>
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
        style={[styles.primaryButton, { backgroundColor: diseaseInfo.color }]}
        onPress={() => setStep('capture')}
        activeOpacity={0.8}
      >
        <Text style={styles.primaryButtonText}>Continue</Text>
      </TouchableOpacity>
    </Animated.View>
  );

  const renderCapture = () => (
    <Animated.View style={[styles.contentContainer, { opacity: fadeAnim, transform: [{ scale: scaleAnim }] }]}>
      <View style={[styles.iconCircle, { backgroundColor: diseaseInfo.color }]}>
        <Camera size={48} color={Colors.text.primary} strokeWidth={2} />
      </View>
      
      <Text style={styles.stepTitle}>Capture Image</Text>
      <Text style={styles.stepSubtitle}>Take or upload an eye image</Text>

      <View style={styles.captureOptions}>
        <TouchableOpacity
          style={[styles.captureButton, { borderColor: diseaseInfo.color }]}
          onPress={() => handleCaptureImage(true)}
          activeOpacity={0.7}
        >
          <Camera size={40} color={Colors.text.primary} strokeWidth={2} />
          <Text style={styles.captureButtonText}>Take Photo</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.captureButton, { borderColor: diseaseInfo.color }]}
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
      <ActivityIndicator size="large" color={diseaseInfo.color} />
      <Text style={[styles.stepTitle, { marginTop: 24 }]}>Analyzing Image</Text>
      <Text style={styles.stepSubtitle}>AI is processing your image...</Text>
      
      {imageUri && (
        <Image source={{ uri: imageUri }} style={styles.previewImage} />
      )}

      <View style={styles.analyzingSteps}>
        <Text style={styles.analyzingStep}>✓ Image uploaded</Text>
        <Text style={styles.analyzingStep}>⟳ Running neural analysis</Text>
        <Text style={styles.analyzingStep}>⟳ Detecting patterns</Text>
        <Text style={styles.analyzingStep}>⟳ Calculating confidence</Text>
      </View>
    </Animated.View>
  );

  const renderResults = () => {
    if (!result) return null;

    return (
      <Animated.View style={[styles.contentContainer, { opacity: fadeAnim, transform: [{ scale: scaleAnim }] }]}>
        <View style={[
          styles.iconCircle, 
          { backgroundColor: result.detected ? Colors.status.warning : Colors.status.success }
        ]}>
          {result.detected ? (
            <AlertCircle size={48} color={Colors.text.primary} strokeWidth={2} />
          ) : (
            <CheckCircle size={48} color={Colors.text.primary} strokeWidth={2} />
          )}
        </View>
        
        <Text style={styles.stepTitle}>
          {result.detected ? 'Detection Alert' : 'No Issues Detected'}
        </Text>
        <Text style={styles.stepSubtitle}>
          {result.detected 
            ? `Possible ${diseaseInfo.name} detected` 
            : `No signs of ${diseaseInfo.name}`}
        </Text>

        {imageUri && (
          <Image source={{ uri: imageUri }} style={styles.resultImage} />
        )}

        <View style={styles.modelBadge}>
          <Text style={styles.modelBadgeLabel}>Model:</Text>
          <Text style={styles.modelBadgeValue}>
            {result.usedModel === 'RETFound' ? '🏥 Medical-Grade' : '📱 On-Device'}
          </Text>
        </View>
        <Text style={styles.modelInfo}>{result.modelInfo}</Text>

        <View style={styles.resultCard}>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Confidence Level</Text>
            <Text style={styles.resultValue}>{(result.confidence * 100).toFixed(1)}%</Text>
          </View>
          
          <View style={styles.confidenceBar}>
            <View 
              style={[
                styles.confidenceFill, 
                { 
                  width: `${result.confidence * 100}%`,
                  backgroundColor: result.detected ? Colors.status.warning : Colors.status.success
                }
              ]} 
            />
          </View>

          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Disease Type</Text>
            <Text style={styles.resultValue}>{diseaseInfo.name}</Text>
          </View>

          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Analysis Time</Text>
            <Text style={styles.resultValue}>{new Date(result.timestamp).toLocaleTimeString()}</Text>
          </View>

          <View style={styles.detailsBox}>
            <Text style={styles.detailsTitle}>Details</Text>
            <Text style={styles.detailsText}>{result.details}</Text>
          </View>
        </View>

        <TouchableOpacity
          style={[styles.primaryButton, { backgroundColor: diseaseInfo.color }]}
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
          title: diseaseInfo.name,
          headerStyle: {
            backgroundColor: diseaseInfo.color,
          },
          headerTintColor: Colors.text.primary,
          headerLeft: () => (
            <TouchableOpacity onPress={handleGoHome} style={{ marginLeft: 8 }}>
              <ArrowLeft size={24} color={Colors.text.primary} />
            </TouchableOpacity>
          ),
        }}
      />
      
      <LinearGradient
        colors={[diseaseInfo.color, Colors.primary.teal, Colors.primary.green]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={StyleSheet.absoluteFillObject}
      />
      
      <View style={styles.safeArea}>
        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
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
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  contentContainer: {
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 16,
    padding: 24,
    marginVertical: 20,
  },
  iconCircle: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
    borderWidth: 3,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  stepTitle: {
    fontSize: 32,
    fontWeight: '800' as const,
    color: '#000000',
    marginBottom: 8,
    textAlign: 'center',
  },
  stepSubtitle: {
    fontSize: 16,
    fontWeight: '500' as const,
    color: '#333333',
    textAlign: 'center',
    marginBottom: 32,
  },
  instructionsList: {
    width: '100%',
    marginBottom: 32,
  },
  instructionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  instructionNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.primary.purple,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  instructionNumberText: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  instructionText: {
    flex: 1,
    fontSize: 15,
    fontWeight: '600' as const,
    color: '#000000',
  },
  captureOptions: {
    width: '100%',
    gap: 16,
    marginBottom: 24,
  },
  captureButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    borderWidth: 2,
  },
  captureButtonText: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginTop: 12,
  },
  primaryButton: {
    width: '100%',
    padding: 18,
    borderRadius: 16,
    alignItems: 'center',
    marginBottom: 12,
  },
  primaryButtonText: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  secondaryButton: {
    width: '100%',
    padding: 18,
    borderRadius: 16,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  secondaryButtonText: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text.primary,
  },
  previewImage: {
    width: 200,
    height: 200,
    borderRadius: 16,
    marginTop: 24,
    borderWidth: 3,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  analyzingSteps: {
    marginTop: 32,
    alignItems: 'flex-start',
    width: '100%',
  },
  analyzingStep: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    marginBottom: 12,
    opacity: 0.9,
  },
  resultImage: {
    width: 250,
    height: 250,
    borderRadius: 20,
    marginBottom: 24,
    borderWidth: 3,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  resultCard: {
    width: '100%',
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    padding: 20,
    marginBottom: 24,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  resultLabel: {
    fontSize: 15,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.8,
  },
  resultValue: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  confidenceBar: {
    width: '100%',
    height: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 4,
    marginBottom: 16,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 4,
  },
  detailsBox: {
    marginTop: 8,
    padding: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
  },
  detailsTitle: {
    fontSize: 15,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 8,
  },
  detailsText: {
    fontSize: 14,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    lineHeight: 20,
  },
  modelBadge: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  modelBadgeLabel: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.8,
  },
  modelBadgeValue: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  modelInfo: {
    fontSize: 12,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.8,
    textAlign: 'center',
    marginBottom: 16,
    paddingHorizontal: 20,
  },
});
