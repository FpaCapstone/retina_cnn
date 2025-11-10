import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Animated } from 'react-native';
import { Image } from 'expo-image';
import { LinearGradient } from 'expo-linear-gradient';
import { Activity, Database, Microscope, AlertCircle, TrendingUp, ScanEye, BarChart3, AlertTriangle, Home } from 'lucide-react-native';
import { useRouter, Stack } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import Colors from '@/constants/colors';
import OfflineIndicator from '@/components/OfflineIndicator';
import { getModelMetrics, formatPercentage } from '@/utils/model-metrics';

type DashboardMode = 'detection' | 'training' | 'info';

export default function HomeScreen() {
  console.log('[HomeScreen] Component mounted');
  const router = useRouter();
  const [mode, setMode] = useState<DashboardMode>('detection');
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;

  useEffect(() => {
    fadeAnim.setValue(0);
    slideAnim.setValue(50);
    
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 600,
        useNativeDriver: true,
      }),
    ]).start();
  }, [mode]);

  const handleDetectionPress = () => {
    router.push('/detect');
  };

  const handleTrainingPress = () => {
    router.push('/training');
  };

  const handleInfoPress = () => {
    setMode('info');
  };

  // Handle navigation from dashboard
  useEffect(() => {
    // Check if we should show info tab (from dashboard navigation)
    // This can be enhanced with deep linking if needed
  }, []);

  return (
    <View style={styles.container}>
      <Stack.Screen options={{ headerShown: false }} />
      
      <OfflineIndicator />
      
      <LinearGradient
        colors={[Colors.primary.purple, Colors.primary.teal, Colors.primary.green]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={StyleSheet.absoluteFillObject}
      />
      
      <View style={styles.backgroundExtension} />
      
      <View style={styles.safeArea}>
        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <Animated.View style={[styles.header, { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }]}>
            <TouchableOpacity 
              style={styles.dashboardButton}
              onPress={() => router.push('/')}
              activeOpacity={0.8}
            >
              <Home size={20} color={Colors.text.primary} strokeWidth={2.5} />
              <Text style={styles.dashboardButtonText}>Dashboard</Text>
            </TouchableOpacity>
            <View style={styles.logoContainer}>
              <Image 
                source={require('@/assets/images/retina_logo.png')} 
                style={styles.logoImage}
                resizeMode="contain"
              />
            </View>
            <Text style={styles.title}>RETINA</Text>
            <Text style={styles.subtitle}>Real-Time Eye-Disease Testing</Text>
            <Text style={styles.subtitleSecondary}>with Intelligent Neural Analysis</Text>
          </Animated.View>

          <View style={styles.modeSelector}>
            <TouchableOpacity
              style={[
                styles.modeTab,
                mode === 'detection' && styles.modeTabActive
              ]}
              onPress={() => setMode('detection')}
              activeOpacity={0.7}
            >
              <Microscope size={20} color={Colors.text.primary} strokeWidth={2.5} />
              <Text style={[
                styles.modeTabText,
                mode === 'detection' && styles.modeTabTextActive
              ]}>Detection</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.modeTab,
                mode === 'training' && styles.modeTabActive
              ]}
              onPress={() => setMode('training')}
              activeOpacity={0.7}
            >
              <Database size={20} color={Colors.text.primary} strokeWidth={2.5} />
              <Text style={[
                styles.modeTabText,
                mode === 'training' && styles.modeTabTextActive
              ]}>Training</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.modeTab,
                mode === 'info' && styles.modeTabActive
              ]}
              onPress={handleInfoPress}
              activeOpacity={0.7}
            >
              <AlertCircle size={20} color={Colors.text.primary} strokeWidth={2.5} />
              <Text style={[
                styles.modeTabText,
                mode === 'info' && styles.modeTabTextActive
              ]}>Info</Text>
            </TouchableOpacity>
          </View>

          {mode === 'detection' && (
            <Animated.View style={[styles.cardsContainer, { opacity: fadeAnim }]}>
              <Text style={styles.sectionTitle}>Eye Disease Detection</Text>
              <Text style={styles.sectionDescription}>AI-powered detection across 5 classes from a single image</Text>
              
              <View style={styles.detectionCard}>
                <View style={[styles.detectionIconCircle, { backgroundColor: `${Colors.primary.purple}50`, borderColor: `${Colors.primary.purple}CC` }]}>
                  <ScanEye size={48} color={Colors.primary.purple} strokeWidth={3} />
                </View>
                <Text style={styles.detectionCardTitle}>Scan Eye for Diseases</Text>
                <Text style={styles.detectionCardDescription}>
                  Upload or capture an eye image to detect Normal, Uveitis, Conjunctivitis, Cataract, and Eyelid Drooping with percentage confidence.
                </Text>
                
                <View style={styles.diseaseGrid}>
                  <View style={styles.diseaseTag}>
                    <Text style={styles.diseaseTagText}>Normal</Text>
                  </View>
                  <View style={styles.diseaseTag}>
                    <Text style={styles.diseaseTagText}>Uveitis</Text>
                  </View>
                  <View style={styles.diseaseTag}>
                    <Text style={styles.diseaseTagText}>Conjunctivitis</Text>
                  </View>
                  <View style={styles.diseaseTag}>
                    <Text style={styles.diseaseTagText}>Cataract</Text>
                  </View>
                  <View style={styles.diseaseTag}>
                    <Text style={styles.diseaseTagText}>Eyelid Drooping</Text>
                  </View>
                </View>

                <TouchableOpacity
                  style={styles.detectionButton}
                  onPress={handleDetectionPress}
                  activeOpacity={0.8}
                >
                  <ScanEye size={24} color={Colors.text.primary} strokeWidth={2.5} />
                  <Text style={styles.detectionButtonText}>Start Detection</Text>
                </TouchableOpacity>
              </View>

              <View style={styles.featuresCard}>
                <Text style={styles.featuresTitle}>Detection Features</Text>
                <View style={styles.featureItem}>
                  <Text style={styles.featureBullet}>✓</Text>
                  <Text style={styles.featureText}>Multi-disease detection in one scan</Text>
                </View>
                <View style={styles.featureItem}>
                  <Text style={styles.featureBullet}>✓</Text>
                  <Text style={styles.featureText}>Percentage confidence for each condition</Text>
                </View>
                <View style={styles.featureItem}>
                  <Text style={styles.featureBullet}>✓</Text>
                  <Text style={styles.featureText}>Works offline with on-device AI</Text>
                </View>
                <View style={styles.featureItem}>
                  <Text style={styles.featureBullet}>✓</Text>
                  <Text style={styles.featureText}>Medical-grade accuracy when online</Text>
                </View>
              </View>
            </Animated.View>
          )}

          {mode === 'training' && (
            <Animated.View style={[styles.cardsContainer, { opacity: fadeAnim }]}>
              <Text style={styles.sectionTitle}>Training Data Management</Text>
              <Text style={styles.sectionDescription}>Upload patient images to improve AI accuracy</Text>

              <View style={styles.trainingCard}>
                <View style={[styles.trainingIconCircle, { backgroundColor: `${Colors.primary.teal}50`, borderColor: `${Colors.primary.teal}CC` }]}>
                  <Database size={40} color={Colors.primary.teal} strokeWidth={3} />
                </View>
                <Text style={styles.trainingCardTitle}>Upload Training Images</Text>
                <Text style={styles.trainingCardDescription}>
                  Add verified patient images to the training dataset. The more data provided, the higher the AI accuracy becomes.
                </Text>
                <TouchableOpacity
                  style={styles.trainingButton}
                  onPress={handleTrainingPress}
                  activeOpacity={0.8}
                >
                  <TrendingUp size={20} color={Colors.text.primary} strokeWidth={2.5} />
                  <Text style={styles.trainingButtonText}>Start Training</Text>
                </TouchableOpacity>
              </View>

              <View style={styles.benefitsCard}>
                <Text style={styles.benefitsTitle}>Benefits of Training</Text>
                <View style={styles.benefitItem}>
                  <Text style={styles.benefitBullet}>✓</Text>
                  <Text style={styles.benefitText}>Increased detection accuracy</Text>
                </View>
                <View style={styles.benefitItem}>
                  <Text style={styles.benefitBullet}>✓</Text>
                  <Text style={styles.benefitText}>Better pattern recognition</Text>
                </View>
                <View style={styles.benefitItem}>
                  <Text style={styles.benefitBullet}>✓</Text>
                  <Text style={styles.benefitText}>Reduced false positives</Text>
                </View>
                <View style={styles.benefitItem}>
                  <Text style={styles.benefitBullet}>✓</Text>
                  <Text style={styles.benefitText}>Improved confidence scores</Text>
                </View>
              </View>
            </Animated.View>
          )}

          {mode === 'info' && (
            <Animated.View style={[styles.cardsContainer, { opacity: fadeAnim }]}>
              <Text style={styles.sectionTitle}>Model Information</Text>
              <Text style={styles.sectionDescription}>Performance metrics and dataset overview</Text>
              
              {/* Medical Disclaimer */}
              <View style={styles.disclaimerCard}>
                <View style={styles.disclaimerHeader}>
                  <AlertTriangle size={24} color={Colors.status.warning} strokeWidth={2.5} />
                  <Text style={styles.disclaimerTitle}>Medical Disclaimer</Text>
                </View>
                <Text style={styles.disclaimerText}>
                  This application is for informational and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or seen in this application.
                </Text>
                <Text style={styles.disclaimerText}>
                  The AI model predictions are not intended to diagnose, treat, cure, or prevent any disease. Results should be interpreted by qualified medical professionals. The accuracy of predictions may vary and should not be the sole basis for medical decisions.
                </Text>
              </View>

              {/* Model Metrics */}
              <View style={styles.metricsCard}>
                <View style={styles.metricsHeader}>
                  <BarChart3 size={24} color={Colors.text.primary} strokeWidth={2} />
                  <Text style={styles.metricsTitle}>Model Performance Metrics</Text>
                </View>
                
                {(() => {
                  const metrics = getModelMetrics();
                  return (
                    <>
                      <View style={styles.overallMetric}>
                        <Text style={styles.overallMetricLabel}>Overall Accuracy</Text>
                        <Text style={styles.overallMetricValue}>
                          {formatPercentage(metrics.overallAccuracy)}
                        </Text>
                        <Text style={styles.overallMetricSubtext}>
                          Tested on {metrics.testSetSize} images
                        </Text>
                      </View>

                      <View style={styles.metricsSection}>
                        <Text style={styles.metricsSectionTitle}>Per-Class Performance</Text>
                        {Object.entries(metrics.classes).map(([className, classMetrics]) => (
                          <View key={className} style={styles.classMetricRow}>
                            <View style={styles.classMetricLeft}>
                              <Text style={styles.classMetricName}>{className}</Text>
                              <Text style={styles.classMetricSupport}>
                                {classMetrics.support} samples
                              </Text>
                            </View>
                            <View style={styles.classMetricRight}>
                              <View style={styles.metricItem}>
                                <Text style={styles.metricLabel}>Precision</Text>
                                <Text style={styles.metricValue}>
                                  {formatPercentage(classMetrics.precision)}
                                </Text>
                              </View>
                              <View style={styles.metricItem}>
                                <Text style={styles.metricLabel}>Recall</Text>
                                <Text style={styles.metricValue}>
                                  {formatPercentage(classMetrics.recall)}
                                </Text>
                              </View>
                              <View style={styles.metricItem}>
                                <Text style={styles.metricLabel}>F1-Score</Text>
                                <Text style={styles.metricValue}>
                                  {formatPercentage(classMetrics.f1Score)}
                                </Text>
                              </View>
                            </View>
                          </View>
                        ))}
                      </View>

                      <View style={styles.avgMetrics}>
                        <View style={styles.avgMetricItem}>
                          <Text style={styles.avgMetricLabel}>Macro Average</Text>
                          <Text style={styles.avgMetricValue}>
                            {formatPercentage(metrics.macroAvg.f1Score)}
                          </Text>
                        </View>
                        <View style={styles.avgMetricItem}>
                          <Text style={styles.avgMetricLabel}>Weighted Average</Text>
                          <Text style={styles.avgMetricValue}>
                            {formatPercentage(metrics.weightedAvg.f1Score)}
                          </Text>
                        </View>
                      </View>
                    </>
                  );
                })()}
              </View>

              {/* Dataset Info */}
              <View style={styles.infoCard}>
                <Text style={styles.infoSectionTitle}>Dataset Overview</Text>
                <Text style={styles.infoDescription}>
                  This dataset contains images and symptom descriptions for Normal, Uveitis, Conjunctivitis, Cataract, and Eyelid Drooping.
                </Text>
                <Text style={styles.infoDescription}>
                  Images were collected from public online sources and quality-checked. SMOTE was applied to balance categories to 649 images each.
                </Text>
                
                <View style={styles.infoSection}>
                  <Text style={styles.infoSectionTitle}>Symptoms Reference:</Text>
                  <Text style={styles.infoListItem}>• Normal: No abnormalities, clear vision</Text>
                  <Text style={styles.infoListItem}>• Uveitis: Redness, pain, blurred vision, photophobia, floaters</Text>
                  <Text style={styles.infoListItem}>• Conjunctivitis: Redness, itching, tearing, discharge, crusting</Text>
                  <Text style={styles.infoListItem}>• Cataract: Cloudy/blurred vision, night difficulty, glare sensitivity</Text>
                  <Text style={styles.infoListItem}>• Eyelid Drooping: Drooping lids, swelling, irritation, eyelid lumps</Text>
                </View>

                <View style={styles.infoSection}>
                  <Text style={styles.infoSectionTitle}>AI Model:</Text>
                  <Text style={styles.infoDescription}>
                    On-device multi-class model returns percentage scores per class and reports the top prediction. Designed for mobile images and web compatibility.
                  </Text>
                </View>
              </View>
            </Animated.View>
          )}

          <View style={styles.footer}>
            <Text style={styles.footerText}>AI-Powered Medical Diagnosis</Text>
            <Text style={styles.footerSubtext}>For professional use only</Text>
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
  backgroundExtension: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 100,
    backgroundColor: Colors.primary.purple,
  },
  safeArea: {
    flex: 1,
    paddingTop: 50,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center',
    paddingTop: 40,
    paddingBottom: 20,
  },
  logoContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
    overflow: 'hidden',
  },
  logoImage: {
    width: '100%',
    height: '100%',
  },
  dashboardButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 20,
    paddingVertical: 10,
    paddingHorizontal: 16,
    marginBottom: 16,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
    alignSelf: 'center',
  },
  dashboardButtonText: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text.primary,
  },
  title: {
    fontSize: 48,
    fontWeight: '800' as const,
    color: Colors.text.primary,
    letterSpacing: 2,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.95,
    textAlign: 'center',
  },
  subtitleSecondary: {
    fontSize: 14,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.8,
    textAlign: 'center',
    marginTop: 4,
  },
  modeSelector: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 4,
    marginTop: 24,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  modeTab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderRadius: 12,
    gap: 6,
  },
  modeTabActive: {
    backgroundColor: 'rgba(255, 255, 255, 0.25)',
  },
  modeTabText: {
    fontSize: 13,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.7,
  },
  modeTabTextActive: {
    opacity: 1,
    fontWeight: '700' as const,
  },
  cardsContainer: {
    marginTop: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 8,
    textAlign: 'center',
  },
  sectionDescription: {
    fontSize: 14,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.8,
    textAlign: 'center',
    marginBottom: 20,
  },
  detectionCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 20,
    padding: 24,
    marginBottom: 16,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  detectionIconCircle: {
    width: 96,
    height: 96,
    borderRadius: 48,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    borderWidth: 3,
  },
  detectionCardTitle: {
    fontSize: 24,
    fontWeight: '800' as const,
    color: Colors.text.primary,
    marginBottom: 12,
    textAlign: 'center',
  },
  detectionCardDescription: {
    fontSize: 14,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 20,
  },
  diseaseGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 8,
    marginBottom: 24,
  },
  diseaseTag: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 12,
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  diseaseTagText: {
    fontSize: 13,
    fontWeight: '600' as const,
    color: Colors.text.primary,
  },
  detectionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: Colors.primary.purple,
    borderRadius: 16,
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  detectionButtonText: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  featuresCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  featuresTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 16,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  featureBullet: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.status.success,
    marginRight: 12,
    width: 20,
  },
  featureText: {
    flex: 1,
    fontSize: 15,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.9,
  },
  trainingCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 20,
    padding: 24,
    marginBottom: 16,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  trainingIconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
    borderWidth: 3,
  },
  trainingCardTitle: {
    fontSize: 22,
    fontWeight: '800' as const,
    color: Colors.text.primary,
    marginBottom: 12,
    textAlign: 'center',
  },
  trainingCardDescription: {
    fontSize: 14,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 20,
  },
  trainingButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: Colors.primary.purple,
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 24,
  },
  trainingButtonText: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  benefitsCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  benefitsTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 16,
  },
  benefitItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  benefitBullet: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.status.success,
    marginRight: 12,
    width: 20,
  },
  benefitText: {
    flex: 1,
    fontSize: 15,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.9,
  },
  infoCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 20,
    padding: 20,
    marginBottom: 16,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  infoDescription: {
    fontSize: 14,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    lineHeight: 20,
    marginBottom: 16,
  },
  infoSection: {
    marginTop: 12,
  },
  infoSectionTitle: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 8,
  },
  infoListItem: {
    fontSize: 13,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    lineHeight: 20,
    marginBottom: 4,
  },
  footer: {
    alignItems: 'center',
    marginTop: 40,
    paddingTop: 20,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.2)',
  },
  footerText: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.9,
  },
  footerSubtext: {
    fontSize: 12,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.7,
    marginTop: 4,
  },
  disclaimerCard: {
    backgroundColor: 'rgba(251, 191, 36, 0.15)',
    borderRadius: 20,
    padding: 20,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: 'rgba(251, 191, 36, 0.4)',
  },
  disclaimerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 16,
  },
  disclaimerTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  disclaimerText: {
    fontSize: 13,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.95,
    lineHeight: 20,
    marginBottom: 12,
  },
  metricsCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 20,
    padding: 20,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  metricsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 20,
  },
  metricsTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  overallMetric: {
    alignItems: 'center',
    paddingVertical: 20,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    marginBottom: 20,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  overallMetricLabel: {
    fontSize: 14,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    marginBottom: 8,
  },
  overallMetricValue: {
    fontSize: 36,
    fontWeight: '800' as const,
    color: Colors.text.primary,
    marginBottom: 4,
  },
  overallMetricSubtext: {
    fontSize: 12,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.7,
  },
  metricsSection: {
    marginBottom: 20,
  },
  metricsSectionTitle: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 16,
  },
  classMetricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  classMetricLeft: {
    flex: 1,
  },
  classMetricName: {
    fontSize: 15,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 4,
  },
  classMetricSupport: {
    fontSize: 12,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.7,
  },
  classMetricRight: {
    flexDirection: 'row',
    gap: 16,
    alignItems: 'center',
  },
  metricItem: {
    alignItems: 'center',
    minWidth: 70,
  },
  metricLabel: {
    fontSize: 11,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.8,
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 14,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  avgMetrics: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  avgMetricItem: {
    flex: 1,
    padding: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  avgMetricLabel: {
    fontSize: 12,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    marginBottom: 8,
  },
  avgMetricValue: {
    fontSize: 20,
    fontWeight: '800' as const,
    color: Colors.text.primary,
  },
});
