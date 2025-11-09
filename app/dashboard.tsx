import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Animated } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { 
  Activity, 
  Database, 
  Microscope, 
  AlertCircle, 
  ScanEye,
  CheckCircle2,
  Image as ImageIcon,
  Sparkles,
  Brain,
  Shield,
  ArrowRight,
  Eye
} from 'lucide-react-native';
import { useRouter, Stack } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import Colors from '@/constants/colors';
import OfflineIndicator from '@/components/OfflineIndicator';

type PipelineStage = {
  number: string;
  title: string;
  description: string;
  icon: any;
  color: string;
};

const PIPELINE_STAGES: PipelineStage[] = [
  {
    number: '1️⃣',
    title: 'Image Quality AI',
    description: 'Detects blur, glare, or crop issues',
    icon: ImageIcon,
    color: Colors.status.info,
  },
  {
    number: '2️⃣',
    title: 'Preprocessing Enhancer',
    description: 'Fixes contrast, sharpness, or alignment',
    icon: Sparkles,
    color: Colors.primary.teal,
  },
  {
    number: '3️⃣',
    title: 'Normal-vs-Abnormal AI',
    description: 'Quickly filters healthy eyes',
    icon: Eye,
    color: Colors.status.success,
  },
  {
    number: '4️⃣',
    title: 'Disease Classifier',
    description: 'Classifies 5 conditions (Retina CNN)',
    icon: Brain,
    color: Colors.primary.purple,
  },
  {
    number: '5️⃣',
    title: 'Confidence/Consistency Validator',
    description: 'Checks confidence & consistency before final output',
    icon: Shield,
    color: Colors.status.warning,
  },
];

export default function DashboardScreen() {
  const router = useRouter();
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;
  const [showPipeline, setShowPipeline] = useState(true);

  useEffect(() => {
    fadeAnim.setValue(0);
    slideAnim.setValue(50);
    
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 800,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const handleGoToHome = () => {
    router.push('/home');
  };

  const handleGoToDetection = () => {
    router.push('/detect');
  };

  const handleGoToTraining = () => {
    router.push('/training');
  };

  const handleGoToInfo = () => {
    router.push('/home');
    // Navigate to info tab after a short delay
    setTimeout(() => {
      // This will be handled by the home screen
    }, 100);
  };

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
      
      <View style={styles.safeArea}>
        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <Animated.View style={[styles.header, { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }]}>
            <View style={styles.logoContainer}>
              <Activity size={48} color={Colors.text.primary} strokeWidth={2.5} />
            </View>
            <Text style={styles.title}>RETINA</Text>
            <Text style={styles.subtitle}>Real-Time Eye-Disease Testing</Text>
            <Text style={styles.subtitleSecondary}>with Intelligent Neural Analysis</Text>
          </Animated.View>

          {/* Enhanced Pipeline Section */}
          <Animated.View style={[styles.pipelineSection, { opacity: fadeAnim }]}>
            <View style={styles.pipelineHeader}>
              <Sparkles size={28} color={Colors.text.primary} strokeWidth={2} />
              <Text style={styles.pipelineTitle}>Enhanced 5-Stage Pipeline</Text>
            </View>
            <Text style={styles.pipelineDescription}>
              Our advanced AI system uses a multi-stage approach to ensure accurate eye disease detection from camera images.
            </Text>

            <View style={styles.stagesContainer}>
              {PIPELINE_STAGES.map((stage, index) => (
                <Animated.View
                  key={index}
                  style={[
                    styles.stageCard,
                    {
                      opacity: fadeAnim,
                      transform: [
                        {
                          translateX: slideAnim.interpolate({
                            inputRange: [0, 50],
                            outputRange: [0, -20],
                          }),
                        },
                      ],
                    },
                  ]}
                >
                  <View style={[styles.stageIconContainer, { backgroundColor: `${stage.color}20` }]}>
                    <stage.icon size={24} color={stage.color} strokeWidth={2.5} />
                  </View>
                  <View style={styles.stageContent}>
                    <View style={styles.stageHeader}>
                      <Text style={styles.stageNumber}>{stage.number}</Text>
                      <Text style={styles.stageTitle}>{stage.title}</Text>
                    </View>
                    <Text style={styles.stageDescription}>{stage.description}</Text>
                  </View>
                </Animated.View>
              ))}
            </View>
          </Animated.View>

          {/* Navigation Options */}
          <Animated.View style={[styles.navigationSection, { opacity: fadeAnim }]}>
            <Text style={styles.navigationTitle}>What would you like to do?</Text>
            
            <TouchableOpacity
              style={styles.navCard}
              onPress={handleGoToDetection}
              activeOpacity={0.8}
            >
              <View style={[styles.navIconCircle, { backgroundColor: `${Colors.primary.purple}30` }]}>
                <ScanEye size={32} color={Colors.primary.purple} strokeWidth={2.5} />
              </View>
              <View style={styles.navContent}>
                <Text style={styles.navTitle}>Detect Eye Diseases</Text>
                <Text style={styles.navDescription}>
                  Scan eye images to detect Normal, Uveitis, Conjunctivitis, Cataract, and Eyelid Drooping
                </Text>
              </View>
              <ArrowRight size={24} color={Colors.text.primary} opacity={0.7} />
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.navCard}
              onPress={handleGoToTraining}
              activeOpacity={0.8}
            >
              <View style={[styles.navIconCircle, { backgroundColor: `${Colors.primary.teal}30` }]}>
                <Database size={32} color={Colors.primary.teal} strokeWidth={2.5} />
              </View>
              <View style={styles.navContent}>
                <Text style={styles.navTitle}>Training Data</Text>
                <Text style={styles.navDescription}>
                  Upload verified patient images to improve AI accuracy
                </Text>
              </View>
              <ArrowRight size={24} color={Colors.text.primary} opacity={0.7} />
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.navCard}
              onPress={handleGoToHome}
              activeOpacity={0.8}
            >
              <View style={[styles.navIconCircle, { backgroundColor: `${Colors.primary.green}30` }]}>
                <AlertCircle size={32} color={Colors.primary.green} strokeWidth={2.5} />
              </View>
              <View style={styles.navContent}>
                <Text style={styles.navTitle}>Model Information</Text>
                <Text style={styles.navDescription}>
                  View performance metrics, dataset info, and medical disclaimer
                </Text>
              </View>
              <ArrowRight size={24} color={Colors.text.primary} opacity={0.7} />
            </TouchableOpacity>
          </Animated.View>

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
    paddingTop: 20,
    paddingBottom: 30,
  },
  logoContainer: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    borderWidth: 3,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  title: {
    fontSize: 52,
    fontWeight: '800' as const,
    color: Colors.text.primary,
    letterSpacing: 3,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 18,
    fontWeight: '600' as const,
    color: Colors.text.primary,
    opacity: 0.95,
    textAlign: 'center',
  },
  subtitleSecondary: {
    fontSize: 15,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.85,
    textAlign: 'center',
    marginTop: 4,
  },
  pipelineSection: {
    marginBottom: 32,
  },
  pipelineHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  pipelineTitle: {
    fontSize: 22,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  pipelineDescription: {
    fontSize: 14,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.9,
    lineHeight: 20,
    marginBottom: 24,
  },
  stagesContainer: {
    gap: 16,
  },
  stageCard: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 16,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    alignItems: 'center',
  },
  stageIconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  stageContent: {
    flex: 1,
  },
  stageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  stageNumber: {
    fontSize: 18,
    fontWeight: '700' as const,
  },
  stageTitle: {
    fontSize: 16,
    fontWeight: '700' as const,
    color: Colors.text.primary,
  },
  stageDescription: {
    fontSize: 13,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.85,
    lineHeight: 18,
  },
  navigationSection: {
    marginBottom: 32,
  },
  navigationTitle: {
    fontSize: 20,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 20,
    textAlign: 'center',
  },
  navCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 20,
    padding: 20,
    marginBottom: 16,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  navIconCircle: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  navContent: {
    flex: 1,
  },
  navTitle: {
    fontSize: 18,
    fontWeight: '700' as const,
    color: Colors.text.primary,
    marginBottom: 6,
  },
  navDescription: {
    fontSize: 13,
    fontWeight: '500' as const,
    color: Colors.text.primary,
    opacity: 0.85,
    lineHeight: 18,
  },
  footer: {
    alignItems: 'center',
    marginTop: 20,
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
});

