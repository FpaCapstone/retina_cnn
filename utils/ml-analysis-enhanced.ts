/**
 * Enhanced 5-Stage Pipeline Analysis
 * 
 * Improves accuracy for camera-captured images through:
 * 1. Image Quality AI - Detects blur, glare, crop issues
 * 2. Preprocessing Enhancer - Fixes contrast, sharpness, alignment
 * 3. Normal-vs-Abnormal AI - Quickly filters healthy eyes
 * 4. Disease Classifier - Classifies 5 conditions
 * 5. Confidence/Consistency Validator - Checks confidence & consistency
 */

import { DiseaseType, AnalysisResult, DiseaseDetection } from '@/types/disease';
import { trpcClient } from '@/lib/trpc';
import { predictWithTFLite, checkTFLiteModelAvailable } from '@/utils/tflite-model';

const DISEASES: DiseaseType[] = ['normal', 'uveitis', 'conjunctivitis', 'cataract', 'eyelid_drooping'];

export interface EnhancedAnalysisResult extends AnalysisResult {
  qualityScore?: number;
  recommendation?: 'accept' | 'retake';
  qualityIssues?: string[];
  preprocessingApplied?: string[];
  normalFilterResult?: {
    isNormal: boolean;
    confidence: number;
  };
  validationResult?: {
    finalConfidence: number;
    isReliable: boolean;
    reasons?: string[];
  };
  stages?: {
    quality?: any;
    preprocessing?: any;
    normal_filter?: any;
    disease_classification?: any;
    validation?: any;
  };
}

export interface EnhancedPipelineOptions {
  enableQualityCheck?: boolean;
  enablePreprocessing?: boolean;
  enableNormalFilter?: boolean;
  enableDiseaseClassification?: boolean;
  enableValidation?: boolean;
}

/**
 * Analyze eye image using the enhanced 5-stage pipeline
 */

export async function analyzeEyeImageEnhanced(
  imageUri: string,
  options: EnhancedPipelineOptions = {}
): Promise<EnhancedAnalysisResult> {
  console.log('[Enhanced Analysis] Starting enhanced 5-stage pipeline analysis...');

  const defaultOptions: Required<EnhancedPipelineOptions> = {
    enableQualityCheck: true,
    enablePreprocessing: true,
    enableNormalFilter: true,
    enableDiseaseClassification: true,
    enableValidation: true,
  };

  const pipelineOptions = { ...defaultOptions, ...options };

  // Try backend enhanced pipeline first
  try {
    console.log('[Enhanced Analysis] Attempting backend enhanced pipeline...');
    const backendResult = await Promise.race([
      trpcClient.detection.analyzeEnhanced.mutate({
        imageUri,
        enableStages: {
          quality_check: pipelineOptions.enableQualityCheck,
          preprocessing: pipelineOptions.enablePreprocessing,
          normal_filter: pipelineOptions.enableNormalFilter,
          disease_classification: pipelineOptions.enableDiseaseClassification,
          validation: pipelineOptions.enableValidation,
        },
      }),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Backend timeout')), 10000)
      ),
    ]) as any;

    if (backendResult && backendResult.prediction) {
      console.log('[Enhanced Analysis] Backend enhanced pipeline successful:', backendResult);

      // Convert backend result to frontend format
      let backendDisease = backendResult.prediction.toLowerCase();
      if (backendDisease === 'eyelid drooping') {
        backendDisease = 'eyelid_drooping';
      } else {
        backendDisease = backendDisease.replace(' ', '_');
      }
      const diseaseType = backendDisease as DiseaseType;
      const confidence = backendResult.final_confidence || backendResult.confidence || 0.85;

      // Build detections from probabilities
      const probabilities = backendResult.all_probabilities || {};
      const detections: DiseaseDetection[] = DISEASES.map((disease) => {
        // Map disease names
        let backendDiseaseName = disease;
        if (disease === 'eyelid_drooping') {
          backendDiseaseName = 'Eyelid Drooping';
        } else {
          backendDiseaseName = disease.charAt(0).toUpperCase() + disease.slice(1);
        }

        const prob = probabilities[backendDiseaseName] || 
                    probabilities[disease] || 
                    (disease === diseaseType ? confidence : (1 - confidence) / (DISEASES.length - 1));
        
        return {
          disease,
          confidence: prob,
          percentage: prob * 100,
        };
      }).sort((a, b) => b.confidence - a.confidence).slice(0, 3);

      const diseaseDescriptions: Record<DiseaseType, { description: string }> = {
        normal: {
          description: 'No eye disease detected. Clear vision with healthy appearance and no redness or swelling. Continue routine eye care.',
        },
        uveitis: {
          description: 'Signs consistent with uveitis detected: redness, pain, light sensitivity, and possible floaters. Inflammatory response likely affecting intraocular structures. Ophthalmic evaluation recommended.',
        },
        conjunctivitis: {
          description: 'Findings consistent with conjunctivitis: conjunctival redness with tearing/itching and possible discharge or eyelid crusting. Consider hygiene and clinical consultation if symptoms persist.',
        },
        cataract: {
          description: 'Lens opacity patterns suggest cataract changes leading to cloudy/blurred vision and glare sensitivity. Consider ophthalmology consult for staging and management.',
        },
        eyelid_drooping: {
          description: 'External features indicate eyelid drooping with possible swelling/irritation and palpable eyelid lumps. Evaluate for ptosis or blepharitis/chalazion. Clinical assessment advised.',
        },
      };

      // Add quality and recommendation messages
      let details = diseaseDescriptions[diseaseType]?.description || 'Analysis complete.';
      
      if (backendResult.quality_score !== undefined) {
        details += `\n\n📊 Image Quality Score: ${(backendResult.quality_score * 100).toFixed(1)}%`;
      }

      if (backendResult.recommendation === 'retake') {
        details += `\n⚠️ Recommendation: Please retake the image for better accuracy.`;
        if (backendResult.stages?.quality?.issues?.length > 0) {
          details += `\nIssues detected: ${backendResult.stages.quality.issues.join(', ')}`;
        }
      } else if (backendResult.recommendation === 'accept') {
        details += `\n✅ Image quality is acceptable for analysis.`;
      }

      return {
        detections,
        primaryDisease: diseaseType,
        timestamp: new Date().toISOString(),
        imageUri,
        details,
        usedModel: 'enhanced',
        modelInfo: backendResult.quality_score 
          ? `Enhanced 5-Stage Pipeline | Quality: ${(backendResult.quality_score * 100).toFixed(1)}%`
          : 'Enhanced 5-Stage Pipeline (Online)',
        qualityScore: backendResult.quality_score,
        recommendation: backendResult.recommendation,
        qualityIssues: backendResult.stages?.quality?.issues || [],
        preprocessingApplied: backendResult.stages?.preprocessing?.preprocessing_applied || [],
        normalFilterResult: backendResult.stages?.normal_filter ? {
          isNormal: backendResult.stages.normal_filter.result === 'normal',
          confidence: backendResult.stages.normal_filter.confidence,
        } : undefined,
        validationResult: backendResult.stages?.validation ? {
          finalConfidence: backendResult.stages.validation.final_confidence,
          isReliable: backendResult.stages.validation.is_reliable,
          reasons: backendResult.stages.validation.reasons || [],
        } : undefined,
        stages: backendResult.stages,
      };
    }
  } catch (backendError) {
    console.log('[Enhanced Analysis] Backend enhanced pipeline not available, falling back to standard analysis:', backendError);
    
    // Fallback to standard analysis
    try {
      const standardResult = await trpcClient.detection.analyze.mutate({ imageUri });
      if (standardResult && standardResult.prediction) {
        // Convert to enhanced format with minimal info
        let backendDisease = standardResult.prediction.toLowerCase();
        if (backendDisease === 'eyelid drooping') {
          backendDisease = 'eyelid_drooping';
        } else {
          backendDisease = backendDisease.replace(' ', '_');
        }
        const diseaseType = backendDisease as DiseaseType;
        const confidence = standardResult.confidence || 0.85;

        const detections: DiseaseDetection[] = [
          {
            disease: diseaseType,
            confidence,
            percentage: confidence * 100,
          },
          ...DISEASES.filter(d => d !== diseaseType).map(d => ({
            disease: d,
            confidence: (1 - confidence) / (DISEASES.length - 1),
            percentage: ((1 - confidence) / (DISEASES.length - 1)) * 100,
          })),
        ].slice(0, 3);

        const diseaseDescriptions: Record<DiseaseType, { description: string }> = {
          normal: {
            description: 'No eye disease detected. Clear vision with healthy appearance and no redness or swelling. Continue routine eye care.',
          },
          uveitis: {
            description: 'Signs consistent with uveitis detected: redness, pain, light sensitivity, and possible floaters. Inflammatory response likely affecting intraocular structures. Ophthalmic evaluation recommended.',
          },
          conjunctivitis: {
            description: 'Findings consistent with conjunctivitis: conjunctival redness with tearing/itching and possible discharge or eyelid crusting. Consider hygiene and clinical consultation if symptoms persist.',
          },
          cataract: {
            description: 'Lens opacity patterns suggest cataract changes leading to cloudy/blurred vision and glare sensitivity. Consider ophthalmology consult for staging and management.',
          },
          eyelid_drooping: {
            description: 'External features indicate eyelid drooping with possible swelling/irritation and palpable eyelid lumps. Evaluate for ptosis or blepharitis/chalazion. Clinical assessment advised.',
          },
        };

        return {
          detections,
          primaryDisease: diseaseType,
          timestamp: new Date().toISOString(),
          imageUri,
          details: diseaseDescriptions[diseaseType]?.description || 'Analysis complete.',
          usedModel: 'backend',
          modelInfo: 'Backend AI Model (Online)',
          recommendation: 'accept', // Default when using standard analysis
        };
      }
    } catch (standardError) {
      console.log('[Enhanced Analysis] Standard analysis also failed, trying TFLite fallback:', standardError);
    }
  }

  // Final fallback: TFLite model
  try {
    const tfliteAvailable = await checkTFLiteModelAvailable();
    if (tfliteAvailable) {
      console.log('[Enhanced Analysis] Using TFLite model fallback...');
      const tfliteResult = await predictWithTFLite(imageUri);

      const diseaseDescriptions: Record<DiseaseType, { description: string }> = {
        normal: {
          description: 'No eye disease detected. Clear vision with healthy appearance and no redness or swelling. Continue routine eye care.',
        },
        uveitis: {
          description: 'Signs consistent with uveitis detected: redness, pain, light sensitivity, and possible floaters. Inflammatory response likely affecting intraocular structures. Ophthalmic evaluation recommended.',
        },
        conjunctivitis: {
          description: 'Findings consistent with conjunctivitis: conjunctival redness with tearing/itching and possible discharge or eyelid crusting. Consider hygiene and clinical consultation if symptoms persist.',
        },
        cataract: {
          description: 'Lens opacity patterns suggest cataract changes leading to cloudy/blurred vision and glare sensitivity. Consider ophthalmology consult for staging and management.',
        },
        eyelid_drooping: {
          description: 'External features indicate eyelid drooping with possible swelling/irritation and palpable eyelid lumps. Evaluate for ptosis or blepharitis/chalazion. Clinical assessment advised.',
        },
      };

      return {
        detections: tfliteResult.detections,
        primaryDisease: tfliteResult.primaryDisease,
        timestamp: new Date().toISOString(),
        imageUri,
        details: diseaseDescriptions[tfliteResult.primaryDisease]?.description || 'Analysis complete.',
        usedModel: 'tflite',
        modelInfo: 'On-Device AI Model (Offline)',
        recommendation: 'accept',
      };
    }
  } catch (tfliteError) {
    console.log('[Enhanced Analysis] TFLite not available:', tfliteError);
  }

  // Ultimate fallback: throw error
  throw new Error('All analysis methods failed. Please check your connection and try again.');
}

