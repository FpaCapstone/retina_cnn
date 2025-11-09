export type DiseaseType = 'normal' | 'uveitis' | 'conjunctivitis' | 'cataract' | 'eyelid_drooping';

export interface DiseaseInfo {
  id: DiseaseType;
  name: string;
  description: string;
  fullDescription: string;
  symptoms: string[];
  riskFactors: string[];
  color: string;
  icon: string;
}

export interface DiseaseDetection {
  disease: DiseaseType;
  confidence: number;
  percentage: number;
}

export interface AnalysisResult {
  detections: DiseaseDetection[];
  primaryDisease: DiseaseType;
  timestamp: string;
  imageUri: string;
  details: string;
  usedModel?: 'backend' | 'enhanced' | 'tflite' | 'offline';
  modelInfo?: string;
}

export interface TrainingImage {
  id: string;
  uri: string;
  disease: DiseaseType;
  uploadedAt: string;
}
