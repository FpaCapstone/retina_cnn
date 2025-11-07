import * as FileSystem from 'expo-file-system';
import { Platform } from 'react-native';
import { DiseaseType } from '@/types/disease';

/**
 * Convert frontend DiseaseType to backend format
 * Frontend: 'normal' | 'uveitis' | 'conjunctivitis' | 'cataract' | 'eyelid_drooping'
 * Backend: 'Normal' | 'Uveitis' | 'Conjunctivitis' | 'Cataract' | 'Eyelid Drooping'
 */
export function convertDiseaseTypeToBackend(disease: DiseaseType): 'Normal' | 'Uveitis' | 'Conjunctivitis' | 'Cataract' | 'Eyelid Drooping' {
  const mapping: Record<DiseaseType, 'Normal' | 'Uveitis' | 'Conjunctivitis' | 'Cataract' | 'Eyelid Drooping'> = {
    'normal': 'Normal',
    'uveitis': 'Uveitis',
    'conjunctivitis': 'Conjunctivitis',
    'cataract': 'Cataract',
    'eyelid_drooping': 'Eyelid Drooping',
  };
  return mapping[disease];
}

const getDatasetPath = () => {
  if (Platform.OS === 'web') {
    return 'datasets/';
  }
  const docDir = (FileSystem as any).documentDirectory || '';
  return `${docDir}datasets/`;
};

export const DATASET_PATH = getDatasetPath();

export interface DatasetStructure {
  glaucoma: string[];
  diabetic_retinopathy: string[];
  cataract: string[];
  normal: string[];
}

export interface DatasetInfo {
  totalImages: number;
  glaucomaCount: number;
  retinopathyCount: number;
  cataractCount: number;
  normalCount: number;
  path: string;
  subfolders: string[];
}

export async function ensureDatasetDirectory(): Promise<void> {
  try {
    const info = await FileSystem.getInfoAsync(DATASET_PATH);
    if (!info.exists) {
      console.log('[Dataset] Creating dataset directory at:', DATASET_PATH);
      await FileSystem.makeDirectoryAsync(DATASET_PATH, { intermediates: true });
      
      await FileSystem.makeDirectoryAsync(`${DATASET_PATH}glaucoma/`, { intermediates: true });
      await FileSystem.makeDirectoryAsync(`${DATASET_PATH}diabetic_retinopathy/`, { intermediates: true });
      await FileSystem.makeDirectoryAsync(`${DATASET_PATH}cataract/`, { intermediates: true });
      await FileSystem.makeDirectoryAsync(`${DATASET_PATH}normal/`, { intermediates: true });
      
      console.log('[Dataset] Dataset directories created successfully');
    }
  } catch (error) {
    console.error('[Dataset] Failed to create dataset directories:', error);
    throw error;
  }
}

export async function scanDatasetDirectory(): Promise<DatasetInfo> {
  try {
    console.log('[Dataset] Scanning dataset directory...');
    
    await ensureDatasetDirectory();
    
    const subfolders = await FileSystem.readDirectoryAsync(DATASET_PATH);
    
    let glaucomaCount = 0;
    let retinopathyCount = 0;
    let cataractCount = 0;
    let normalCount = 0;
    
    for (const subfolder of subfolders) {
      const subfolderPath = `${DATASET_PATH}${subfolder}/`;
      const info = await FileSystem.getInfoAsync(subfolderPath);
      
      if (info.exists && info.isDirectory) {
        const files = await FileSystem.readDirectoryAsync(subfolderPath);
        const imageFiles = files.filter(f => 
          f.endsWith('.jpg') || f.endsWith('.jpeg') || f.endsWith('.png')
        );
        
        const count = imageFiles.length;
        
        if (subfolder === 'glaucoma') {
          glaucomaCount = count;
        } else if (subfolder === 'diabetic_retinopathy') {
          retinopathyCount = count;
        } else if (subfolder === 'cataract') {
          cataractCount = count;
        } else if (subfolder === 'normal') {
          normalCount = count;
        }
      }
    }
    
    const totalImages = glaucomaCount + retinopathyCount + cataractCount + normalCount;
    
    console.log('[Dataset] Scan complete:');
    console.log(`  - Glaucoma: ${glaucomaCount} images`);
    console.log(`  - Diabetic Retinopathy: ${retinopathyCount} images`);
    console.log(`  - Cataract: ${cataractCount} images`);
    console.log(`  - Normal: ${normalCount} images`);
    console.log(`  - Total: ${totalImages} images`);
    
    return {
      totalImages,
      glaucomaCount,
      retinopathyCount,
      cataractCount,
      normalCount,
      path: DATASET_PATH,
      subfolders,
    };
  } catch (error) {
    console.error('[Dataset] Failed to scan dataset directory:', error);
    throw error;
  }
}

export async function addImageToDataset(
  imageUri: string,
  category: DiseaseType | 'normal'
): Promise<string> {
  try {
    console.log(`[Dataset] Adding image to ${category} category...`);
    
    await ensureDatasetDirectory();
    
    const categoryFolder = category === 'retinopathy' 
      ? 'diabetic_retinopathy' 
      : category;
    
    const timestamp = Date.now();
    const filename = `${timestamp}.jpg`;
    const destPath = `${DATASET_PATH}${categoryFolder}/${filename}`;
    
    await FileSystem.copyAsync({
      from: imageUri,
      to: destPath,
    });
    
    console.log('[Dataset] Image added successfully:', destPath);
    return destPath;
  } catch (error) {
    console.error('[Dataset] Failed to add image to dataset:', error);
    throw error;
  }
}

export async function getDatasetImagesForCategory(
  category: DiseaseType | 'normal'
): Promise<string[]> {
  try {
    const categoryFolder = category === 'retinopathy' 
      ? 'diabetic_retinopathy' 
      : category;
    
    const folderPath = `${DATASET_PATH}${categoryFolder}/`;
    const info = await FileSystem.getInfoAsync(folderPath);
    
    if (!info.exists) {
      return [];
    }
    
    const files = await FileSystem.readDirectoryAsync(folderPath);
    const imageFiles = files
      .filter(f => f.endsWith('.jpg') || f.endsWith('.jpeg') || f.endsWith('.png'))
      .map(f => `${folderPath}${f}`);
    
    return imageFiles;
  } catch (error) {
    console.error('[Dataset] Failed to get images for category:', error);
    return [];
  }
}

export function getDatasetInstructions(): string {
  const docDir = (FileSystem as any).documentDirectory || '';
  const modelPath = Platform.OS === 'web' ? 'models/eye_disease_cnn/model.json' : `${docDir}models/eye_disease_cnn/model.json`;
  return `
DATASET SETUP INSTRUCTIONS
==========================

To train your custom CNN model, you need to place your 700MB image dataset in the following structure:

Location: ${DATASET_PATH}

Required folder structure:
├── glaucoma/          (Place glaucoma eye images here)
├── diabetic_retinopathy/  (Place diabetic retinopathy images here)
├── cataract/          (Place cataract eye images here)
└── normal/            (Place healthy eye images here)

Image Requirements:
- Format: JPG, JPEG, or PNG
- Recommended size: At least 224x224 pixels
- File naming: Any name (will be processed automatically)

Steps to add your dataset:
1. Connect your device to a computer
2. Navigate to: ${DATASET_PATH}
3. Copy your images into the respective subfolders
4. Return to the app and scan the dataset

The app will automatically:
- Count images in each category
- Validate image formats
- Prepare data for training

Note: Training should be done offline using Python/TensorFlow
The trained model should be saved and placed in:
${modelPath}
  `.trim();
}
